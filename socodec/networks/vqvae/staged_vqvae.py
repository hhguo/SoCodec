from torch import nn
from torch.nn import functional as F

import numpy as np
import random
import torch

from .vector_quantization import VectorQuantization
from ..common.ecapa_tdnn import ECAPA_TDNN as SpeakerEncoder
from ...utils.audio import TorchMelSpectrogram


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        # self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.gn = nn.Identity()

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]
    
    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[...,:-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=32):
        super().__init__()
        
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, groups=groups),
            GroupNorm(out_channels),
            Swish(),
            CausalConv1d(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=kernel_size, groups=groups)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels,
                         out_channels=out_channels),
            GroupNorm(out_channels),
            Swish(),
            ResidualUnit(in_channels=out_channels,
                         out_channels=out_channels),
            GroupNorm(out_channels),
            Swish(),
            CausalConv1d(in_channels=out_channels,
                         out_channels=out_channels,
                         kernel_size=2 * stride,
                         stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        out_channels = in_channels # * 2
        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=2*stride, stride=stride),
            GroupNorm(out_channels),
            Swish(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            GroupNorm(out_channels),
            Swish(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D, strides=[2, 2]):
        super().__init__()
        self.downsample_scale = np.cumprod(np.asarray(strides))[-1]
        self.layers = [
            CausalConv1d(in_channels=C, out_channels=D, kernel_size=3),
            Swish(),
        ]
        for stride in strides:
            self.layers += [
                EncoderBlock(out_channels=D, stride=stride),
                GroupNorm(D),
                Swish(),
            ]
        self.layers += [
            CausalConv1d(in_channels=D, out_channels=D, kernel_size=3)
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x.transpose(1, 2)).transpose(1, 2)


class Decoder(nn.Module):
    def __init__(self, C, D, strides=[2, 2]):
        super().__init__()
        self.layers = [
            CausalConv1d(in_channels=D, out_channels=D, kernel_size=3),
            Swish()
        ]
        for stride in strides:
            self.layers += [
                DecoderBlock(in_channels=D, stride=stride),
                GroupNorm(D),
                Swish()
            ]
        self.layers += [
            CausalConv1d(in_channels=D, out_channels=C, kernel_size=3)
        ]
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x, g=None):
        x = x.transpose(1, 2)
        if g is not None:
            up_g = g.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = x + up_g
        
        h = self.layers[: -1](x)
        y = self.layers[-1](h)

        return y.transpose(1, 2), h.transpose(1, 2)


class TimeRegulator(nn.Module):

    def __init__(self, in_dim, scale, learnable=False):
        super().__init__()
        self.scale = scale
        self.learnable = learnable
        if learnable:
            self.downsampler = CausalConv1d(in_channels=in_dim,
                                            out_channels=in_dim,
                                            kernel_size=2 * scale,
                                            stride=scale)
            self.upsampler = CausalConvTranspose1d(in_channels=in_dim,
                                                out_channels=in_dim,
                                                kernel_size=2 * scale,
                                                stride=scale)
    
    def forward(self, x, x_len, downsample=True):
        if downsample:
            x = self.downsample(x, x_len)
        else:
            x = self.upsample(x, x_len)
        return x

    def downsample(self, x, x_len):
        if self.learnable:
            x = self.downsampler(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = torch.nn.functional.avg_pool1d(
                x.transpose(1, 2), self.scale, stride=self.scale,
                ceil_mode=True).transpose(1, 2)
        x_len = (x_len / self.scale).ceil()
        return x, x_len

    def upsample(self, x, x_len):
        if self.learnable:
            x = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = torch.repeat_interleave(x, self.scale, dim=1)
        return x


class TreeVectorQuantization(nn.Module):
    
    def __init__(self,
                 in_dim,
                 vq_class='VectorQuantization',
                 vq_config={},
                 tree_config={},
                 ):
        super().__init__()
        self.vq_config = vq_config
        self.tree_config = tree_config

        self.quantizers = nn.ModuleList()
        self.time_regulators = nn.ModuleList()
        for config in self.tree_config:
            vq_config = self.vq_config.copy()
            if not isinstance(vq_config['codebook_size'], (tuple, list)):
                vq_config['codebook_size'] = [vq_config['codebook_size']]
                vq_config['codebook_dim'] = [vq_config['codebook_dim']]
            vq_config['codebook_size'] = vq_config['codebook_size'] * config['n_groups']
            vq_config['codebook_dim'] = vq_config['codebook_dim'] * config['n_groups']
            self.quantizers.append(VectorQuantization(in_dim,
                                                      n_groups=config.get('n_groups', 1),
                                                      dropout_rate_per_group=config.get('dropout_rate_per_group', 0),
                                                      ordered=config.get('ordered', False),
                                                      **vq_config))
            self.time_regulators.append(
                TimeRegulator(in_dim, config['downsample_rate'], config.get('learnable_time_regulator', False))
            )

    def forward(self, inp, inp_len, enable_vq=True, update_codebook=True, return_pre_quant=False):
        output, (quants, losses, embed_inds) = self.quantize(inp, inp_len,
                                                             enable_vq=enable_vq,
                                                             update_codebook=update_codebook,
                                                             return_pre_quant=return_pre_quant)
        loss = sum(losses) / len(losses)
        return output, (quants, loss, embed_inds)

    def quantize(self, inp, inp_len, enable_vq=True, update_codebook=True, return_pre_quant=False):
        quants, losses, embed_inds = [], [], []

        pre_quant_output, quant_output, residual = 0, 0, inp
        for tree_config, quantizer, regulator in zip(self.tree_config, self.quantizers, self.time_regulators):
            # Downsample
            x, x_len = regulator(residual, inp_len, True)

            # Quantization
            q, diff, embed_ind = quantizer(x, x_len,
                                           enable_vq=enable_vq,
                                           update_codebook=update_codebook,
                                           return_pre_quant=return_pre_quant)
            if return_pre_quant:
                pq, q = q

            # Upsample
            x = regulator(q, x_len, False)[:, : residual.shape[1]]

            residual = residual - x
            quant_output = quant_output + x

            if return_pre_quant:
                pq = regulator(pq, x_len, False)[:, : residual.shape[1]]
                pre_quant_output = pre_quant_output + pq
            
            quants.append(q)
            losses.append(diff)
            embed_inds.append(embed_ind)

        if return_pre_quant:
            return (pre_quant_output, quant_output), (quants, losses, embed_inds)
        return quant_output, (quants, losses, embed_inds)
    
    def decode(self, seqs, seq_lens=None):
        if not isinstance(seqs, (tuple, list)):
            tokens, token_lens = self.deserialize(seqs, seq_lens)
        else:
            tokens, token_lens = seqs, seq_lens

        quant_output = 0
        for token, quantizer, regulator in zip(tokens, self.quantizers, self.time_regulators):
            x = quantizer.decode(token).transpose(1, 2)
            x = regulator(x, None, False)
            if torch.is_tensor(quant_output):
                x = x[:, : quant_output.size(1)]
            quant_output = quant_output + x

        return quant_output, token_lens

    def serialize(self, tokens, token_lens):
        assert len(tokens) <= 2, 'we only support 1 or 2-scale sequences now...'
        
        scale = self.tree_config[0]['downsample_rate']
        token_lens = ((token_lens.float() / scale).ceil() * scale).int()
        
        seq1 = tokens[0].unsqueeze(-1)
        
        if len(tokens) == 1:
            seq_cat = seq1.view(seq1.shape[0], -1)
            seq_cat_lens = (token_lens / scale * seq1.shape[2]).int()
        elif len(tokens) == 2:
            seq2 = F.pad(tokens[1], (0, token_lens.max() - tokens[1].size(1)), 'replicate')
            seq2 = torch.stack([seq2[:, i:: scale] for i in range(scale)], dim=-1)
            seq_cat = torch.cat((seq1, seq2), dim=-1).view(seq1.shape[0], -1)
            seq_cat_lens = (token_lens / scale + token_lens).int()
            
        return seq_cat, seq_cat_lens

    def deserialize(self, seqs, seq_lens):
        if len(self.tree_config) == 1:
            return [seqs], seq_lens

        max_scale = max(config['downsample_rate'] for config in self.tree_config)
        total_scale = sum(config['downsample_rate'] for config in self.tree_config)
        
        # Cut for aligning
        if seq_lens is None:
            seq_lens = torch.full([seqs.shape[0]], seqs.shape[1]).to(seqs.device)
        seq_lens = (seq_lens / total_scale).int() * total_scale
        token_lens = (seq_lens / total_scale).int() * max_scale
        seqs = seqs[:, : seq_lens.max()]

        # Separate
        tokens = torch.stack([seqs[:, i:: total_scale] for i in range(total_scale)], dim=-1)
        seq1 = tokens[..., 0]
        seq2 = tokens[..., 1:].contiguous().view(tokens.shape[0], -1)
        
        return [seq1, seq2], token_lens


class StagedVQVAE(nn.Module):
    """ Transformer-based VQ-VAE model """

    def __init__(self,
                 in_dim,
                 out_dim,
                 n_model_size,
                 downsample_scales=[1, 2],
                 upsample_scales=[[2, 1], [2, 1]],
                 mel_config={},
                 ssl_config={},
                 # Quantization
                 vq_class='VectorQuantization',
                 vq_config={},
                 tree_config={},
                 # Speaker
                 speaker_version='v1',
                 speaker_embeddding_model='ECAPA-TDNN',
                 speaker_embedding_ckpt=None,
                 speaker_vq_config=None,
                 # Training
                 dual_decoding=False,
                 n_samples_per_token=640,
                 ):
        super(StagedVQVAE, self).__init__()
        self.in_dim = in_dim
        self.n_model_size = n_model_size
        self.mel_config = mel_config
        self.dual_decoding = dual_decoding
        self.vq_config = vq_config
        self.tree_config = tree_config
        self.output_feature = 'mel'
        self.n_samples_per_token = n_samples_per_token

        self.mel_spectrogram = TorchMelSpectrogram(**mel_config)
        
        from ...utils.hubert import HuBERT
        self.ssl_extractor = HuBERT(**ssl_config)
        for name, param in self.ssl_extractor.named_parameters():
            param.requires_grad = False

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(out_dim, n_model_size,
                                              channels=[256, 256, 256, 256, 768],
                                              kernel_sizes=[5, 3, 3, 3, 1],
                                              dilations=[1, 2, 3, 4, 1],
                                              attention_channels=64,
                                              res2net_scale=2,
                                              se_channels=64,
                                              global_context=True,
                                              batch_norm=False)

        # Encoder & decoder
        self.encoder = Encoder(in_dim, n_model_size, downsample_scales)
        self.decoder_1 = Decoder(in_dim, n_model_size, upsample_scales[0])
        self.decoder_2 = Decoder(out_dim, n_model_size, upsample_scales[1])

        # Quantization
        self.quantizer = TreeVectorQuantization(
            n_model_size,
            vq_class=vq_class,
            vq_config=vq_config,
            tree_config=tree_config
        )

    def forward(self, wav, wav_length, enable_vq=True, decode=True, extract_spk=True, shuffle=False):
        output_dict = {}

        with torch.no_grad():
            # Pad waveform
            if wav.shape[1] % self.n_samples_per_token > 0:
                pad_size = self.n_samples_per_token - wav.shape[1] % self.n_samples_per_token
                wav = F.pad(wav, (0, pad_size), value=0)
                wav_length += pad_size

            # Extract mel & sll
            mel, mel_length = self.mel_spectrogram(wav, wav_length)
            output_dict.update({'mel': mel, 'mel_length': mel_length})

            ssl, ssl_length = self.ssl_extractor(wav, wav_length)
            output_dict.update({'ssl': ssl.float(), 'ssl_length': ssl_length})
        
        input, input_length = ssl, ssl_length
        output, output_length = mel, mel_length 

        encoder_outputs = self.encoder(input)
        quant_length = torch.ceil(input_length / self.encoder.downsample_scale)
        quant_length = quant_length.clamp(max=encoder_outputs.shape[1])

        quant, (quants, diff, embed_ind) = self.quantizer(encoder_outputs, quant_length,
                                                          enable_vq=enable_vq,
                                                          update_codebook=True,
                                                          return_pre_quant=self.dual_decoding)

        output_dict.update({
            'quants': quants,
            'token': embed_ind,
            'token_length': quant_length.int(),
            'encoder_diffs': diff,
        })

        # Speaker
        if extract_spk:
            # Enable shuffle in training or explicitly
            if self.training or shuffle:
                cond, cond_length = self._random_clip(output, output_length)
                speaker_embedding = self.speaker_encoder(cond, cond_length)
            else:
                speaker_embedding = self.speaker_encoder(output, output_length)
            output_dict['spk'] = speaker_embedding
            speaker_embedding_1 = speaker_embedding_2 = speaker_embedding

        if decode:
            ssl_pred, h = self.decoder_1(quant, speaker_embedding_1)
            mel_pred, h = self.decoder_2(h, speaker_embedding_2)
        
            assert ssl_pred.shape[1] == ssl.shape[1], '{}, {}'.format(ssl_pred.shape[1], ssl.shape[1])
            assert mel_pred.shape[1] == mel.shape[1], '{}, {}'.format(mel_pred.shape[1], mel.shape[1])

            output_dict.update({'ssl_pred': ssl_pred})
            output_dict.update({'mel_pred': mel_pred})
        
        return output_dict

    @torch.no_grad()
    def extract_speech_tokens(self, wav, wav_length, serialize=True, extract_spk=True, shuffle=False):
        output_dict = self.forward(wav, wav_length, True, False, extract_spk=extract_spk, shuffle=shuffle)
        token_seqs, token_length = output_dict['token'], output_dict['token_length']

        # Align sequences
        scale = self.tree_config[0]['downsample_rate']
        token_length = (torch.ceil(token_length / scale) * scale).int()

        new_token_seqs, new_token_lens = [], []
        for i, token_seq in enumerate(token_seqs):
            # discrete-continuous tokens
            residual = None
            if isinstance(token_seq, (tuple, list)):
                token_seq, residual = token_seq

            scale = self.tree_config[i]['downsample_rate']
            new_token_len = token_length // scale
            pad = int(new_token_len.max()) - token_seq.shape[1]
            token_seq = token_seq.float()
            # WAR: fix "RuntimeError: "replication_pad2d_cuda" not implemented for 'Long'"
            token_seq = F.pad(token_seq,
                              (0, pad) if len(token_seq.shape) == 2 else (0, 0, 0, pad),
                              'replicate')
            token_seq = token_seq.long()

            if residual is not None:
                token_seq = (token_seq, residual)
            new_token_seqs.append(token_seq)
            new_token_lens.append(new_token_len)
        
        if len(new_token_seqs) == 1:
            new_token_seqs, new_token_lens = new_token_seqs[0], new_token_lens[0]
        elif serialize:
            new_token_seqs, new_token_lens = self.quantizer.serialize(new_token_seqs, new_token_lens)

        output_dict.update({
            'embed': output_dict['quants'],
            'token': new_token_seqs,
            'token_length': new_token_lens,
        })

        return output_dict

    @torch.no_grad()
    def reconstruct_mel(self, token, spk):
        quant, _ = self.quantizer.decode(token, None)
        speaker_embedding_1 = speaker_embedding_2 = spk
        ssl_pred, h = self.decoder_1(quant, speaker_embedding_1)
        mel_pred, h = self.decoder_2(h, speaker_embedding_2)
        output_dict = {
            'ssl_pred': ssl_pred,
            'mel_pred': mel_pred,
        }
        return output_dict

    @torch.no_grad()
    def code_to_latent(self, token, mel=None):
        quant, _ = self.quantizer.decode(token, None)
        speaker_embedding = self.speaker_encoder(mel)  
        latents = quant + speaker_embedding.unsqueeze(1).repeat(1, quant.shape[1], 1)
        return {
            'latents': latents,
        }
    
    def _random_clip(self, sequences, lengths, max_ratio=0.75, min_ratio=0.25, n_segments=3, min_length=100):
        truncated_lengths = (
            lengths * (torch.rand_like(lengths.float()) * (max_ratio - min_ratio) + min_ratio)
        ).long()
        min_length = max(min_length, truncated_lengths.max())

        new_sequences, new_lengths = [], []
        for seq, org_len, new_len in zip(sequences, lengths, truncated_lengths):
            # Clip
            start = random.randint(0, org_len - new_len)
            seg = seq[start : start + int(new_len)]

            # Shuffle
            segment_length = seg.shape[0] // n_segments
            seg = seg[: seg.shape[0] // n_segments * n_segments]
            slices = [
                seg[i: i + segment_length]
                for i in range(0, seg.shape[0], segment_length)
            ]
            random.shuffle(slices)
            seg = torch.cat(slices, dim=0)
            
            if seg.shape[0] < min_length:
                seg = torch.cat([seg] * (min_length // seg.shape[0] + 1))[: min_length]
            new_sequences.append(seg)
            new_lengths.append(new_len)

        new_sequences = torch.stack(new_sequences, dim=0)
        new_lengths = torch.tensor(new_lengths, device=new_sequences.device)
        return new_sequences, new_lengths