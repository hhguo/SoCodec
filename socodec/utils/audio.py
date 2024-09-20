from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F

import math
import numpy as np
import torch
import torchaudio


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class TorchMelSpectrogram(nn.Module):

    def __init__(self,
                 filter_length=1024,
                 hop_length=200,
                 win_length=800,
                 n_mel_channels=80,
                 mel_fmin=0,
                 mel_fmax=8000,
                 sampling_rate=16000,
                 sampling_rate_org=None,
                 normalize=False,
                 mel_norm_file=None,
                 scale=1.0,
                 padding='center',
                 style='Tortoise'):

        super().__init__()
        self.style = style
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.sampling_rate_org = sampling_rate_org if sampling_rate_org is not None else sampling_rate
        
        if self.style == 'Vocos':
            self.mel_stft = torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=filter_length,
                hop_length=hop_length,
                n_mels=n_mel_channels,
                center=(padding == "center"),
                power=1,
            )
        elif self.style == 'Tortoise':
            self.mel_stft = torchaudio.transforms.MelSpectrogram(
                n_fft=self.filter_length, hop_length=self.hop_length,
                win_length=self.win_length, power=1, normalized=normalize,
                sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                norm="slaney", mel_scale="htk"
            )
            if mel_norm_file is not None:
                self.mel_norms = torch.load(mel_norm_file)
            else:
                self.mel_norms = None
        elif self.style == 'BigVGAN':
            self.mel_basis = {}
            self.hann_window = {}
        
        self.scale = scale

    def forward(self, *args, **kwargs):
        if self.style == 'Tortoise':
            return self.forward_tortoise(*args, **kwargs)
        elif self.style == 'BigVGAN':
            return self.forward_bigvgan(*args, **kwargs)
        elif self.style == 'Vocos':
            return self.forward_vocos(*args, **kwargs)

    def forward_bigvgan(self, inp, length=None):
        if len(inp.shape) == 3:
            inp = inp.squeeze(1) if inp.shape[1] == 1 else inp.squeeze(2)
        assert len(inp.shape) == 2

        if self.sampling_rate_org != self.sampling_rate:
            inp = torchaudio.functional.resample(inp, self.sampling_rate_org, self.sampling_rate)

        y = inp
        if len(list(self.mel_basis.keys())) == 0:
            mel = librosa_mel_fn(sr=self.sampling_rate, n_fft=self.filter_length, n_mels=self.n_mel_channels, fmin=self.mel_fmin, fmax=self.mel_fmax)
            self.mel_basis[str(self.mel_fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(y.device)

        y = torch.nn.functional.pad(y.unsqueeze(1), (int((self.filter_length-self.hop_length)/2), int((self.filter_length-self.hop_length)/2)), mode='reflect')
        y = y.squeeze(1)

        # complex tensor as default, then use view_as_real for future pytorch compatibility
        spec = torch.stft(y, self.filter_length, hop_length=self.hop_length, win_length=self.win_length, window=self.hann_window[str(y.device)],
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        spec = torch.matmul(self.mel_basis[str(self.mel_fmax)+'_'+str(y.device)], spec)
        spec = spectral_normalize_torch(spec)

        max_mel_length = math.ceil(y.shape[-1] / self.hop_length)
        spec = spec[..., : max_mel_length].transpose(1, 2)

        if length is None:
            return spec
        else:
            spec_len = torch.ceil(length / self.hop_length).clamp(max=spec.shape[1])
            return spec, spec_len
    
    def forward_vocos(self, inp, length=None):
        if len(inp.shape) == 3:
            inp = inp.squeeze(1) if inp.shape[1] == 1 else inp.squeeze(2)
        assert len(inp.shape) == 2

        if self.sampling_rate_org != self.sampling_rate:
            inp = torchaudio.functional.resample(inp, self.sampling_rate_org, self.sampling_rate)

        self.mel_stft = self.mel_stft.to(inp.device)

        mel = self.mel_stft(inp)
        mel = torch.log(torch.clip(mel, min=1e-7))
        return mel
        
    def forward_tortoise(self, inp, length=None):
        # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
        if len(inp.shape) == 3:
            inp = inp.squeeze(1) if inp.shape[1] == 1 else inp.squeeze(2)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)

        # Pre-emphasis
        # inp[..., 1:] -= 0.97 * inp[..., :-1] # in-place operation
        inp = torch.cat((inp[..., :1], inp[..., 1:] - 0.97 * inp[..., :-1]), dim=-1)

        # Extraction Mel-spectrogram (0 ~ 20dB)
        mel = self.mel_stft(inp)
        if self.scale == 1.0:
            mel = 20 * torch.log10(torch.clamp(mel, min=1e-3)) # -60 ~ 40
            mel = torch.clamp(mel + 60, 0, 80) / 80
        elif self.scale == [-4.0, 4.0]:
            mel = 20 * torch.log10(torch.clamp(mel, min=1e-5)) # -100 ~ 40
            mel = mel - 20 # -120 ~ 20
            mel = 8 * torch.clamp(mel / 100 + 0.5, -0.5, 0.5) # -4 ~ 4
        else:
            raise ValueError(f"Wrong scale: {self.scale}")

        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)

        max_mel_length = math.ceil(inp.shape[-1] / self.hop_length)
        mel = mel[..., : max_mel_length].transpose(1, 2)

        if length is None:
            return mel
        else:
            mel_length = torch.ceil(length / self.hop_length).clamp(max=mel.shape[1])
            return mel, mel_length