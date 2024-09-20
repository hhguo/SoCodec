from fairseq import checkpoint_utils
from torch.nn.utils.rnn import pad_sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    max_len = torch.max(lengths).item() if max_len is None else max_len
    ids = torch.arange(0, max_len).to(lengths.device)
    mask = ~(ids < lengths.unsqueeze(1)).bool()
    return mask


class HuBERT(nn.Module):

    def __init__(self,
                 sampling_rate=16000):

        super().__init__()
        model_path = "ckpts/chinese-hubert-large-fairseq-ckpt.pt"

        print("loading model(s) from {}".format(model_path))
        models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            suffix="",
        )
        print("loaded model(s) from {}".format(model_path))

        model = models[0]
        model = model.half()
        model.eval()
        self.model = model
        
        for param in self.parameters():
            param.requires_grad = False

        self.sampling_rate = sampling_rate
        self.normalize = saved_cfg.task.normalize
        print(f"normalize: {saved_cfg.task.normalize}")

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False, dtype=torch.float16)
    def forward(self, inp, length=None, split=True, split_size=4):
        self.model.eval()
        if self.training and split:
            split_size = int(math.ceil(inp.shape[0] / 4))
            outs, out_lens = [], []
            for i in range(0, inp.shape[0], split_size):
                inp_, length_ = inp[i: i + split_size], length[i: i + split_size]
                out_, out_len_ = self._extract(inp_, length_)
                outs.append(out_)
                out_lens.append(out_len_)
            max_length = max([max(ols) for ols in out_lens])

            return torch.cat([F.pad(o, (0, 0, 0, max_length - o.shape[1]), value=0) for o in outs], dim=0), torch.cat(out_lens, dim=0)
        else:
            return self._extract(inp, length)

    @torch.no_grad()
    def _extract(self, inp, length):
        frame_samples = int(self.sampling_rate * 0.02)
        device = inp.device

        if len(inp.shape) == 3:
            inp = inp.squeeze(1) if inp.shape[1] == 1 else inp.squeeze(2)
        assert len(inp.shape) == 2
        assert self.sampling_rate == 16000

        feats = inp

        # Padding with 0
        padding_size = 3200 # Longer to cover receptive field
        feats = F.pad(feats, (0, padding_size), mode='constant', value=0)

        # Norm volume using LN
        feats = self._postprocess(feats, length + padding_size, normalize=self.normalize)

        if length is None:
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        else:
            length = torch.ceil(length / 320).int()
            padding_mask = get_mask_from_lengths(length).bool()
            padding_mask = F.pad(padding_mask, (0, 9), value=True)

        inputs = {
            "source": feats.half().to(device),
            "padding_mask": padding_mask.to(device),
            "mask": False,
        }
        logits, _ = self.model.extract_features(**inputs)
        logits = logits[:, : length.max()].float()
        
        return logits, length

    def _postprocess(self, feats, lengths, normalize=False):
        assert feats.dim() == 2, feats.dim()

        if normalize:
            with torch.no_grad():
                # feats = F.layer_norm(feats, feats.shape)
                feats = [
                    F.layer_norm(feat[: length], feat[: length].shape)
                    for feat, length in zip(feats, lengths)
                ]
                feats = pad_sequence(
                    feats,
                    batch_first=True,
                    padding_value=0
                )
        return feats


if __name__ == '__main__':
    import soundfile as sf
    from time import time

    model = HuBERT().cuda()

    wav, sr = sf.read('/mnt/public/usr/haohanguo/workspace/voicemaker2/egs/inference/tts/reference_audio/wenetspeech_test_net/TEST_NET_Y0000000003_0l7Ov1lh8pk_S00068.wav')
    x = torch.tensor([wav]).cuda()
    x = F.pad(x, (0, 96000), value=0).repeat(2, 1)
    l = torch.tensor([len(wav) + 96000, len(wav)]).cuda().int()
    
    t1 = time()

    y, yl = model(x, l)
    y1 = y[1, : len(wav) // 320].float()
    y, yl = model(x[1:, : ], l[1:])
    y2 = y[0, : len(wav) // 320].float()
    print(F.mse_loss(y1, y2))

    t2 = time()
    print(t2 - t1)