import torch

from ..tasks.base_task import BaseTask


class NeuralCodec(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_icl = True

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(self, input_dict, mode=None):
        output_dict = {}

        wav = input_dict.get('wav', None)
        wav_length = input_dict.get('wav_length', None)
        ref = input_dict.get('ref', None)
        ref_length = input_dict.get('ref_length', None)
        spk = input_dict.get('spk', None)

        if ref is None:
            ref, ref_length = wav, wav_length

        # Speech tokenization
        analyzer_dict = {}
        if wav is not None:
            analyzer_dict = self.analyzer.extract_speech_tokens(wav, wav_length)
            if mode == 'speech_tokenization':
                return analyzer_dict

        # Speech Reconstruction
        token = input_dict.get('token', analyzer_dict.get('token', None))

        feat_pred = self._decoding_from_semantic_codec(token, ref, ref_length, spk)
        wav_pred = self.vocoder(feat_pred).squeeze(1)

        output_dict['wav'] = wav_pred
        return output_dict

    def _decoding_from_semantic_codec(self, token, ref=None, ref_length=None, spk=None):
        if spk is not None:
            analyzer_outputs = self.analyzer.reconstruct_mel(token, spk=spk)
        else:
            cond_mel, _ = self.analyzer.mel_spectrogram(ref, ref_length)
            analyzer_outputs = self.analyzer.reconstruct_mel(token, mel=cond_mel)
        feat_pred = analyzer_outputs['mel_pred']
        return feat_pred