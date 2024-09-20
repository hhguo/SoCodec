import argparse
import torch
import torchaudio

from socodec.tasks import build_task
from socodec.utils.config import Config
from socodec.utils.utils import to_model


class SoCodec():

    def __init__(self, config):
        # Startup task
        task = build_task(config.task, 'eval')
        if torch.cuda.is_available():
            task = task.cuda()
        task.eval()
        self.model = task

    def analysis_synthesis(self, audio):
        features = self.analysis(audio)
        syn_audio = self.synthesis(features)
        return features, syn_audio

    def analysis(self, audio):
        if not isinstance(audio, (tuple, list)):
            audio = [audio]
        
        audio_length = torch.tensor([max(x.shape) for x in audio])
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)

        features = {'wav': audio, 'wav_length': audio_length}
        features = to_model(features)

        features = self.model(features, mode='speech_tokenization')
        tokens = features['token']
        token_lengths = features['token_length']
        spks = features['spk']

        features = [
            {'token': token[: token_length].cpu(), 'speaker': spk.cpu()}
            for token, token_length, spk in zip(tokens, token_lengths, spks)
        ]
        if len(features) == 1:
            features = features[0]
        
        return features

    def synthesis(self, features):
        if not isinstance(features, (tuple, list)):
            features = [features]

        token_lengths = torch.tensor([len(x['token']) for x in features])
        tokens = torch.nn.utils.rnn.pad_sequence([x['token'] for x in features], batch_first=True)
        spks = torch.stack([x['speaker'] for x in features])
        features = {
            'token': tokens,
            'token_length': token_lengths,
            'spk': spks,
        }

        features = to_model(features)
        saved_features = self.model(features, mode='speech_reconstruction')
        
        wavs = [x.cpu() for x in saved_features['wav']]
        if len(wavs) == 1:
            wavs = wavs[0]
        return wavs


def init_codec():
    config = {
        'task': {
            '_name': 'NeuralCodec',
            'network': {
                'analyzer': {
                    '_conf': 'configs/socodec_16384x4_120ms_16khz_chinese.yaml',
                    '_ckpt': {
                        'path': 'ckpts/socodec_16384x4_120ms_16khz_chinese.safetensors',
                        'strict': False
                    }
                },
                'vocoder': {
                    '_conf': 'configs/mel_vocoder_80dim_10ms_16khz.yaml',
                    '_ckpt': {
                        'path': 'ckpts/mel_vocoder_80dim_10ms_16khz.safetensors',
                        'strict': True
                    }
                }
            }
        },
        'sampling_rate': 24000,
    }
    return SoCodec(Config(config))


def test(input_file, output_file):
    # Init codec
    codec = init_codec()

    # Encode or decode audio
    if input_file.endswith('wav'):
        # Load audio and resample to 16khz
        audio, sr = torchaudio.load(input_file)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        
        # Trick: prepend 0.5s silence for casual convolutions
        # (TODO) will be removed in the next version
        silence = torch.zeros((1, 8000))
        audio = torch.cat((silence, audio), dim=-1)

        features, syn_audio = codec.analysis_synthesis(audio)
    elif input_file.endswith('pt'):
        features = torch.load(input_file)
        syn_audio = codec.reconstruct(features)

    # Save audio or features
    if output_file.endswith('wav'):
        if len(syn_audio.shape) == 1:
            syn_audio = syn_audio.unsqueeze(0)
        torchaudio.save(output_file, syn_audio, 16000)
    elif output_file.endswith('pt'):
        torch.save(features, output_file)
    else:
        raise TypeError('Unacceptable file format:', output_file)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input")
    parser.add_argument('-o', "--output")
    args = parser.parse_args()

    # Inference
    with torch.no_grad():
        test(args.input, args.output)


if __name__ == '__main__':
    main()
