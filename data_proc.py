from numpy import arange
from torch.nn import Module, Sequential
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import ctc_decoder
from torch import tensor, transpose, tensor, argmax, float32
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

from constants import *
from utils import download_librispeech_kenlm_decoder_model


# waveform to spectral augmentation
class WaveformTransform(Module):
    def __init__(self, test=False):
        super(WaveformTransform, self).__init__()
        self.test = test
        self.mel_trans = MelSpectrogram(sample_rate = mel_sample_rate,
                                        n_fft = mel_n_fft,
                                        n_mels = n_mels,
                                        win_length = mel_win_length,
                                        hop_length = mel_hop_length)
        self.mel_trans = self.mel_trans.to(device)
        freq_mask_list = [FrequencyMasking(freq_mask_param=27)] * num_freq_masks
        self.freq_mask = Sequential(*freq_mask_list)
        self.freq_mask = self.freq_mask.to(device)
        time_mask_list = [TimeMasking(time_mask_param=35, p=0.05)] * num_time_masks
        self.time_mask = Sequential(*time_mask_list)
        self.time_mask = self.time_mask.to(device)

    def forward(self, data):
        data = data.to(device)
        if not self.test:
            out = self.mel_trans(data)
            out = self.freq_mask(out)
            out = self.time_mask(out)
            out = out.permute(2, 1, 0)
            return out
        else:
            out = self.mel_trans(data)
            out = out.permute(2, 1, 0)
            return out


# utterance to encoded text transform
class TextTransform(Module):
    def __init__(self,):
        super(TextTransform, self).__init__()
        files = download_librispeech_kenlm_decoder_model()
        self.beam_search_decoder = ctc_decoder(
                                lexicon=files.lexicon,
                                tokens= charlist, #files.tokens,
                                lm=files.lm if beam_use_lm else None,
                                # nbest=beam_nbest,
                                # beam_size=beam_decoder_size,
                                # lm_weight=beam_lm_weight,
                                # word_score=beam_word_score,
                            )

        self.n_classes = n_classes
        self.chars = charlist
        self.indices = arange(len(self.chars)).tolist()
        self.encode_map = dict(zip(self.chars, self.indices))
        self.decode_map = dict(zip(self.indices, self.chars))
        self.encode_map[''] = pad_val
        self.decode_map[pad_val] = ''
        assert len(self.encode_map) == len(self.decode_map)
        
    def encode_sequence(self, sequence):
        final = []
        for char in sequence:
            if char == ' ' or char == '|':
                final.append(self.encode_map['|'])
            else:
                final.append(self.encode_map[char])
        out = tensor(final).to(device)
        return out
                
    def greedy_decode_sequences(self, sequences, label_lengths=None):
        out = []
        if label_lengths == None:
            sequences = transpose(argmax(sequences, dim=-1), 0, 1).tolist()
        else:
            sequences = [sequences[i][:label_lengths[i]].tolist() for i in range(len(label_lengths))]
        for sequence in sequences:
            final = []
            for i, char in enumerate(sequence):
                if char != blank:
                    if label_lengths == None and i != 0 and char == sequence[i - 1]:
                        continue
                    final.append(self.decode_map[char])
            out.append(' '.join(''.join(final).split('|')))
        return out

    def beam_decode_sequences(self, sequences):
        sequences = sequences.permute(1, 0, 2)
        sequences = sequences.type(float32).cpu()
        results = self.beam_search_decoder(sequences)
        out = []
        for sample in results:
            hypothesis = sample[0]
            out.append(' '.join(hypothesis.words))
        return out


def data_processing(data, wav_trans, text_trans):
    waveforms = []
    spec_lengths = []
    texts = []
    text_lengths = []
    for (waveform, _, transcript, _, _, _) in data:
        waveform = wav_trans(waveform)
        waveforms.append(waveform)
        spec_lengths.append(waveform.shape[0])
        encoded_text = text_trans.encode_sequence(transcript.lower())
        texts.append(encoded_text)
        text_lengths.append(encoded_text.shape[0])
    waveforms = transpose(pad_sequence(waveforms, padding_value=pad_val).squeeze(), 0, 1).to(device)
    spec_lengths = tensor(spec_lengths).to(device)
    texts = pad_sequence(texts, padding_value=pad_val).permute(1, 0).to(device)
    text_lengths = tensor(text_lengths).to(device)
    return waveforms, spec_lengths, texts, text_lengths