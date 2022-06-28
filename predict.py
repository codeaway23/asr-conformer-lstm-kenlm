import os
import pandas as pd
from tqdm import tqdm
from torch import no_grad, load
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH

from constants import *
from utils import natural_sort
from model import SpeechRegonitionModel
from data_proc import WaveformTransform, TextTransform, data_processing

def predict(dataloader,
         model,
         text_trans,
         labels_data=False,
         print_eval_samples=print_samples_n):

        if not test_decoder == None:
            out_seqs = []
            label_seqs = []
        else:
            out_seqs_greedy = []
            out_seqs_beam = []
            label_seqs = []
        
        model.eval()
        
        print('predicting...')
        print('--------------------------------------------------------------------------')
        for step, batch in tqdm(enumerate(dataloader), total = len(dataloader)):

            # if step == 10:
            #     break

            if labels_data:
                specs, spec_lengths, labels, label_lengths = batch
                specs = specs.to(device)
                spec_lengths = spec_lengths.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)
            else:
                specs, spec_lengths, _, _ = batch
                specs = specs.to(device)
                spec_lengths = spec_lengths.to(device)
                            

            with no_grad():
                out = model(specs, spec_lengths)            


            if test_decoder == 'beam':
                out_seqs += text_trans.beam_decode_sequences(out)
            elif test_decoder == 'greedy':
                out_seqs += text_trans.greedy_decode_sequences(out)
            elif test_decoder == None:
                out_seqs_beam += text_trans.beam_decode_sequences(out)
                out_seqs_greedy += text_trans.greedy_decode_sequences(out)

            if labels_data:
                label_seqs += text_trans.greedy_decode_sequences(labels, label_lengths=label_lengths)

        if print_eval_samples and labels_data:
            print('randomly sampled validation predictions...')
            if not test_decoder == None:
                df = pd.DataFrame({'label': label_seqs, 'pred': out_seqs})
            else:
                df = pd.DataFrame({'label': label_seqs, 'pred_greedy': out_seqs_greedy, 'pred_beam': out_seqs_beam})
            print(df.sample(print_eval_samples))
        elif print_eval_samples:
            df = pd.Series(out_seqs)
            print(df.sample(print_eval_samples))
            return df

if __name__ == "__main__":

    wav_test_trans = WaveformTransform(test=True)
    text_trans = TextTransform()

    wav_test_trans = wav_test_trans.to(device)
    text_trans = text_trans.to(device)

    test_dataset = LIBRISPEECH(root = 'librispeech/data', 
                            url = 'test-clean'
                        )

    test_dataloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn = lambda x: data_processing(x, wav_test_trans, text_trans),
                            drop_last = True
                        )

    if model_path:
        model = SpeechRegonitionModel()
        model.load_state_dict(load(model_path))
        model = model.to(device)
        out = predict(test_dataloader,
                    model,
                    text_trans,
                    labels_data=True)
    else:
        raise Exception('please provide model path')
        # paths = [os.path.join('models', model_save_path, x) for x in os.listdir(os.path.join('models', model_save_path))]
        # paths = natural_sort(paths)
        # for model_path in paths:
        #     print('epoch: ', model_path)
        #     print('-----------------------------------')
        #     model = SpeechRegonitionModel()
        #     model.load_state_dict(load(model_path))
        #     model = model.to(device)
        #     out = predict(test_dataloader,
        #                 model,
        #                 text_trans,
        #                 labels_data=True)
