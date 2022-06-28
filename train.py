import os
import wandb
import pandas as pd
from tqdm import tqdm
from numpy import mean
from jiwer import wer, cer
from torch import no_grad, save

from constants import *

def train(train_dataloader,
          model, 
          loss_fct, 
          optimizer,
          scheduler,
          epochs,
          text_trans,
          test_dataloader=None):

    wandb.watch(model, criterion=loss_fct, log="all")

    print('training...')
    print('--------------------------------------------------------------------------')
    for epoch in range(epochs):    

        if not train_decoder == None:
            epoch_train_loss = []
            epoch_train_wer = []
            epoch_train_cer = []
        else:
            epoch_train_loss = []
            epoch_train_wer_greedy = []
            epoch_train_cer_greedy = []
            epoch_train_wer_beam = []
            epoch_train_cer_beam = []
            

        model.train()        
        print('Epoch: ', epoch + 1)
        
        for step, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):

            # if step == 10:
            #     break

            optimizer.zero_grad()

            specs, spec_lengths, labels, label_lengths = batch
            specs = specs.to(device)
            spec_lengths = spec_lengths.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            out = model(specs, spec_lengths)  
            
            loss = loss_fct(out, labels, spec_lengths, label_lengths)
            epoch_train_loss.append(loss.float().detach().cpu().numpy().mean())
            
            if train_decoder == 'beam':
                out_seqs = text_trans.beam_decode_sequences(out)
            elif train_decoder == 'greedy':
                out_seqs = text_trans.greedy_decode_sequences(out)
            elif train_decoder == None:
                out_seqs_beam = text_trans.beam_decode_sequences(out)
                out_seqs_greedy = text_trans.greedy_decode_sequences(out)
            
            label_seqs = text_trans.greedy_decode_sequences(labels, label_lengths=label_lengths)
            
            this_batch = specs.shape[0]
            
            if not train_decoder == None:
                wer_val = mean([wer(label_seqs[i], out_seqs[i]) for i in range(this_batch)])
                epoch_train_wer.append(wer_val)
                cer_val = mean([cer(label_seqs[i], out_seqs[i]) for i in range(this_batch)])
                epoch_train_cer.append(cer_val)

                wandb.log({
                    "train/loss": epoch_train_loss[-1], 
                    "train/wer": wer_val,
                    "train/cer": cer_val,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/step": epoch * len(train_dataloader) + step
                    })

            else:
                wer_val_greedy = mean([wer(label_seqs[i], out_seqs_greedy[i]) for i in range(this_batch)])
                epoch_train_wer_greedy.append(wer_val_greedy)
                cer_val_greedy = mean([cer(label_seqs[i], out_seqs_greedy[i]) for i in range(this_batch)])
                epoch_train_cer_greedy.append(cer_val_greedy)

                wer_val_beam = mean([wer(label_seqs[i], out_seqs_beam[i]) for i in range(this_batch)])
                epoch_train_wer_beam.append(wer_val_beam)
                cer_val_beam = mean([cer(label_seqs[i], out_seqs_beam[i]) for i in range(this_batch)])
                epoch_train_cer_beam.append(cer_val_beam)
                
                wandb.log({
                    "train/loss": epoch_train_loss[-1], 
                    "train/wer_greedy": wer_val_greedy,
                    "train/cer_greedy": cer_val_greedy,
                    "train/wer_beam": wer_val_beam,
                    "train/cer_beam": cer_val_beam,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/step": epoch * len(train_dataloader) + step
                    })

            loss.backward()
            optimizer.step()   
            scheduler.step()


        if not train_decoder == None:
            wandb.log({
                "epoch/train/loss": mean(epoch_train_loss), 
                "epoch/train/wer": mean(epoch_train_wer),
                "epoch/train/cer": mean(epoch_train_cer),
                "epoch": epoch
            })
        else:
            wandb.log({
                "epoch/train/loss": mean(epoch_train_loss), 
                "epoch/train/wer_greedy": mean(epoch_train_wer_greedy),
                "epoch/train/cer_greedy": mean(epoch_train_cer_greedy),
                "epoch/train/wer_beam": mean(epoch_train_wer_beam),
                "epoch/train/cer_beam": mean(epoch_train_cer_beam),
                "epoch": epoch
            })
            

        if isinstance(model_save_path, str):
            save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

        if test_dataloader:
            test(test_dataloader, model, loss_fct, optimizer, text_trans, epoch=epoch, print_eval_samples=print_samples_n)

    return model

def test(test_dataloader,
         model,
         loss_fct, 
         optimizer,
         text_trans,
         epoch=None,
         print_eval_samples=None):

        if not test_decoder == None:
            epoch_test_loss = []
            epoch_test_wer = []
            epoch_test_cer = []
            out_seqs = []
            label_seqs = []
        else:
            epoch_test_loss = []
            epoch_test_wer_greedy = []
            epoch_test_cer_greedy = []
            out_seqs_greedy = []
            epoch_test_wer_beam = []
            epoch_test_cer_beam = []
            out_seqs_beam = []
            label_seqs = []
        
        model.eval()
        
        print('evaluating...')
        print('--------------------------------------------------------------------------')
        for step, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):

            # if step == 5:
            #     break

            optimizer.zero_grad()

            specs, spec_lengths, labels, label_lengths = batch
            specs = specs.to(device)
            spec_lengths = spec_lengths.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            

            with no_grad():
                out = model(specs, spec_lengths)            

            loss = loss_fct(out, labels, spec_lengths, label_lengths)
            epoch_test_loss.append(loss.float().detach().cpu().numpy().mean())

            if test_decoder == 'beam':
                out_seqs += text_trans.beam_decode_sequences(out)
            elif test_decoder == 'greedy':
                out_seqs += text_trans.greedy_decode_sequences(out)
            elif test_decoder == None:
                out_seqs_beam += text_trans.beam_decode_sequences(out)
                out_seqs_greedy += text_trans.greedy_decode_sequences(out)
            
            label_seqs += text_trans.greedy_decode_sequences(labels, label_lengths=label_lengths)

            this_batch = specs.shape[0]

            if not test_decoder == None:
                wer_val = mean([wer(label_seqs[-this_batch:][i], out_seqs[-this_batch:][i]) for i in range(this_batch)])
                epoch_test_wer.append(wer_val)
                cer_val = mean([cer(label_seqs[-this_batch:][i], out_seqs[-this_batch:][i]) for i in range(this_batch)])
                epoch_test_cer.append(cer_val)

                wandb.log({
                    "test/loss": epoch_test_loss[-1], 
                    "test/wer": wer_val,
                    "test/cer": cer_val,
                    "test/step": epoch * len(test_dataloader) + step
                    })

            else:
                wer_val_greedy = mean([wer(label_seqs[-this_batch:][i], out_seqs_greedy[-this_batch:][i]) for i in range(this_batch)])
                epoch_test_wer_greedy.append(wer_val_greedy)
                cer_val_greedy = mean([cer(label_seqs[-this_batch:][i], out_seqs_greedy[-this_batch:][i]) for i in range(this_batch)])
                epoch_test_cer_greedy.append(cer_val_greedy)

                wer_val_beam = mean([wer(label_seqs[-this_batch:][i], out_seqs_beam[-this_batch:][i]) for i in range(this_batch)])
                epoch_test_wer_beam.append(wer_val_beam)
                cer_val_beam = mean([cer(label_seqs[-this_batch:][i], out_seqs_beam[-this_batch:][i]) for i in range(this_batch)])
                epoch_test_cer_beam.append(cer_val_beam)
                
                wandb.log({
                    "test/loss": epoch_test_loss[-1], 
                    "test/wer_greedy": wer_val_greedy,
                    "test/cer_greedy": cer_val_greedy,
                    "test/wer_beam": wer_val_beam,
                    "test/cer_beam": cer_val_beam,
                    "test/step": epoch * len(test_dataloader) + step
                    })


        if not test_decoder == None:
            wandb.log({
                "epoch/test/loss": mean(epoch_test_loss), 
                "epoch/test/wer": mean(epoch_test_wer),
                "epoch/test/cer": mean(epoch_test_cer),
                "epoch": epoch
            })
        else:
            wandb.log({
                "epoch/test/loss": mean(epoch_test_loss), 
                "epoch/test/wer_greedy": mean(epoch_test_wer_greedy),
                "epoch/test/cer_greedy": mean(epoch_test_cer_greedy),
                "epoch/test/wer_beam": mean(epoch_test_wer_beam),
                "epoch/test/cer_beam": mean(epoch_test_cer_beam),
                "epoch": epoch
            })

        if print_eval_samples and not test_decoder == None:
            print('randomly sampled validation predictions...')
            df = pd.DataFrame({'label': label_seqs, 'pred': out_seqs})
            print(df.sample(print_eval_samples))
        else:
            print('randomly sampled validation predictions...')
            df = pd.DataFrame({'label': label_seqs, 'pred_greedy': out_seqs_greedy, 'pred_beam': out_seqs_beam})
            print(df.sample(print_eval_samples))
            
