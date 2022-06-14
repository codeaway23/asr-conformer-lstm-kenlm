import os
from torch import transpose, argmax, no_grad, save
from numpy import mean
import wandb
from jiwer import wer, cer
import pandas as pd
from tqdm import tqdm

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

        epoch_train_loss = []
        epoch_train_wer = []
        epoch_train_cer = []

        model.train()        
        print('Epoch: ', epoch + 1)
        
        for _, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
            
            optimizer.zero_grad()

            specs, spec_lengths, labels, label_lengths = batch
            specs = specs.to(device)
            spec_lengths = spec_lengths.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            out = model(specs, spec_lengths)  
            
            loss = loss_fct(out, labels, spec_lengths, label_lengths)
            epoch_train_loss.append(loss.float().detach().cpu().numpy().mean())
            
            out = transpose(argmax(out, dim=-1), 0, 1)
            out_seqs = [text_trans.greedy_decode_sequence(x, preds=True) for x in out]
            label_seqs = [text_trans.greedy_decode_sequence(x[:label_lengths[i]]) for i,x in enumerate(labels)]
            
            this_batch = specs.shape[0]
            
            wer_val = mean([wer(label_seqs[i], out_seqs[i]) for i in range(this_batch)])
            epoch_train_wer.append(wer_val)
            cer_val = mean([cer(label_seqs[i], out_seqs[i]) for i in range(this_batch)])
            epoch_train_cer.append(cer_val)

            loss.backward()
            optimizer.step()   
            scheduler.step()

            wandb.log({
                "train_loss": epoch_train_loss[-1], 
                "train_wer": wer_val,
                "train_cer": cer_val,
                "learning_rate": optimizer.param_groups[0]['lr']
                })


        wandb.log({
            "epoch_train_loss": mean(epoch_train_loss), 
            "epoch_train_wer": mean(epoch_train_wer),
            "epoch_train_cer": mean(epoch_train_cer)
        })

        if isinstance(model_save_path, str):
            if not os.path.exists(os.path.join('./models', model_save_path)):
                os.makedirs(os.path.join('./models', model_save_path))
            save(model.state_dict(), os.path.join('./models', model_save_path, 'epoch_{}.pth'.format(epoch)))

        if test_dataloader:
            test(test_dataloader, model, loss_fct, optimizer, text_trans)



def test(test_dataloader,
         model,
         loss_fct, 
         optimizer,
         text_trans,
         print_eval_samples=None):

        epoch_test_loss = []
        epoch_test_wer = []
        epoch_test_cer = []

        out_seqs = []
        label_seqs = []
        out_tensors = []
        label_tensors = []
        
        model.eval()
        
        print('evaluating...')
        print('--------------------------------------------------------------------------')
        for _, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):

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

            out = transpose(argmax(out, dim=-1), 0, 1)
            out_seqs += [text_trans.greedy_decode_sequence(x, preds=True) for x in out]
            label_seqs += [text_trans.greedy_decode_sequence(x[:label_lengths[i]]) for i,x in enumerate(labels)]
            out_tensors += [x.tolist() for x in out]
            label_tensors += [x[:label_lengths[i]].tolist() for i,x in enumerate(labels)]

            this_batch = specs.shape[0]

            wer_val = mean([wer(label_seqs[-this_batch:][i], out_seqs[-this_batch:][i]) for i in range(this_batch)])
            epoch_test_wer.append(wer_val)
            cer_val = mean([cer(label_seqs[-this_batch:][i], out_seqs[-this_batch:][i]) for i in range(this_batch)])
            epoch_test_cer.append(cer_val)
            
            wandb.log({
                "test_loss": epoch_test_loss[-1], 
                "test_wer": wer_val,
                "test_cer": cer_val
                })

        
        if print_eval_samples:
            print('randomly sampled validation predictions...')
            df = pd.DataFrame({'label': label_seqs, 'pred': out_seqs, 'label_tensors': label_tensors, 'pred_tensors': out_tensors})
            print(df.sample(print_eval_samples))
