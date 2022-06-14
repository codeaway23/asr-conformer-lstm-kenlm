import os
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH

from torch.optim import Adam
from transformers.optimization import get_scheduler

import wandb

from data_proc import WaveformTransform, TextTransform
from data_proc import data_processing

from model import SpeechRegonitionModel

from constants import *
from train import train, test


if __name__ == "__main__":

    wandb.init(project="asr-conformer-lstm", entity="codeaway23")


    wav_train_trans = WaveformTransform()
    wav_test_trans = WaveformTransform(test=True)
    text_trans = TextTransform()

    wav_train_trans = wav_train_trans.to(device)
    wav_test_trans = wav_test_trans.to(device)
    text_trans = text_trans.to(device)
    model = SpeechRegonitionModel()

    model = model.to(device)

    print(model)

    for param in model.parameters():
        param.requires_grad = True

    train_dataset = LIBRISPEECH(root = 'librispeech/data', 
                            url = 'train-clean-100'
                        )
    test_dataset = LIBRISPEECH(root = 'librispeech/data', 
                            url = 'test-clean'
                        )

    train_dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn = lambda x: data_processing(x, wav_train_trans, text_trans),
                            drop_last = True
                        )
    test_dataloader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn = lambda x: data_processing(x, wav_test_trans, text_trans),
                            drop_last = True
                        )

    loss_fct = CTCLoss(blank=blank)

    num_training_steps = epochs * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * scheduler_warmup_ratio)

    print('--------------------------------------------------------------------------')
    print('Number of steps per epoch: ', len(train_dataloader))
    print('Number of epochs: ', epochs)
    print('Number of training steps: ', num_training_steps)
    print('Number of warmup steps: ', num_warmup_steps)

    optimizer = Adam(model.parameters(), lr=max_learning_rate, betas=(b1, b2), eps=epsilon)
    scheduler = get_scheduler('polynomial', 
                              optimizer,
                              num_warmup_steps = num_warmup_steps,
                              num_training_steps = num_training_steps,
                        )

    train(train_dataloader, 
        model, 
        loss_fct, 
        optimizer, 
        scheduler,
        epochs,
        text_trans)

    test(test_dataloader,
         model,
         loss_fct, 
         optimizer,
         text_trans,
         print_samples_n)

