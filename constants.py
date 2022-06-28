import os
import shutil
import pandas as pd
from torch import manual_seed, device

# waveform
mel_n_fft = 400
mel_sample_rate = 16000
n_mels = 80
mel_win_length = 400
mel_hop_length = 160

num_time_masks = 2
num_freq_masks = 2


# text
charlist = list("-|etaonihsrdlumwcfgypbvk'xjqz")
blank = 0
pad_val = len(charlist)
n_classes = len(charlist) + 1

# decoder
train_decoder = 'greedy' # both beam and greedy
test_decoder = None
beam_use_lm = True

# beam_lm_weight = ...
# beam_word_score = ...
# beam_nbest = ...
# beam_decoder_size = ...


# model constants
conformer_num_heads = 4
conformer_ffn_dim = 144
conformer_num_layers = 4
conformer_depthwise_conv_kernel_size = 31
conformer_dropout = 0.2

lstm_layers = 3
lstm_dim = 320

# torch params 
device = device("cuda")
manual_seed(8)

# train params
epochs = 30
batch_size = 2

# optimizer params
max_learning_rate = 1e-4 
scheduler_warmup_ratio = 1 / (3 * epochs)
b1 = 0.9
b2 = 0.98
epsilon = 1e-09

# model save dir (in models folder)
experiment_num = os.listdir('models')
model_save_dir = os.path.join('models', 'experiment_' + str(len(experiment_num)))
model_save_path = os.path.join(model_save_dir, 'epochs')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
shutil.copy('constants.py', model_save_dir)

# predict
model_path = None #'models/experiment_0/epochs/epoch_44.pth'

# pandas eval display options
print_samples_n = 10 # None if you don't want the display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)