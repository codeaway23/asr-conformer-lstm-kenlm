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
charlist = list(" abcdefghijklmnopqrstuvwxyz0123456789,'\"")
pad_val = 0
blank = len(charlist) + 1
if blank == pad_val:
    n_classes = len(charlist) + 1
else:
    n_classes = len(charlist) + 2
decode_collapse = True

## model constants
conformer_num_heads = 4
conformer_ffn_dim = 144
conformer_num_layers = 4
conformer_depthwise_conv_kernel_size = 31
conformer_dropout = 0.2

lstm_layers = 3
lstm_dim = 320

# torch params 
device = device("cuda")
manual_seed(7)

# train
epochs = 10
batch_size = 3

max_learning_rate = 1e-4 
scheduler_warmup_ratio = 1 / (3 * epochs)
b1 = 0.9
b2 = 0.98
epsilon = 1e-09

# model save dir (in models folder)
model_save_path = 'model_1'

# eval
print_samples_n = None
