from torch.nn import LSTM, Linear, LogSoftmax, Module, ReLU
from torchaudio.models import Conformer

from constants import *

class SpeechRegonitionModel(Module):
    def __init__(self, ):
        super(SpeechRegonitionModel, self).__init__()
        self.conformer = Conformer(
                                input_dim = n_mels, 
                                num_heads = conformer_num_heads, 
                                ffn_dim = conformer_ffn_dim, 
                                num_layers = conformer_num_layers, 
                                depthwise_conv_kernel_size = conformer_depthwise_conv_kernel_size, 
                                dropout = conformer_dropout
                            )

        self.lstm = LSTM(input_size=n_mels,
                        hidden_size=lstm_dim,
                        num_layers=lstm_layers,
                        batch_first=True)
        self.relu = ReLU()

        self.classifier = Linear(lstm_dim, n_classes)
        self.softmax = LogSoftmax(dim=2)
                
    def forward(self, data, lengths):
        out, _ = self.conformer(data, lengths)
        out, _ = self.lstm(out)
        out = self.relu(out)
        out = self.classifier(out)
        out = self.softmax(out)
        out = out.permute(1, 0, 2)
        return out
