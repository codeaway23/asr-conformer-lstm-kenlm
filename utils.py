import os
import re
from torchaudio.datasets import  LIBRISPEECH
from torchaudio.models.decoder import download_pretrained_files

# urls = [    
#         "dev-clean", 
#         "dev-other", 
#         "test-clean", 
#         "test-other", 
#         "train-clean-100", 
#         "train-clean-360",
#         "train-other-500"
#     ]


def download_librispeech_data(urls=None):
    root = 'librispeech/data'
    if not os.path.exists(root):
        os.makedirs(root)
    urls= ['train-clean-100', 'test-clean']
    for url in urls:
        LIBRISPEECH(root=root, 
                    url=url, 
                    download=True)


def download_librispeech_kenlm_decoder_model():
    files = download_pretrained_files("librispeech-4-gram")
    return files


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)