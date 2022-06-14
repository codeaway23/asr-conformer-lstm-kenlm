from torchaudio.datasets import  LIBRISPEECH
import os


def download_librispeech_data(urls=None):
    root = 'librispeech/data'
    if not os.path.exists(root):
        os.makedirs(root)
    # urls = [    
    #         "dev-clean", 
    #         "dev-other", 
    #         "test-clean", 
    #         "test-other", 
    #         "train-clean-100", 
    #         "train-clean-360",
    #         "train-other-500"
    #     ]
    urls= ['train-clean-100', 'test-clean']
    for url in urls:
        LIBRISPEECH(root=root, 
                    url=url, 
                    download=True)
