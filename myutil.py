import torch
import numpy as np
import json

""" configuration json """


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


def getConfig():
    config = Config({
        # device
        "GPU_ID": "0",
        "num_workers": 0,

        "n_enc_seq": 12 * 16,
        # feature map dimension (H x W) from backbone, this size is related to crop_size（crop size/32）
        "n_dec_seq": 12 * 16,  # feature map dimension (H x W) from backbone, this size is related to crop_size
        "n_layer": 1,  # number of encoder/decoder layers
        "d_hidn": 128,  # input channel (C) of encoder / decoder (input: C x N)
        "i_pad": 0,
        "d_ff": 1024,  # feed forward hidden layer dimension
        "d_MLP_head": 256,  # hidden layer of final MLP
        # "n_head": 6,# number of head (in multi-head attention)
        "n_head": 4,
        "d_head": 128,  # input channel (C) of each head (input: C x N) -> same as d_hidn
        "dropout": 0.1,  # dropout ratio of transformer
        "emb_dropout": 0.1,  # dropout ratio of input embedding
        "layer_norm_epsilon": 1e-12,
        "n_output": 1,  # dimension of final prediction
    })
    return config
