
import os
import copy

import torch

from ptranking.base.utils import get_stacked_FFNet
from ptranking.base.list_ranker import MultiheadAttention, PositionwiseFeedForward, Encoder, EncoderLayer, ListNeuralRanker

dc = copy.deepcopy

class XListNeuralRanker(ListNeuralRanker):
    '''
    A univariate scoring function for diversified ranking, where listwise information is integrated.
    '''
    def __init__(self, id='XListNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(XListNeuralRanker, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.encoder_type = self.sf_para_dict[self.sf_para_dict['sf_id']]['encoder_type']
