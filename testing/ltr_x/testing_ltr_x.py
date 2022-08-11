#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

import numpy as np

from ptranking.ltr_global import ltr_seed
from ptranking.ltr_x.eval.ltr_x import XLTREvaluator

np.random.seed(seed=ltr_seed)
torch.manual_seed(seed=ltr_seed)

if __name__ == '__main__':

    cuda = None  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

    debug = True  # in a debug mode, we just check whether the model can operate

    sf_id = 'pointsf'  # pointsf | listsf, namely the type of neural scoring function

    config_with_json = False  # specify configuration with json files or not

    models_to_run = [
        'PDGD',
    ]

    evaluator = XLTREvaluator(cuda=cuda)

    if config_with_json:  # specify configuration with json files
        # the directory of json files
        dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/json/iimac/'

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, config_with_json=config_with_json, dir_json=dir_json)

    else:  # specify configuration manually
        data_id = 'MQ2008_Super'
        #data_id = 'MSLRWEB10K'

        ''' Location of the adopted data '''
        dir_data = '/Users/iimac/Workbench/Corpus/L2R/LETOR4.0/MQ2008/'
        #dir_data = '/Users/iimac/Workbench/Corpus/L2R/MSLR-WEB10K/'

        ''' Output directory '''
        dir_output = '/Users/iimac/Workbench/CodeBench/Output/NeuralLTR/'

        grid_search = False  # with grid_search, we can explore the effects of different hyper-parameters of a model
        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output,
                          sf_id=sf_id, grid_search=grid_search)
