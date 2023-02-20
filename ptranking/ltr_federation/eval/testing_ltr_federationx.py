#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

import numpy as np

from ptranking.ltr_global import ltr_seed
from ptranking.ltr_federation.eval.ltr_federation import FederationLTREvaluator

np.random.seed(seed=ltr_seed)
torch.manual_seed(seed=ltr_seed)

if __name__ == '__main__':

    cuda = None  # the gpu id, e.g., 0 or 1, otherwise, set it as None indicating to use cpu

    debug = True  # in a debug mode, we just check whether the model can operate

    sf_id = 'pointsf'  # pointsf | listsf, namely the type of neural scoring function

    config_with_json = False  # specify configuration with json files or not

    federation_id = None

    models_to_run = [
        'Federated_PDGD',
    ]

    evaluator = FederationLTREvaluator(cuda=cuda)

    if config_with_json:  # specify configuration with json files
        # the directory of json files
        dir_json = '/Users/iimac/II-Research Dropbox/Hai-Tao Yu/CodeBench/GitPool/json/iimac/'

        for model_id in models_to_run:
            evaluator.run(debug=debug, model_id=model_id, config_with_json=config_with_json, dir_json=dir_json)

    else:  # specify configuration manually
        data_id = 'MQ2008_Super'
        #data_id = 'MSLRWEB10K'

        ''' Location of the adopted data '''
        #dir_data = '/Users/iimac/Workbench/Corpus/L2R/LETOR4.0/MQ2008/'

        dir_data = '/Users/kanazawaatsuya/dataset/MQ2008/'
        #dir_data = '/Users/iimac/Workbench/Corpus/L2R/MSLR-WEB10K/'
        #dir_data = '/Users/solar/WorkBench/Datasets/L2R/LETOR4.0/MQ2008/'

        ''' Output directory '''
        dir_output = '/Users/kanazawaatsuya/lab/output/out_fed'
        #dir_output = '/Users/iimac/Workbench/CodeBench/Output/Out_Federation/'
        #dir_output = '/Users/solar/WorkBench/CodeBench/PyCharmProject/Project_output/Out_Federation/'

        grid_search = False  # with grid_search, we can explore the effects of different hyper-parameters of a model
        for model_id in models_to_run:
            evaluator.run(debug=debug, federation_id=federation_id, model_id=model_id, sf_id=sf_id,
                          data_id=data_id, dir_data=dir_data, dir_output=dir_output,
                          grid_search=grid_search)
