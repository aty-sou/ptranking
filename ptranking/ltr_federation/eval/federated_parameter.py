
import json
import numpy as np
from itertools import product

from ptranking.utils.bigdata.BigPickle import pickle_save
from ptranking.ltr_adhoc.eval.parameter import ModelParameter

class FederationParameter(ModelParameter):
    """
    The parameter class w.r.t. a neural scoring fuction
    """
    def __init__(self, debug=False, federation_id=None, fed_json=None):
        super(FederationParameter, self).__init__(para_json=fed_json)
        self.debug = debug
        if self.use_json:
            self.federation_id = self.json_dict['federation_id']
        else:
            self.federation_id = federation_id

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["FederationParameter"]
        return json_dict

    def default_para_dict(self):
        """
        Default parameter setting for federated LTR
        """
        self.fed_para_dict = dict(federation_id=self.federation_id, num_clients=3,
                                  interactions_per_feedback=4, interaction_budget=600, per_interaction_update=False,
                                  epsilon=None, sensitivity=None,
                                  enable_noise=False)
        return self.fed_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for federated LTR
        :return:
        """
        s1, s2 = (':', '\n') if log else ('_', '_')
        # using specified para-dict or inner para-dict
        fed_para_dict = given_para_dict if given_para_dict is not None else self.fed_para_dict

        num_session, num_burn_in_epoch = fed_para_dict['num_session'], fed_para_dict['num_burn_in_epoch']

        fed_para_str = s1.join(['Session', str(num_session), 'BurnIn', str(num_burn_in_epoch)])
        return fed_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for RankNet
        """
        # todo TBA
        if self.use_json:
            choice_num_session = self.json_dict['num_session']
            choice_num_burn_in_epoch = self.json_dict['num_burn_in_epoch']
        else:
            choice_num_session = [5] if self.debug else [10000]
            choice_num_burn_in_epoch = [10] if self.debug else [50]

        for num_session, num_burn_in_epoch in product(choice_num_session, choice_num_burn_in_epoch):
            self.pdgd_para_dict = dict(model_id=self.model_id,
                                       num_session=num_session, num_burn_in_epoch=num_burn_in_epoch)
            yield self.pdgd_para_dict

class FederationSummaryTape(object):
    """
    Using multiple metrics to perform epoch-wise evaluation on train-data, validation-data, test-data
    """
    def __init__(self, cutoffs, label_type, test_presort, gpu):
        self.gpu = gpu
        self.cutoffs = cutoffs
        self.label_type = label_type
        self.list_fold_k_test_track = []
        self.test_presort = test_presort

    def epoch_summary(self, ranker, test_data):
        #print("epoch_summary")
        ''' Summary in terms of nDCG '''
        fold_k_epoch_k_test_ndcg_ks = ranker.ndcg_at_ks(test_data=test_data, ks=self.cutoffs, device='cpu',
                                                        label_type=self.label_type, presort=self.test_presort)
        np_fold_k_epoch_k_test_ndcg_ks  = \
            fold_k_epoch_k_test_ndcg_ks.cpu().numpy() if self.gpu else fold_k_epoch_k_test_ndcg_ks.data.numpy()
        self.list_fold_k_test_track.append(np_fold_k_epoch_k_test_ndcg_ks)

    def fold_summary(self, fold_k, dir_run, train_data_length):
        #print("fold_summary")
        sy_prefix = '_'.join(['Fold', str(fold_k)])

        fold_k_test_eval = np.vstack(self.list_fold_k_test_track)
        pickle_save(fold_k_test_eval, file=dir_run + '_'.join([sy_prefix, 'test_eval.np']))
