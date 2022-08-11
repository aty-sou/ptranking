
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from itertools import product

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.base.point_ranker import PointNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter

class DLA(PointNeuralRanker):
    '''
    Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018.
    Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18
    '''
    def __init__(self, id='DLA', sf_para_dict=None, model_para_dict=None, click_model=None,
                 weight_decay=1e-3, gpu=False, device=None):
        super(DLA, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.click_model = click_model
        self.num_session, self.num_burn_in_epoch = model_para_dict['num_session'], model_para_dict['num_burn_in_epoch']

    def init(self):
        ''' '''
        ''' Components of the ranking model:
        (1) scoring function: self.point_sf, (2) optimizer: self.optimizer '''
        self.point_sf = self.config_point_neural_scoring_function()
        self.config_optimizer()
        '''
        Components of the propensity model:
        (1) scoring function: self.propensity_sf, (2) optimizer: self.propensity_optimizer
        '''
        self.propensity_sf = self.config_propensity_scoring_function()
        self.config_propensity_optimizer()

    def config_propensity_scoring_function(self):
        serp_size = self.click_model.get_serp_size()

        propensity_sf = nn.Sequential()
        propensity_sf.add_module('L0', nn.Linear(serp_size, 1))
        propensity_sf.add_module('AF0', nn.ELU())
        return propensity_sf

    def get_propensity_parameters(self):
        return self.propensity_sf.parameters()

    def config_propensity_optimizer(self):
        '''
        Configure the optimizer correspondingly.
        '''
        if 'Adam' == self.opt:
            self.propensity_optimizer = optim.Adam(self.get_propensity_parameters(), lr = self.lr, weight_decay = self.weight_decay)
        elif 'RMS' == self.opt:
            self.propensity_optimizer = optim.RMSprop(self.get_propensity_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif 'Adagrad' == self.opt:
            self.propensity_optimizer = optim.Adagrad(self.get_propensity_parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        self.propensity_scheduler = StepLR(self.propensity_optimizer, step_size=20, gamma=0.5)

    def scheduler_step(self):
        self.scheduler.step()
        self.propensity_scheduler.step()

    def propensity_eval_mode(self):
        self.propensity_sf.eval()

    def propensity_train_mode(self):
        self.propensity_sf.train(mode=True)

    def x_train(self, train_data=None, epoch_k=None, burn_in_train_data=None, initial_ranker=None):
        ''' One epoch training '''

        ''' The burn-in process for initial_ranker '''
        if 1 == epoch_k:
            assert initial_ranker is not None
            initial_ranker.init()
            for epoch_i in range(1, self.num_burn_in_epoch):
                initial_ranker.train(train_data=burn_in_train_data, epoch_k=epoch_i,
                                     presort=True, label_type=LABEL_TYPE.MultiLabel)

        ''' Main training process based on unbiased signals'''
        # TODO: record clicks, etc or as a threshold
        num_queries = 0
        serp_size = self.click_model.serp_size
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            batch_size = batch_std_labels.size(0)
            num_queries += len(batch_ids)
            if self.gpu:
                batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device),\
                                                        batch_std_labels.to(self.device)
            ''' generate SERPs '''
            # -- eval_mode, i.e., being deployed as a system, respectively --
            if 1 == epoch_k:
                initial_ranker.eval_mode() # switch evaluation mode
                batch_system_preds = initial_ranker.predict(batch_q_doc_vectors)
            else:
                self.eval_mode()
                with torch.no_grad(): # deployed as a ranking system
                    batch_system_preds = self.predict(batch_q_doc_vectors)

            batch_system_desc_preds, batch_system_desc_pred_inds = torch.sort(batch_system_preds, dim=1, descending=True)
            batch_system_rankings = torch.gather(batch_std_labels, dim=1, index=batch_system_desc_pred_inds)
            batch_serp_std_labels = batch_system_rankings[:, 0:serp_size]
            # simulating users' serf-behaviors based on a click model
            batch_serp_clicks = self.click_model.surf_serp(batch_serp_std_labels, bool=False)

            batch_rank_reprs = self.get_rank_reprs(batch_size=batch_size, ranking_size=serp_size)
            self.propensity_eval_mode() # deployed as a propensity model
            with torch.no_grad():
                batch_system_propensity_scores = self.propensity_sf(batch_rank_reprs).view(batch_size, -1)
                batch_serp_system_obs_probs = F.softmax(batch_system_propensity_scores, dim=-1)
                batch_inverse_propensity_weights = self.get_normalized_weights(batch_size, batch_serp_system_obs_probs)

            with torch.no_grad():
                batch_serp_system_rele_scores = batch_system_desc_preds[:, 0:serp_size]
                batch_serp_system_rele_probs = F.softmax(batch_serp_system_rele_scores, dim=-1)
                batch_inverse_relevance_weights = self.get_normalized_weights(batch_size, batch_serp_system_rele_probs)

            # -- train_mode, i.e., being the optimizing target, respectively --
            if 1 == epoch_k: # required to feature vectors w.r.t. SERPs TODO: any efficient alternative ?
                list_serp_feats = []
                for i in range(batch_size):
                    serp_feats_inds = batch_system_desc_pred_inds[i, 0:serp_size]
                    serp_feats = batch_q_doc_vectors[i, serp_feats_inds, :]
                    list_serp_feats.append(serp_feats)
                batch_serp_q_doc_vectors = torch.stack(list_serp_feats, dim=0)

                self.train_mode()
                batch_serp_train_rele_scores = self.predict(batch_serp_q_doc_vectors)
                batch_serp_train_rele_probs = F.softmax(batch_serp_train_rele_scores, dim=-1)
            else:
                self.train_mode()
                batch_train_preds = self.predict(batch_q_doc_vectors)
                batch_train_desc_preds = torch.gather(batch_train_preds, dim=1, index=batch_system_desc_pred_inds)
                batch_serp_train_rele_scores = batch_train_desc_preds[:, 0:serp_size]
                batch_serp_train_rele_probs = F.softmax(batch_serp_train_rele_scores, dim=-1)

            batch_ranking_loss = self.listwise_softmax_loss(batch_serp_train_rele_probs, batch_serp_clicks,
                                                            batch_inverse_propensity_weights)
            self.optimizer.zero_grad()
            batch_ranking_loss.backward()
            self.optimizer.step()

            self.propensity_train_mode()
            batch_train_propensity_scores = self.propensity_sf(batch_rank_reprs).view(batch_size, -1)
            batch_serp_train_obs_probs = F.softmax(batch_train_propensity_scores, dim=-1)

            batch_propensity_model_loss = self.listwise_softmax_loss(batch_serp_train_obs_probs, batch_serp_clicks,
                                                                     batch_inverse_relevance_weights)
            self.propensity_optimizer.zero_grad()
            batch_propensity_model_loss.backward()
            self.propensity_optimizer.step()

    def get_normalized_weights(self, batch_size, batch_serp_x_probs):
        batch_normalized_weights = batch_serp_x_probs[:, 0].view(batch_size, -1) / batch_serp_x_probs
        return batch_normalized_weights

    def get_rank_reprs(self, batch_size, ranking_size):
        one_mat = torch.sparse.torch.eye(ranking_size)
        batch_rank_reprs = torch.unsqueeze(one_mat, dim=0).expand(batch_size, -1, -1)
        return batch_rank_reprs

    def listwise_softmax_loss(self, batch_probs, batch_labels, weights=None):
        '''
        Listwise softmax loss with propensity weighting.
        @param batch_probs:  [batch_size, ranking_size]. Each value is a probability
        @param batch_labels: [batch_size, ranking_size]. A value >= 1 means a relevant example
        @param weights:
        @param binarized_label: whether the input label is binary or not
        '''
        batch_neg_log_probs = F.binary_cross_entropy(input=batch_probs, target=batch_labels,
                                                     weight=weights, reduction='none')
        batch_loss = torch.sum(batch_neg_log_probs, dim=(1, 0))
        return batch_loss

###### Parameter of DLA ######

class DLAParameter(ModelParameter):
    ''' Parameter class for DLA '''
    def __init__(self, debug=False, para_json=None):
        super(DLAParameter, self).__init__(model_id='DLA', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for DLA
        """
        self.dla_para_dict = dict(model_id=self.model_id, num_session=10, num_burn_in_epoch=10)
        return self.dla_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        s1, s2 = (':', '\n') if log else ('_', '_')
        # using specified para-dict or inner para-dict
        dla_para_dict = given_para_dict if given_para_dict is not None else self.dla_para_dict

        num_session, num_burn_in_epoch = dla_para_dict['num_session'], dla_para_dict['num_burn_in_epoch']

        dla_para_str = s1.join(['Session', str(num_session), 'BurnIn', str(num_burn_in_epoch)])
        return dla_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for RankNet
        """
        if self.use_json:
            choice_num_session = self.json_dict['num_session']
            choice_num_burn_in_epoch = self.json_dict['num_burn_in_epoch']
        else:
            choice_num_session = [5] if self.debug else [10000]
            choice_num_burn_in_epoch = [10] if self.debug else [50]

        for num_session, num_burn_in_epoch in product(choice_num_session, choice_num_burn_in_epoch):
            self.dla_para_dict = dict(model_id=self.model_id,
                                      num_session=num_session, num_burn_in_epoch=num_burn_in_epoch)
            yield self.dla_para_dict
