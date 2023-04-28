
import torch

from itertools import product

from ptranking.ltr_x.debiasing.pdgd import compute_pdgd_loss
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_federation.base.point_federated_ranker import PointFederatedRanker
from ptranking.ltr_adhoc.util.sampling_utils import sample_ranking_PL_gumbel_softmax
from ptranking.ltr_federation.util.eval_utils import compute_metric_score

class Federated_PDGD(PointFederatedRanker):
    '''
    Oosterhuis, Harrie, and Maarten de Rijke. "Differentiable unbiased online learning to rank."
    In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1293-1302. ACM, 2018.
    '''
    def __init__(self, id='Federated_PDGD', sf_para_dict=None, model_para_dict=None, click_model=None,
                 weight_decay=1e-3, gpu=False, device=None):
        super(Federated_PDGD, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.click_model = click_model
        self.num_session, self.num_burn_in_epoch = model_para_dict['num_session'], model_para_dict['num_burn_in_epoch']

    def online_learning_to_rank(self, batch_q_doc_vectors, batch_std_labels, compute_performance=False, presort=False):
        self.eval_mode() # With the eval_mode, the federated ranker is deployed as a ranking system to generate SERPs
        with torch.no_grad():  # deployed as a real ranking system
            batch_system_preds = self.predict(batch_q_doc_vectors)

        batch_system_desc_pred_inds = sample_ranking_PL_gumbel_softmax(
            batch_preds=batch_system_preds, only_indices=True, temperature=1.0, device=self.device)

        batch_system_rankings = torch.gather(batch_std_labels, dim=1, index=batch_system_desc_pred_inds)

        # evaluate the online-ltr performance
        if compute_performance:
            batch_ndcg_at_k = compute_metric_score(batch_predict_rankings=batch_system_rankings,
                                                   batch_std_labels=batch_std_labels,
                                                   k=10, presort=presort, device=self.device)

        serp_size = self.click_model.serp_size
        batch_serp_std_labels = batch_system_rankings[:, 0:serp_size]
        # simulating users' serf-behaviors based on a click model
        batch_serp_clicks = self.click_model.surf_serp(batch_serp_std_labels, bool=False)
        # todo : 考えること...user modelを追加した場合値がおかしくなる
        #batch_serp_clicks = self.click_model.unbiased_surf_serp(batch_serp_std_labels, user_model="PERFECT")

        self.train_mode()
        batch_train_preds = self.predict(batch_q_doc_vectors)
        batch_train_desc_preds = torch.gather(batch_train_preds, dim=1, index=batch_system_desc_pred_inds)
        batch_serp_train_rele_scores = batch_train_desc_preds[:, 0:serp_size]

        has_train_signal, per_query_pdgd_loss = compute_pdgd_loss(batch_serp_clicks=batch_serp_clicks, batch_serp_scores=batch_serp_train_rele_scores)

        if compute_performance:
            return per_query_pdgd_loss, batch_ndcg_at_k
        else:
            return per_query_pdgd_loss, None

###### Parameter of Federated_PDGD ######

class Federated_PDGDParameter(ModelParameter):
    ''' Parameter class for Federated_PDGD '''
    def __init__(self, debug=False, para_json=None):
        super(Federated_PDGDParameter, self).__init__(model_id='Federated_PDGD', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for DLA
        """
        self.pdgd_para_dict = dict(model_id=self.model_id, num_session=10, num_burn_in_epoch=10)
        return self.pdgd_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        s1, s2 = (':', '\n') if log else ('_', '_')
        # using specified para-dict or inner para-dict
        pdgd_para_dict = given_para_dict if given_para_dict is not None else self.pdgd_para_dict

        num_session, num_burn_in_epoch = pdgd_para_dict['num_session'], pdgd_para_dict['num_burn_in_epoch']

        pdgd_para_str = s1.join(['Session', str(num_session), 'BurnIn', str(num_burn_in_epoch)])
        #print("pdgd_para_str:{}".format(pdgd_para_str))
        return pdgd_para_str

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
            self.pdgd_para_dict = dict(model_id=self.model_id,
                                      num_session=num_session, num_burn_in_epoch=num_burn_in_epoch)
            yield self.pdgd_para_dict
