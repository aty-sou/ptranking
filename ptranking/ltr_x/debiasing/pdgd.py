
import torch

from itertools import product

from ptranking.base.point_ranker import PointNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.sampling_utils import sample_ranking_PL_gumbel_softmax

def get_PL_ranking_probability(batch_serp_scores):
    m, _ = torch.max(batch_serp_scores, dim=1, keepdim=True)
    y = batch_serp_scores - m
    y = torch.exp(y)
    y_backward_cumsum = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1), dims=[1])
    batch_logcumsumexps = torch.log(y_backward_cumsum) + m  # corresponding to the '-m' operation
    batch_pl_probs = torch.exp(torch.sum((batch_serp_scores - batch_logcumsumexps), dim=1))
    return batch_pl_probs

def compute_pdgd_loss(batch_serp_clicks, batch_serp_scores):
    batch_size, serp_size = batch_serp_clicks.size()
    # note: some pairs including documents which we are not sure about their observation status
    batch_preference_cmps = torch.unsqueeze(batch_serp_clicks, dim=2) - torch.unsqueeze(batch_serp_clicks, dim=1)
    # print('batch_preference_cmps', batch_preference_cmps.size())
    # print('batch_preference_cmps', batch_preference_cmps)

    batch_serp_obs = torch.ones_like(batch_serp_clicks)
    batch_backward_cumsums = torch.flip(torch.cumsum(torch.flip(batch_serp_clicks, dims=[1]), dim=1), dims=[1])
    # a value > 0 denotes an observation
    batch_serp_obs[:, 1:serp_size] = batch_backward_cumsums[:, 0:serp_size - 1]
    #print('batch_serp_obs', batch_serp_obs)

    # a value > 0 denotes a preference between a pair of documents both of which are observed based on our assumption
    batch_3d_obs_mask = torch.unsqueeze(batch_serp_obs, dim=2) * torch.unsqueeze(batch_serp_obs, dim=1)
    # print('batch_3d_obs_mask', batch_3d_obs_mask)

    # a True value indicates an inferred >_c pairs
    batch_3d_inferred_preferences = (batch_preference_cmps > 0) & (batch_3d_obs_mask > 0)
    #print('batch_3d_inferred_preferences', batch_3d_inferred_preferences)

    list_per_query_pairs = []
    for i in range(batch_size):  # per-query computation
        # skip if there is no click for a serp since there would be no inferred preference pairs
        if 0>= torch.sum(batch_serp_clicks[i, :]):
            continue

        original_serp_ranking = batch_serp_scores[i, :]
        serp_pl_prob = get_PL_ranking_probability(original_serp_ranking.view(1, -1))

        inferred_2d_preferences = batch_3d_inferred_preferences[i, :, :]

        # note: gradient is required
        serp_ranking_as_col_vec = torch.unsqueeze(original_serp_ranking, dim=1)
        pairwise_denominators = serp_ranking_as_col_vec + torch.unsqueeze(original_serp_ranking, dim=0)
        pairwise_beat_probs = serp_ranking_as_col_vec / pairwise_denominators
        inferred_preference_probs = pairwise_beat_probs[inferred_2d_preferences]
        #print('inferred_preference_probs', inferred_preference_probs.size())
        #print('inferred_preference_probs', inferred_preference_probs)

        with torch.no_grad(): # gradient is not required during the coefficient computation
            preference_inds = torch.nonzero(inferred_2d_preferences, as_tuple=False)
            #print('preference_inds', preference_inds)
            preference_ascend_inds, _ = torch.sort(preference_inds, descending=False)
            #print('preference_ascend_inds', preference_ascend_inds)
            num_pairs = preference_ascend_inds.size(0)

            ascend_pair_heads = torch.index_select(original_serp_ranking, dim=0, index=preference_ascend_inds[:, 0])
            # print('ascend_pair_heads', ascend_pair_heads)
            ascend_pair_tails = torch.index_select(original_serp_ranking, dim=0, index=preference_ascend_inds[:, 1])
            # print('ascend_pair_tails', ascend_pair_tails)
            source_pair_scores = torch.stack([ascend_pair_heads, ascend_pair_tails], dim=1)
            #print('source_pair_scores', source_pair_scores)

            # the number of swapped rankings is equal to the number of inferred preference pairs
            expanded_orig_serp_rankings = torch.clone(original_serp_ranking).detach().expand(num_pairs, -1)
            #print('expanded_orig_serp_rankings', expanded_orig_serp_rankings)
            swapped_preference_ascend_inds = torch.flip(preference_ascend_inds, dims=[1])
            #print('swapped_preference_ascend_inds', swapped_preference_ascend_inds)
            swapped_serp_rankings = expanded_orig_serp_rankings.scatter(dim=1, index=swapped_preference_ascend_inds,
                                                                        src=source_pair_scores)
            #print('swapped_serp_rankings', swapped_serp_rankings)
            swapped_serp_pl_porbs = get_PL_ranking_probability(swapped_serp_rankings)
            pairwise_rhos = swapped_serp_pl_porbs / (serp_pl_prob + swapped_serp_pl_porbs)
            #print('pairwise_rhos', pairwise_rhos)

        weighted_inferred_preference_probs = pairwise_rhos * inferred_preference_probs
        #print('weighted_inferred_preference_probs', weighted_inferred_preference_probs)
        list_per_query_pairs.append(weighted_inferred_preference_probs)
        #print('==', list_per_query_pairs)

    has_train_signal = False
    if len(list_per_query_pairs) > 0:
        has_train_signal = True
        # print('---list_per_query_pairs', list_per_query_pairs)
        per_query_pdgd_loss = -torch.sum(torch.cat(list_per_query_pairs, dim=0))
        return has_train_signal, per_query_pdgd_loss
    else:
        return has_train_signal, None

class PDGD(PointNeuralRanker):
    '''
    Oosterhuis, Harrie, and Maarten de Rijke. "Differentiable unbiased online learning to rank."
    In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1293-1302. ACM, 2018.
    '''
    def __init__(self, id='PDGD', sf_para_dict=None, model_para_dict=None, click_model=None,
                 weight_decay=1e-3, gpu=False, device=None):
        super(PDGD, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.click_model = click_model
        self.num_session, self.num_burn_in_epoch = model_para_dict['num_session'], model_para_dict['num_burn_in_epoch']

    def scheduler_step(self):
        self.scheduler.step()

    def x_train(self, train_data=None, epoch_k=None, burn_in_train_data=None, initial_ranker=None):
        ''' One epoch training '''
        # TODO: using **kwargs to cover non-used arguments
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
            # -- eval_mode, i.e., being deployed as a ranking system --
            self.eval_mode()
            with torch.no_grad(): # deployed as a ranking system
                batch_system_preds = self.predict(batch_q_doc_vectors)

            batch_system_desc_pred_inds = sample_ranking_PL_gumbel_softmax(
                batch_preds=batch_system_preds, only_indices=True, temperature=1.0, device=self.device)

            batch_system_rankings = torch.gather(batch_std_labels, dim=1, index=batch_system_desc_pred_inds)
            batch_serp_std_labels = batch_system_rankings[:, 0:serp_size]
            # simulating users' serf-behaviors based on a click model
            batch_serp_clicks = self.click_model.surf_serp(batch_serp_std_labels, bool=False)

            # -- train_mode, i.e., being the optimizing target, respectively --
            self.train_mode()
            batch_train_preds = self.predict(batch_q_doc_vectors)
            batch_train_desc_preds = torch.gather(batch_train_preds, dim=1, index=batch_system_desc_pred_inds)
            batch_serp_train_rele_scores = batch_train_desc_preds[:, 0:serp_size]

            has_train_signal, per_query_pdgd_loss = compute_pdgd_loss(batch_serp_clicks=batch_serp_clicks,
                                                                      batch_serp_scores=batch_serp_train_rele_scores)
            if has_train_signal:
                self.optimizer.zero_grad()
                per_query_pdgd_loss.backward()
                self.optimizer.step()
            else:
                continue

###### Parameter of PDGD ######

class PDGDParameter(ModelParameter):
    ''' Parameter class for PDGD '''
    def __init__(self, debug=False, para_json=None):
        super(PDGDParameter, self).__init__(model_id='PDGD', para_json=para_json)
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
