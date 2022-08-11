
import torch
import torch.nn.functional as F

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.base.point_ranker import PointNeuralRanker


class XPointNeuralRanker(PointNeuralRanker):
    '''
    A one-size-fits-all neural ranker.
    Given the documents associated with the same query, this ranker scores each document independently.
    '''
    def __init__(self, id='XPointNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(XPointNeuralRanker, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)

    ''' >>>> X Ranking >>>> '''

    def x_train(self, burn_in_train_data, train_data, initial_ranker=None, epoch_k=None, click_model=None):
        '''
        One epoch training using the entire training data
        TODO: (1) initial_ranker can be itself; (2) perform burn-in based on epoch_k
        '''

        ''' The burn-in process for initial_ranker '''
        if 1 == epoch_k and initial_ranker is not None:
            initial_ranker.init()
            for epoch_k in range(1, 10):
                '''
                for batch_ids, batch_q_doc_vectors, batch_std_labels in burn_in_train_data:
                    print(batch_ids)
                    print(batch_q_doc_vectors.size())
                    print(batch_std_labels.size())
                '''
                initial_ranker.train(train_data=burn_in_train_data, epoch_k=epoch_k, presort=True,
                                     label_type=LABEL_TYPE.MultiLabel)
        elif initial_ranker is None:
            pass

        # configure deployed ranker
        # TODO : self copy
        product_ranker = initial_ranker
        product_ranker.eval_mode() # switch evaluation mode

        ''' Main training process based on unbiased signals'''
        self.train_mode()

        num_queries = 0
        size_serp = click_model.size_serp
        # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:
            batch_size = batch_std_labels.size(0)
            #print('batch_std_labels', batch_std_labels.size())
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = \
                batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            ''' get predicted rankings based on initial_ranker '''
            batch_preds = product_ranker.predict(batch_q_doc_vectors)
            #print('batch_preds', batch_preds.size())
            _, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
            batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

            #print('batch_predict_rankings[:, 0:size_serp]', batch_predict_rankings[:, 0:size_serp].size())
            serp_batch_std_labels = batch_predict_rankings[:, 0:size_serp]
            unbiased_batch_simulated_clicks = click_model.unbiased_surf_serp(serp_batch_std_labels)
            #print('batch_q_doc_vectors', batch_q_doc_vectors.size())
            #print('batch_pred_desc_inds', batch_pred_desc_inds.size())

            list_serp_feats = [] # TODO: efficient alternatives
            for i in range(batch_size):
                serp_feats_inds = batch_pred_desc_inds[i, 0:size_serp]
                serp_feats = batch_q_doc_vectors[0, serp_feats_inds, :]
                list_serp_feats.append(serp_feats)

            serp_batch_q_doc_vectors = torch.stack(list_serp_feats, dim=0)
            #print('serp_batch_q_doc_vectors', serp_batch_q_doc_vectors.size())

            self.x_train_op(serp_batch_q_doc_vectors, serp_batch_std_labels, unbiased_batch_simulated_clicks)

    def x_train_op(self, serp_batch_q_doc_vectors, serp_batch_std_labels, unbiased_batch_simulated_clicks, **kwargs):
        '''
        '''
        serp_batch_preds = self.forward(serp_batch_q_doc_vectors)
        return self.x_custom_loss_function(serp_batch_preds, serp_batch_std_labels, unbiased_batch_simulated_clicks, **kwargs)


    def x_custom_loss_function(self, batch_preds, batch_std_labels, propensity_weights=None, **kwargs):
        '''
        '''
        batch_loss = self.softmax_loss(batch_preds, batch_std_labels, propensity_weights)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()


    def sigmoid_loss_on_list(self, output, labels,
                             propensity_weights=None):
        """Computes pointwise sigmoid loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        criterion =  torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = criterion(output, labels) * propensity_weights
        return torch.mean(torch.sum(loss, dim=1))

    def pairwise_loss_on_list(self, output, labels,
                              propensity_weights=None):
        """Computes pairwise entropy loss.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
                relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        loss = None
        sliced_output = torch.unbind(output, dim=1)
        sliced_label = torch.unbind(labels, dim=1)
        sliced_propensity = torch.unbind(propensity_weights, dim=1)
        for i in range(len(sliced_output)):
            for j in range(i + 1, len(sliced_output)):
                cur_label_weight = torch.sign(
                    sliced_label[i] - sliced_label[j])
                cur_propensity = sliced_propensity[i] * \
                    sliced_label[i] + \
                    sliced_propensity[j] * sliced_label[j]
                cur_pair_loss = - \
                    torch.exp(
                        sliced_output[i]) / (torch.exp(sliced_output[i]) + torch.exp(sliced_output[j]))
                if loss is None:
                    loss = cur_label_weight * cur_pair_loss
                loss += cur_label_weight * cur_pair_loss * cur_propensity
        batch_size = labels.size()[0]
        return torch.sum(loss) / batch_size.type(torch.float32)

    def softmax_loss(self, output, labels, propensity_weights=None):
        """Computes listwise softmax loss without propensity weighting.

        Args:
            output: (torch.Tensor) A tensor with shape [batch_size, list_size]. Each value is
            the ranking score of the corresponding example.
            labels: (torch.Tensor) A tensor of the same shape as `output`. A value >= 1 means a
            relevant example.
            propensity_weights: (torch.Tensor) A tensor of the same shape as `output` containing the weight of each element.

        Returns:
            (torch.Tensor) A single value tensor containing the loss.
        """
        if propensity_weights is None:
            propensity_weights = torch.ones_like(labels)

        weighted_labels = (labels + 0.0000001) * propensity_weights
        label_dis = weighted_labels / \
            torch.sum(weighted_labels, 1, keepdim=True)
        label_dis = torch.nan_to_num(label_dis)
        loss = softmax_cross_entropy_with_logits(
            logits = output, labels = label_dis)* torch.sum(weighted_labels, 1)
        return torch.sum(loss) / torch.sum(weighted_labels)

def softmax_cross_entropy_with_logits(logits, labels):
    """Computes softmax cross entropy between logits and labels.

    Args:
        output: A tensor with shape [batch_size, list_size]. Each value is
        the ranking score of the corresponding example.
        labels: A tensor of the same shape as `output`. A value >= 1 means a
        relevant example.
    Returns:
        A single value tensor containing the loss.
    """
    #print('logits', logits.size())
    #print('labels', labels.size())

    loss = torch.sum(- labels * F.log_softmax(logits, -1), -1)
    return loss

    ''' <<<< X Ranking <<<< '''


