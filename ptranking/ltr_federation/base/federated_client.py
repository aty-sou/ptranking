
import copy
import numpy as np
import torch
from ptranking.ltr_federation.util.dp import gamma_noise

class FederatedClient():
    """
    Mimic client
    The main component is a point-ranker that explicitly & manually perform optimization, such as
    (1) explicitly getting the parameters of the inner scoring function (i.e., self.get_parameters())
    (2) explicitly getting the the gradients of parameters (i.e., get_gradients())
    """
    def __init__(self, dataset=None, presort=None, global_ranker=None, generation_serial=None,
                 seed=None, sensitivity=None, epsilon=None, enable_noise=True, n_clients=None):
        self.dataset = dataset
        self.presort = presort
        self.federated_ranker = global_ranker
        self.generation_serial = 1 if generation_serial is None else generation_serial # indicating the n-th generation given the initial global ranker
        self.random_state = np.random.RandomState(seed)
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.enable_noise = True  # set True if you want to add DP noise, otherwise set False
        self.n_clients = n_clients

    def aggregate_gradients(self, dict_batch_gradients, dict_gradients):
        for name, gradient in dict_gradients.items():
            if name in dict_batch_gradients:
                dict_batch_gradients[name] = dict_batch_gradients[name] + gradient
            else:
                dict_batch_gradients[name] = gradient

    def on_device_learning(self, interactions_per_feedback, per_interaction_update=None):
        dict_batch_gradients = {}
        # randomly choose queries for simulation on each client (number of queries based on the set n_interactions)
        list_inds = self.random_state.randint(self.dataset.__len__(), size=interactions_per_feedback)

        sum_ndcg_at_k = torch.zeros(1)
        for i in range(interactions_per_feedback):
            q_index = list_inds[i]
            # the batch dimension is not automatically activated when using __getitem__()
            qid, q_doc_vectors, std_labels = self.dataset.__getitem__(q_index)
            batch_q_doc_vectors, batch_std_labels = torch.unsqueeze(q_doc_vectors, dim=0), torch.unsqueeze(std_labels, dim=0)

            per_query_pdgd_loss, batch_ndcg_at_k = \
                self.federated_ranker.online_learning_to_rank(batch_q_doc_vectors, batch_std_labels,
                                                              compute_performance=True, presort=self.presort)
            sum_ndcg_at_k += torch.sum(batch_ndcg_at_k)  # due to batch processing

            if per_query_pdgd_loss is None:
                continue
            else:
                per_query_pdgd_loss.backward()

                if per_interaction_update:
                    dict_gradients = self.federated_ranker.gradient_descent_update()
                else:
                    dict_gradients = self.federated_ranker.get_named_gradients()

                self.aggregate_gradients(dict_batch_gradients, dict_gradients)

        if not per_interaction_update and not len(dict_batch_gradients)==0:
            self.federated_ranker.batch_gradient_descent_update(dict_batch_gradients)

        dict_weights = self.federated_ranker.get_updated_weights()

        #if add_noise:
        #    pass

        # print("enable noise:{}".format(self.enable_noise))

        if self.enable_noise:
            # gammma noiseを追加している

            noise = gamma_noise(dict_weights["ff_2.weight"].size()[1], self.sensitivity, self.epsilon, self.n_clients)
            # noise = gamma_noise(np.shape(dict_weights), self.sensitivity, self.epsilon, self.n_clients)
            # print(noise)

            dict_weights["ff_2.weight"] += noise

        #return ClientMessage
        avg_ndcg_at_k = sum_ndcg_at_k / interactions_per_feedback

        return dict_batch_gradients, dict_weights, avg_ndcg_at_k

    def fetch_newest_global_ranker(self, newest_global_ranker, generation_serial=None):
        self.federated_ranker = copy.deepcopy(newest_global_ranker)
        self.generation_serial = generation_serial
