
import copy
from tqdm import tqdm

import torch

from ptranking.ltr_federation.base.federated_client import FederatedClient
#from ptranking.ltr_federation.base.point_federated_ranker import PointFederatedRanker
from ptranking.ltr_federation.federated_debiasing.federated_pdgd import Federated_PDGD

class FederatedServer():
    """
    Mimic federation framework
    The main mechanism is performing federated training over a group of clients
    """
    #def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model: CcmClickModel, sensitivity, epsilon, enable_noise, n_clients):
    #    # todo ? this init is not needed
    #    pass

    def __init__(self, sf_para_dict=None, model_para_dict=None, federation_para_dict=None, click_model=None):
        # todo-1
        # initialize the ranker, later for deepcopy (aiming for the same start point)
        # global_ranker vs initial_ranker
        #self.global_ranker = PointFederatedRanker(sf_para_dict=sf_para_dict)
        self.global_ranker = Federated_PDGD(sf_para_dict=sf_para_dict, model_para_dict=model_para_dict, click_model=click_model)
        self.generation_serial = 0 # recording the iteration serial of self.global_ranker

        self.federation_para_dict = federation_para_dict
        self.num_clients, self.interactions_per_feedback, self.per_interaction_update =\
            federation_para_dict["num_clients"], federation_para_dict["interactions_per_feedback"], federation_para_dict["per_interaction_update"]

    def init(self, train_data=None, test_data=None, data_dict=None):
        """ Conduct necessary initialization for federated learning """
        self.global_ranker.init()
        self.test_data = test_data
        self.train_data = train_data
        self.train_presort, self.test_presort = data_dict['train_presort'], data_dict['test_presort']

        seed = 1
        # seed = self.federation_para_dict["seed"]
        # todo
        # np.random.seed(seed)

        self.federated_client_pool = \
            [FederatedClient(dataset=self.train_data, presort=self.train_presort,
                             global_ranker=copy.deepcopy(self.global_ranker), generation_serial=self.generation_serial,
                             seed=seed * self.num_clients + client_id, sensitivity=None, epsilon=None, enable_noise=None,
                             n_clients=None)
             for client_id in range(self.num_clients)]

    def federated_train(self, epoch_k=None):
        self.generation_serial = epoch_k

        trend_avg_client_ndcg = []
        list_client_feedbacks = []
        sum_client_ndcg_at_k = torch.zeros(1)
        for client in self.federated_client_pool:
            dict_batch_gradients, dict_weights, client_ndcg_at_k = \
                client.on_device_learning(self.interactions_per_feedback, self.per_interaction_update)
            if len(dict_batch_gradients) > 0:
                list_client_feedbacks.append(dict_batch_gradients)
            # online evaluation
            sum_client_ndcg_at_k += torch.sum(client_ndcg_at_k)

        # online-line metrics
        trend_avg_client_ndcg.append(sum_client_ndcg_at_k/self.num_clients)

        self.federated_aggregation(list_client_feedbacks)

        # the server send the newly trained model to every client
        for client in self.federated_client_pool:
            client.fetch_newest_global_ranker(newest_global_ranker=self.global_ranker, generation_serial=self.generation_serial)

        #return TrainResult(ranker=self.global_ranker, ndcg_server=ndcg_server, mrr_server=mrr_server, ndcg_client=trend_avg_client_ndcg)

    def federated_aggregation(self, list_client_feedbacks):
        #todo the current method is too simple, namely averaging
        list_names = self.global_ranker.get_parameter_names()
        #print('list_names', list_names)

        dict_aggregated_gradients = {}
        for name in list_names:
            dict_aggregated_gradients[name] = 0
            #print(name)
            for dict_batch_gradients in list_client_feedbacks:
                #print('!!!!!')
                #print(dict_batch_gradients.keys())
                #print(dict_batch_gradients[name])
                dict_aggregated_gradients[name] = dict_aggregated_gradients[name] + dict_batch_gradients[name]

        for name in dict_aggregated_gradients:
            dict_aggregated_gradients[name] = dict_aggregated_gradients[name]/self.num_clients

        self.global_ranker.batch_gradient_descent_update(dict_aggregated_gradients)

    def save(self, dir, name):
        self.global_ranker.save(dir=dir, name=name)

    def evaluate(self):
        """ Evaluating the performance of final global_ranker against test data """
        pass

    """
    compute_performance(ranker as argument) : will be called within both training and evaluation
    evaluation
    """
