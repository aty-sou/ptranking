import copy

from ptranking.ltr_x.base.x_point_ranker import XPointNeuralRanker
from ptranking.ltr_federation.base.federated_ranker import FederatedRanker

class PointFederatedRanker(XPointNeuralRanker, FederatedRanker):
    """
    Mimic client
    The main component is a point-ranker that explicitly & manually perform optimization, such as
    (1) explicitly getting the parameters of the inner scoring function (i.e., self.get_parameters())
    (2) explicitly getting the the gradients of parameters (i.e., get_gradients())
    """
    #def __init__(self, dataset: LetorDataset, init_model, seed: int, click_model: CcmClickModel, sensitivity, epsilon, enable_noise, n_clients):
    #    pass
    def __init__(self, id='PointFederatedRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(PointFederatedRanker, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.learning_rate_decay = 1.0

    def init(self):
        """ self.config_optimizer() is not needed due to federated optimization """
        self.point_sf = self.config_point_neural_scoring_function()

    def get_named_parameters(self):
        # it seems not useful
        return self.point_sf.named_parameters()

    def get_parameter_names(self):
        # it seems not useful
        list_names = []
        for name, _ in self.point_sf.named_parameters():
            list_names.append(name)
        return list_names

    def get_gradients(self):
        # it seems not useful
        for param in self.get_parameters():
            yield param.grad

    def get_named_gradients(self):
        # it seems not useful
        dict_gradients = {}
        for name, param in self.get_named_parameters():
            #yield (name, param.grad.data)
            dict_gradients[name] = copy.deepcopy(param.grad.data)

        return dict_gradients

    def gradient_descent_update(self):
        """
        Update the inner ranker (i.e., scoring function) based on gradient descent algorithm
        todo: have a look at online reference: manually updating models, e.g., with decay, learning-rate, etc.
        #手动更新参数
        w1.data.zero_() #BP求导更新参数之前，需要先把导数置0，否则会积累梯度
        w1.data.sub_(lr*w1.grad.data)
        """
        dict_gradients = {}
        for name, param in self.get_named_parameters():
            # is this right for assigning to zeros ? todo check the meaning of each part
            param.data.zero_()
            param.data.sub_(self.lr * param.grad.data)
            dict_gradients[name] = copy.deepcopy(param.grad.data)

        return dict_gradients

    def batch_gradient_descent_update(self, dict_batch_gradients):
        #print('=====')
        #print(dict_batch_gradients.keys())
        #print('------')
        for name, param in self.get_named_parameters():
            #print(name)
            param.data.sub_(self.lr * dict_batch_gradients[name])

        self.lr *= self.learning_rate_decay

    def get_updated_weights(self):
        dict_weights = {}
        for name, param in self.get_named_parameters():
            dict_weights[name] = copy.deepcopy(param.data)

        return dict_weights

