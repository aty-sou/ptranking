
import torch

class ClickModel():
    def __init__(self, id, gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

    def surf_serp(self):
        pass

    def unbiased_surf_serp(self):
        pass


class PBM(ClickModel):
    def __init__(self, id='PBM', eta=1., epsilon=0., serp_size=10, max_label=None, gpu=False, device=None):
        super(PBM, self).__init__(id=id, gpu=gpu, device=device)
        self.eta = eta
        self.epsilon = epsilon
        self.gpu, self.device = gpu, device

        self.max_label = torch.tensor([max_label], device=self.device)
        self.serp_size = serp_size
        ranks = torch.arange(serp_size, dtype=torch.int, device=self.device) + 1.0
        self.examination_probs = (1.0/ranks) ** self.eta

    def get_serp_size(self):
        return self.serp_size

    def get_rele_probs(self, batch_std_labels):
        if self.epsilon > 0:
            batch_rele_probs = (torch.pow(2., batch_std_labels) - 1.) / (torch.pow(2., self.max_label) - 1.) \
                                * (1.0 - self.epsilon) + self.epsilon
        else:
            batch_rele_probs = (torch.pow(2., batch_std_labels) - 1.) / (torch.pow(2., self.max_label) - 1.)

        return batch_rele_probs

    def surf_serp(self, batch_predicted_rankings, bool=False):
        '''
        Surf the SERP (search result page)
        @param batch_predicted_rankings:
        @param bool:
        @return: click information
        '''
        batch_rele_probs = self.get_rele_probs(batch_predicted_rankings)
        batch_click_probs = self.examination_probs * batch_rele_probs

        bool_batch_simulated_clicks = torch.rand(size=batch_predicted_rankings.size()) < batch_click_probs
        if bool:
            return bool_batch_simulated_clicks
        else:
            batch_simulated_clicks = torch.zeros_like(batch_predicted_rankings)
            batch_simulated_clicks[bool_batch_simulated_clicks] = 1.
        
        return batch_simulated_clicks

    def unbiased_surf_serp(self, batch_predicted_rankings):
        '''
        Surf the SERP (search result page) with inverse propensity scoring (IPS)
        @param batch_predicted_rankings:
        @param bool:
        @return: click information
        '''
        batch_rele_probs = self.get_rele_probs(batch_predicted_rankings)
        batch_click_probs = self.examination_probs * batch_rele_probs

        bool_batch_simulated_clicks = torch.rand(size=batch_predicted_rankings.size()) < batch_click_probs
        batch_simulated_clicks = torch.zeros_like(batch_predicted_rankings)
        batch_simulated_clicks[bool_batch_simulated_clicks] = 1.
        unbiased_batch_simulated_clicks = batch_simulated_clicks / self.examination_probs
        return unbiased_batch_simulated_clicks






class UBM(ClickModel):
    def __init__(self, id='UBM'):
        super(PBM, self).__init__()

    def simulate_click_behavior(self, batch_predicted_rankings):
        pass

class CascadeModel(ClickModel):
    def __init__(self, id='CascadeModel'):
        super(PBM, self).__init__()

    def simulate_click_behavior(self, batch_predicted_rankings):
        pass
