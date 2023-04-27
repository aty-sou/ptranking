
from ptranking.ltr_x.base.x_list_ranker import XListNeuralRanker
from ptranking.ltr_x.base.x_point_ranker import XPointNeuralRanker

class XNeuralRanker(XPointNeuralRanker, XListNeuralRanker):
    '''
    A combination of PointNeuralRanker & PENeuralRanker
    '''
    def __init__(self, id='XNeuralRanker', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        self.id = id
        self.gpu, self.device = gpu, device

        self.sf_para_dict = sf_para_dict
        self.sf_id = sf_para_dict['sf_id']
        assert self.sf_id in ['pointsf', 'listsf']

        self.opt, self.lr = sf_para_dict['opt'], sf_para_dict['lr']
        self.weight_decay = weight_decay

        self.stop_check_freq = 10

        if 'pointsf' == self.sf_id: # corresponding to the concatenation operation, i.e., q_repr + doc_repr + latent_cross
            #self.sf_para_dict[self.sf_id]['num_features'] *= 3
            pass
        elif 'listsf' == self.sf_id:
            self.encoder_type = self.sf_para_dict[self.sf_para_dict['sf_id']]['encoder_type']


    def init(self):
        if self.sf_id.startswith('pointsf'):
            XPointNeuralRanker.init(self)
        elif self.sf_id.startswith('listsf'):
            XListNeuralRanker.init(self)

    def get_parameters(self):
        if self.sf_id.startswith('pointsf'):
            return XPointNeuralRanker.get_parameters(self)
        elif self.sf_id.startswith('listsf'):
            return XListNeuralRanker.get_parameters(self)

    def div_forward(self, q_repr, doc_reprs):
        if self.sf_id.startswith('pointsf'):
            return XPointNeuralRanker.div_forward(self, q_repr, doc_reprs)
        elif self.sf_id.startswith('listsf'):
            return XListNeuralRanker.div_forward(self, q_repr, doc_reprs)

    def eval_mode(self):
        if self.sf_id.startswith('pointsf'):
            XPointNeuralRanker.eval_mode(self)
        elif self.sf_id.startswith('listsf'):
            XListNeuralRanker.eval_mode(self)

    def train_mode(self):
        if self.sf_id.startswith('pointsf'):
            XPointNeuralRanker.train_mode(self)
        elif self.sf_id.startswith('listsf'):
            XListNeuralRanker.train_mode(self)

    def save(self, dir, name):
        if self.sf_id.startswith('pointsf'):
            XPointNeuralRanker.save(self, dir=dir, name=name)
        elif self.sf_id.startswith('listsf'):
            XListNeuralRanker.save(self, dir=dir, name=name)

    def load(self, file_model, device):
        if self.sf_id.startswith('pointsf'):
            XPointNeuralRanker.load(self, file_model=file_model, device=device)
        elif self.sf_id.startswith('listsf'):
            XListNeuralRanker.load(self, file_model=file_model, device=device)

    def get_tl_af(self):
        if self.sf_id.startswith('pointsf'):
            XPointNeuralRanker.get_tl_af(self)
        elif self.sf_id.startswith('listsf'):
            self.sf_para_dict[self.sf_para_dict['sf_id']]['AF']
