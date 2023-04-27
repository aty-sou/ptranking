
import sys
import datetime
import numpy as np

import torch

from ptranking.base.ranker import LTRFRAME_TYPE
from ptranking.ltr_x.click_model.pbm import PBM
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from ptranking.ltr_adhoc.eval.parameter import SummaryTape, ValidationTape, CVTape
from ptranking.ltr_x.eval.x_parameter import XCVTape, XValidationTape
from ptranking.data.data_utils import LTRDataset, SPLIT_TYPE, LETORSampler, LETORPercentSampler
from ptranking.ltr_x.eval.x_parameter import XDataSetting, XEvalSetting, XScoringFunctionParameter

from ptranking.ltr_adhoc.pointwise.rank_mse import RankMSE
from ptranking.ltr_x.debiasing.pdgd import PDGD, PDGDParameter
from ptranking.ltr_x.debiasing.dla import DLA, DLAParameter

LTR_X_MODEL = ['DLA', 'PDGD', 'PairwiseDebiasing', 'PolicyFairness', 'ProbFairness', 'ProbFairnessGlobal']
LTR_X_MODEL_Debiasing = ['DLA', 'PDGD', 'PairwiseDebiasing']
LTR_X_MODEL_Fairness = ['PolicyFairness', 'ProbFairness', 'ProbFairnessGlobal']
LTR_X_MODEL_Debiasing_Fairness = ['']

class XLTREvaluator(LTREvaluator):
    def __init__(self, frame_id=LTRFRAME_TYPE.X, cuda=None):
        super(XLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)

    def setup_eval(self, data_dict, eval_dict, sf_para_dict, model_para_dict):
        """
        Finalize the evaluation setting correspondingly
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        sf_para_dict[sf_para_dict['sf_id']].update(dict(num_features=data_dict['num_features']))

        self.dir_run  = self.setup_output(data_dict, eval_dict)

        if eval_dict['do_log'] and not self.eval_setting.debug:
            time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
            sys.stdout = open(self.dir_run + '_'.join(['log', time_str])+'.txt', "w")

        #if self.do_summary: self.summary_writer = SummaryWriter(self.dir_run + 'summary')
        """
        Aiming for efficient batch processing, please use a large batch_size, e.g., {train_rough_batch_size, validation_rough_batch_size, test_rough_batch_size = 300, 300, 300}
        """
        #assert data_dict['train_rough_batch_size'] > 1


    def load_ranker(self, sf_para_dict, model_para_dict, click_model=None):
        """
        Load a ranker correspondingly
        :param sf_para_dict:
        :param model_para_dict:
        :param kwargs:
        :return:
        """
        model_id = model_para_dict['model_id']

        if model_id in ['DLA', 'PairwiseDebiasing']: # where burn_in_ranker is required
            burn_in_ranker = RankMSE(sf_para_dict=sf_para_dict, gpu=self.gpu, device=self.device)

            ranker = globals()[model_id](sf_para_dict=sf_para_dict, model_para_dict=model_para_dict,
                                         click_model=click_model, gpu=self.gpu, device=self.device)
            return burn_in_ranker, ranker
        elif model_id in ['PDGD']: # where burn_in_ranker is not required
            burn_in_ranker = None
            ranker = globals()[model_id](sf_para_dict=sf_para_dict, model_para_dict=model_para_dict,
                                         click_model=click_model, gpu=self.gpu, device=self.device)
            return burn_in_ranker, ranker
        elif model_id in LTR_X_MODEL_Fairness:
            ranker = globals()[model_id](sf_para_dict=sf_para_dict, model_para_dict=model_para_dict,
                                         gpu=self.gpu, device=self.device)
            return ranker
        else:
            raise NotImplementedError


    def set_data_setting(self, debug=False, data_id=None, dir_data=None, x_data_json=None):
        if x_data_json is not None:
            self.data_setting = XDataSetting(x_data_json=x_data_json)
        else:
            self.data_setting = XDataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def set_eval_setting(self, debug=False, dir_output=None, x_eval_json=None):
        if x_eval_json is not None:
            self.eval_setting = XEvalSetting(debug=debug, x_eval_json=x_eval_json)
        else:
            self.eval_setting = XEvalSetting(debug=debug, dir_output=dir_output)

    def set_scoring_function_setting(self, debug=None, sf_id=None, sf_json=None):
        if sf_json is not None:
            self.sf_parameter = XScoringFunctionParameter(sf_json=sf_json)
        else:
            self.sf_parameter = XScoringFunctionParameter(debug=debug, sf_id=sf_id)

    def set_model_setting(self, debug=False, model_id=None, para_json=None):
        if para_json is not None:
            self.model_parameter = globals()[model_id + "Parameter"](para_json=para_json)
        else:
            self.model_parameter = globals()[model_id + "Parameter"](debug=debug)

    def load_data_debiasing(self, eval_dict, data_dict, fold_k):
        """
        Load the dataset correspondingly.
        :param eval_dict:
        :param data_dict:
        :param fold_k:
        :param model_para_dict:
        :return:
        """
        file_train, file_vali, file_test = self.determine_files(data_dict, fold_k=fold_k)

        input_eval_dict = eval_dict if eval_dict['mask_label'] else None # required when enabling masking data

        _train_data = LTRDataset(file=file_train, split_type=SPLIT_TYPE.Train, presort=data_dict['train_presort'],
                                 data_dict=data_dict, eval_dict=input_eval_dict)
        train_letor_sampler = LETORSampler(data_source=_train_data, rough_batch_size=data_dict['train_rough_batch_size'])
        train_loader = torch.utils.data.DataLoader(_train_data, batch_sampler=train_letor_sampler, num_workers=0)

        # TODO: percent -> json file
        burn_in_train_letor_sampler = LETORPercentSampler(data_source=_train_data, percent=.1)
        burn_in_train_loader = torch.utils.data.DataLoader(_train_data, batch_sampler=burn_in_train_letor_sampler, num_workers=0)

        _test_data = LTRDataset(file=file_test, split_type=SPLIT_TYPE.Test, data_dict=data_dict, presort=data_dict['test_presort'])
        test_letor_sampler = LETORSampler(data_source=_test_data, rough_batch_size=data_dict['test_rough_batch_size'])
        test_loader = torch.utils.data.DataLoader(_test_data, batch_sampler=test_letor_sampler, num_workers=0)

        if eval_dict['do_validation'] or eval_dict['do_summary']: # vali_data is required
            _vali_data = LTRDataset(file=file_vali, split_type=SPLIT_TYPE.Validation, data_dict=data_dict, presort=data_dict['validation_presort'])
            vali_letor_sampler = LETORSampler(data_source=_vali_data, rough_batch_size=data_dict['validation_rough_batch_size'])
            vali_loader = torch.utils.data.DataLoader(_vali_data, batch_sampler=vali_letor_sampler, num_workers=0)
        else:
            vali_loader = None

        return burn_in_train_loader, train_loader, test_loader, vali_loader


    def kfold_cv_eval(self, data_dict=None, eval_dict=None, sf_para_dict=None, model_para_dict=None):

        #print("**data_dict**:{}".format(data_dict))

        self.display_information(data_dict, model_para_dict)
        self.check_consistency(data_dict, eval_dict, sf_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, model_para_dict)

        model_id = model_para_dict['model_id']
        fold_num, label_type, max_label = data_dict['fold_num'], data_dict['label_type'], data_dict['max_rele_level']
        train_presort, validation_presort, test_presort = data_dict['train_presort'], data_dict['validation_presort'], \
                                                          data_dict['test_presort']

        if model_id in LTR_X_MODEL_Debiasing: # debiasing methods
            click_model = PBM(max_label=4, gpu=self.gpu, device=self.device)
            burn_in_ranker, ranker = self.load_ranker(model_para_dict=model_para_dict, sf_para_dict=sf_para_dict,
                                                      click_model=click_model)
        elif model_id in LTR_X_MODEL_Fairness:
            ranker = self.load_ranker(model_para_dict=model_para_dict, sf_para_dict=sf_para_dict)
            ranker.uniform_eval_setting(eval_dict=eval_dict)
        elif model_id in LTR_X_MODEL_Debiasing_Fairness:
            raise NotImplementedError

        # for quick access of common evaluation settings
        epochs, loss_guided = eval_dict['epochs'], eval_dict['loss_guided']
        vali_k, log_step, cutoffs = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        do_vali, vali_metric, do_summary = eval_dict['do_validation'], eval_dict['vali_metric'], eval_dict['do_summary']

        if model_id in LTR_X_MODEL_Debiasing:
            cv_tape = CVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali)
        else:
            cv_tape = XCVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=do_vali)

        for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
            ranker.init()  # initialize or reset with the same random initialization

            if model_id in LTR_X_MODEL_Debiasing:
                neg_metric = False
                burn_in_train_data, train_data, test_data, vali_data = \
                    self.load_data_debiasing(data_dict=data_dict, eval_dict=eval_dict, fold_k=fold_k)
            elif model_id in LTR_X_MODEL_Fairness:
                neg_metric = True
                train_data, test_data, vali_data = \
                    self.load_data(data_dict=data_dict, eval_dict=eval_dict, fold_k=fold_k)

            if do_vali:
                if model_id in LTR_X_MODEL_Debiasing:
                    vali_tape = ValidationTape(fold_k=fold_k, num_epochs=epochs, validation_metric=vali_metric,
                                                validation_at_k=vali_k, dir_run=self.dir_run)
                else:
                    vali_tape = XValidationTape(fold_k=fold_k, num_epochs=epochs, validation_metric=vali_metric,
                                            validation_at_k=vali_k, dir_run=self.dir_run, neg_metric=neg_metric)
            if do_summary:
                summary_tape = SummaryTape(do_validation=do_vali, cutoffs=cutoffs, label_type=label_type,
                                           train_presort=train_presort, test_presort=test_presort, gpu=self.gpu)

            for epoch_k in range(1, epochs + 1):
                if model_id in LTR_X_MODEL_Debiasing:
                    ranker.x_train(burn_in_train_data=burn_in_train_data, train_data=train_data, epoch_k=epoch_k,
                                   initial_ranker=burn_in_ranker)
                elif model_id in LTR_X_MODEL_Fairness:
                    ranker.x_train(train_data=train_data, epoch_k=epoch_k, presort=train_presort)

                ranker.scheduler_step()  # adaptive learning rate with step_size=40, gamma=0.5

                if (do_summary or do_vali) and (epoch_k % log_step == 0 or epoch_k == 1):  # stepwise check
                    if do_vali:  # per-step validation score
                        if model_id in LTR_X_MODEL_Debiasing:
                            torch_vali_metric_value = ranker.validation(vali_data=vali_data, k=vali_k, device='cpu',
                                                                        vali_metric=vali_metric, label_type=label_type,
                                                                        max_label=max_label, presort=validation_presort)
                        elif model_id in LTR_X_MODEL_Fairness:
                            torch_vali_metric_value = \
                                ranker.validation(vali_data=vali_data, vali_metric=vali_metric, k=vali_k)

                        vali_metric_value = torch_vali_metric_value.squeeze(-1).data.numpy()
                        vali_tape.epoch_validation(ranker=ranker, epoch_k=epoch_k, metric_value=vali_metric_value)

                    if do_summary:  # summarize per-step performance w.r.t. train, test
                        summary_tape.epoch_summary(ranker=ranker, train_data=train_data, test_data=test_data,
                                                   vali_metric_value=vali_metric_value if do_vali else None)
                elif loss_guided:  # stopping check via epoch-loss
                    raise NotImplementedError

            if do_summary:  # track
                summary_tape.fold_summary(fold_k=fold_k, dir_run=self.dir_run, train_data_length=train_data.__len__())

            if do_vali:  # using the fold-wise optimal model for later testing based on validation data
                ranker.load(vali_tape.get_optimal_path(), device=self.device)
                vali_tape.clear_fold_buffer(fold_k=fold_k)
            else:  # buffer the model after a fixed number of training-epoches if no validation is deployed
                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/',
                            name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')

            if model_id in LTR_X_MODEL_Debiasing:
                cv_tape.fold_evaluation(model_id=model_id, ranker=ranker, test_data=test_data,
                                        max_label=max_label, fold_k=fold_k)
            else:
                cv_tape.fold_evaluation(model_id=model_id, ranker=ranker, test_data=test_data, fold_k=fold_k)

        ndcg_cv_avg_scores = cv_tape.get_cv_performance()
        return ndcg_cv_avg_scores

    def grid_run(self, debug=True, model_id=None, sf_id=None, data_id=None, dir_data=None, dir_output=None,
                 dir_json=None):
        """
        Perform diversified ranking based on grid search of optimal parameter setting
        """
        if dir_json is not None:
            x_data_eval_sf_json = dir_json + 'X_Data_Eval_ScoringFunction.json'
            para_json = dir_json + model_id + "Parameter.json"
            self.set_eval_setting(debug=debug, x_eval_json=x_data_eval_sf_json)
            self.set_data_setting(x_data_json=x_data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=x_data_eval_sf_json)
            self.set_model_setting(model_id=model_id, para_json=para_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)

            self.set_scoring_function_setting(debug=debug, sf_id=sf_id)

            self.set_model_setting(debug=debug, model_id=model_id)

        ''' select the best setting through grid search '''
        vali_k, cutoffs = 50, [10, 50, 100]  # cutoffs should be consistent w.r.t. eval_dict
        max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
        k_index = cutoffs.index(vali_k)
        max_common_para_dict, max_sf_para_dict, max_div_para_dict = None, None, None

        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                for sf_para_dict in self.iterate_scoring_function_setting():
                    for x_para_dict in self.iterate_model_setting():
                        curr_cv_avg_scores = \
                            self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                                               model_para_dict=x_para_dict)
                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                            max_cv_avg_scores, max_sf_para_dict, max_eval_dict, max_div_para_dict = \
                                curr_cv_avg_scores, sf_para_dict, eval_dict, x_para_dict

        # log max setting
        self.log_max(data_dict=data_dict, eval_dict=max_eval_dict,
                     max_cv_avg_scores=max_cv_avg_scores, sf_para_dict=max_sf_para_dict,
                     log_para_str=self.model_parameter.to_para_string(log=True, given_para_dict=max_div_para_dict))

    def point_run(self, debug=False, model_id=None, sf_id=None, data_id=None, dir_data=None, dir_output=None):
        """

        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """

        self.set_eval_setting(debug=debug, dir_output=dir_output)
        self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()

        self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
        sf_para_dict = self.get_default_scoring_function_setting()

        self.set_model_setting(debug=debug, model_id=model_id)
        model_para_dict = self.get_default_model_setting()

        self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                           model_para_dict=model_para_dict)

    def run(self, debug=False, model_id=None, sf_id=None, config_with_json=None, dir_json=None,
            data_id=None, dir_data=None, dir_output=None, grid_search=False):
        if config_with_json:
            assert dir_json is not None
            self.grid_run(debug=debug, model_id=model_id, dir_json=dir_json)
        else:
            assert sf_id in ['pointsf', 'listsf']
            if grid_search:
                self.grid_run(debug=debug, model_id=model_id, sf_id=sf_id,
                              data_id=data_id, dir_data=dir_data, dir_output=dir_output)
            else:
                self.point_run(debug=debug, model_id=model_id, sf_id=sf_id,
                               data_id=data_id, dir_data=dir_data, dir_output=dir_output)