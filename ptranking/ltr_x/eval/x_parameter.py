
import os
import json
import datetime
import numpy as np

from ptranking.metric.metric_utils import sort_nicely
from ptranking.metric.metric_utils import metric_results_to_string
from ptranking.ltr_adhoc.eval.parameter import EvalSetting, DataSetting, ScoringFunctionParameter

class XScoringFunctionParameter(ScoringFunctionParameter):
    """  """
    def __init__(self, debug=False, sf_id=None, sf_json=None):
        super(XScoringFunctionParameter, self).__init__(debug=debug, sf_id=sf_id, sf_json=sf_json)

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["XSFParameter"]
        return json_dict

    def default_pointsf_para_dict(self):
        """
        A default setting of the hyper-parameters of the stump neural scoring function.
        """
        # common default settings for a scoring function based on feed-forward neural networks

        self.sf_para_dict = dict()

        if self.use_json:
            opt = self.json_dict['opt'][0]
            lr = self.json_dict['lr'][0]
            pointsf_json_dict = self.json_dict[self.sf_id]
            num_layers = pointsf_json_dict['layers'][0]
            af = pointsf_json_dict['AF'][0]
            tl_af = pointsf_json_dict['TL_AF'][0]
            apply_tl_af = pointsf_json_dict['apply_tl_af'][0]
            BN = pointsf_json_dict['BN'][0]
            bn_type = pointsf_json_dict['bn_type'][0]
            bn_affine = pointsf_json_dict['bn_affine'][0]

            self.sf_para_dict['opt'] = opt
            self.sf_para_dict['lr'] = lr
            pointsf_para_dict = dict(num_layers=num_layers, AF=af, TL_AF=tl_af, apply_tl_af=apply_tl_af,
                                     BN=BN, bn_type=bn_type, bn_affine=bn_affine)
            self.sf_para_dict['sf_id'] = self.sf_id
            self.sf_para_dict[self.sf_id] = pointsf_para_dict
        else:
            self.sf_para_dict['opt'] = 'Adam'  # Adam | RMS | Adagrad
            self.sf_para_dict['lr'] = 0.001  # learning rate

            pointsf_para_dict = dict(num_layers=5, AF='GE', TL_AF='GE', apply_tl_af=True,
                                     BN=True, bn_type='BN', bn_affine=True)
            self.sf_para_dict['sf_id'] = self.sf_id
            self.sf_para_dict[self.sf_id] = pointsf_para_dict

        return self.sf_para_dict


class XEvalSetting(EvalSetting):
    """
    Class object for evaluation settings w.r.t. diversified ranking.
    """
    def __init__(self, debug=False, dir_output=None, x_eval_json=None):
        super(XEvalSetting, self).__init__(debug=debug, dir_output=dir_output, eval_json=x_eval_json)

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["XEvalSetting"]
        return json_dict

    def default_setting(self):
        """
        A default setting for evaluation
        :param debug:
        :param data_id:
        :param dir_output:
        :return:
        """
        if self.use_json:
            dir_output = self.json_dict['dir_output']
            epochs = self.json_dict['epochs'] # debug is added for a quick check
            do_validation = self.json_dict['do_validation']
            vali_k = self.json_dict['vali_k'] if do_validation else None
            vali_metric = self.json_dict['vali_metric'] if do_validation else None

            cutoffs = self.json_dict['cutoffs']
            do_log, log_step = self.json_dict['do_log'], self.json_dict['log_step']
            do_summary = self.json_dict['do_summary']
            loss_guided = self.json_dict['loss_guided']
            mask_label = self.json_dict['mask']['mask_label']
            mask_type = self.json_dict['mask']['mask_type']
            mask_ratio = self.json_dict['mask']['mask_ratio']

            self.eval_dict = dict(debug=False, grid_search=False, dir_output=dir_output,
                                  cutoffs=cutoffs, do_validation=do_validation, vali_k=vali_k, vali_metric=vali_metric,
                                  do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=loss_guided,
                                  epochs=epochs, mask_label=mask_label, mask_type=mask_type, mask_ratio=mask_ratio)
        else:
            do_log = False if self.debug else True
            do_validation, do_summary = True, False  # checking loss variation
            log_step = 1
            epochs = 2 if self.debug else 100
            vali_k = 50 if do_validation else None
            vali_metric = 'nDCG' if do_validation else None

            ''' setting for exploring the impact of randomly removing some ground-truth labels '''
            mask_label = False
            mask_type = 'rand_mask_all'
            mask_ratio = 0.2

            # more evaluation settings that are rarely changed
            self.eval_dict = dict(debug=self.debug, grid_search=False, dir_output=self.dir_output,
                                  do_validation=do_validation, vali_k=vali_k, vali_metric=vali_metric,
                                  cutoffs=[10, 50, 100], epochs=epochs,
                                  do_summary=do_summary, do_log=do_log, log_step=log_step, loss_guided=False,
                                  mask_label=mask_label, mask_type=mask_type, mask_ratio=mask_ratio)

        return self.eval_dict


class XDataSetting(DataSetting):
    """
    Class object for data settings w.r.t. data loading and pre-process w.r.t. diversified ranking
    """
    def __init__(self, debug=False, data_id=None, dir_data=None, x_data_json=None):
        super(XDataSetting, self).__init__(debug=debug, data_id=data_id, dir_data=dir_data, data_json=x_data_json)

    def load_para_json(self, para_json):
        with open(para_json) as json_file:
            json_dict = json.load(json_file)["XDataSetting"]
        return json_dict

##########
# Tape-recorder objects for logging during the training, validation processes.
##########

class XValidationTape(object):
    """
    Using a specified metric to perform epoch-wise evaluation over the validation dataset.
    """
    def __init__(self, fold_k, num_epochs, validation_metric, validation_at_k, dir_run, neg_metric=False):
        self.dir_run = dir_run
        self.num_epochs = num_epochs

        self.optimal_metric_value = -1000000 if neg_metric else 0.0
        self.optimal_epoch_value = None
        self.validation_at_k = validation_at_k
        self.validation_metric = validation_metric
        self.fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])

    def epoch_validation(self, epoch_k, metric_value, ranker):
        if epoch_k > 1: # report and buffer currently optimal model
            if (metric_value > self.optimal_metric_value) \
                    or (epoch_k == self.num_epochs and metric_value == self.optimal_metric_value):
                # we need at least a reference, in case all zero
                print('\t', epoch_k, '- {}@{} - '.format(self.validation_metric, self.validation_at_k), metric_value)
                self.optimal_epoch_value = epoch_k
                self.optimal_metric_value = metric_value
                ranker.save(dir=self.dir_run + self.fold_optimal_checkpoint + '/',
                            name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')
            else:
                print('\t\t', epoch_k, '- {}@{} - '.format(self.validation_metric, self.validation_at_k), metric_value)

    def get_optimal_path(self):
        buffered_optimal_model = '_'.join(['net_params_epoch', str(self.optimal_epoch_value)]) + '.pkl'
        path = self.dir_run + self.fold_optimal_checkpoint + '/' + buffered_optimal_model
        return path

    def clear_fold_buffer(self, fold_k):
        subdir = '-'.join(['Fold', str(fold_k)])
        run_fold_k_dir = os.path.join(self.dir_run, subdir)
        fold_k_files = os.listdir(run_fold_k_dir)
        list_model_files = []
        if fold_k_files is not None and len(fold_k_files) > 1:
            for f in fold_k_files:
                if f.endswith('.pkl'):
                    list_model_files.append(f)

            if len(list_model_files) > 1:
                sort_nicely(list_model_files)
            for k in range(1, len(list_model_files)):
                tmp_model_file = list_model_files[k]
                os.remove(os.path.join(run_fold_k_dir, tmp_model_file))

class XCVTape(object):
    """
    Using multiple metrics to perform (1) fold-wise evaluation; (2) k-fold averaging
    """

    print("check")

    def __init__(self, model_id, fold_num, cutoffs, do_validation):
        self.cutoffs = cutoffs
        self.fold_num = fold_num
        self.model_id = model_id
        self.do_validation = do_validation
        self.ee_cv_avg_scores = np.zeros(len(cutoffs))
        self.time_begin = datetime.datetime.now() # timing

    def fold_evaluation(self, ranker, test_data, fold_k, model_id):
        avg_ee_at_ks = ranker.eval_at_ks(test_data=test_data, vali_metric='ExpectedExposure', ks=self.cutoffs,
                                         gpu=False, device='cpu')
        fold_ee_at_ks = avg_ee_at_ks.data.numpy()

        self.ee_cv_avg_scores = np.add(self.ee_cv_avg_scores, fold_ee_at_ks)

        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=fold_ee_at_ks,
                                                         list_cutoffs=self.cutoffs, metric='EE'))
        metric_string = '\n\t'.join(list_metric_strs)
        print("\n{} on Fold - {}\n\t{}".format(model_id, str(fold_k), metric_string))

    def get_cv_performance(self):
        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - self.time_begin)

        ee_cv_avg_scores = np.divide(self.ee_cv_avg_scores, self.fold_num)

        eval_prefix = str(self.fold_num) + '-fold cross validation scores:' if self.do_validation \
                      else str(self.fold_num) + '-fold average scores:'

        list_metric_strs = []
        list_metric_strs.append(metric_results_to_string(list_scores=ee_cv_avg_scores,
                                                         list_cutoffs=self.cutoffs, metric='EE'))
        metric_string = '\n'.join(list_metric_strs)
        print("\n{} {}\n{}".format(self.model_id, eval_prefix, metric_string))
        print('Elapsed time:\t', elapsed_time_str + "\n\n")
        return ee_cv_avg_scores