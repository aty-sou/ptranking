
import sys
import datetime
import numpy as np
import copy

import torch

from ptranking.base.ranker import LTRFRAME_TYPE
from ptranking.ltr_x.click_model.pbm import PBM
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator
from ptranking.ltr_adhoc.eval.parameter import CVTape
from ptranking.ltr_federation.eval.federated_parameter import FederationParameter
from ptranking.data.data_utils import LTRDataset, SPLIT_TYPE, LETORSampler, LETORPercentSampler
from ptranking.ltr_x.eval.x_parameter import XDataSetting, XEvalSetting, XScoringFunctionParameter

from ptranking.ltr_adhoc.pointwise.rank_mse import RankMSE

from ptranking.ltr_federation.base.federated_server import FederatedServer
from ptranking.ltr_federation.eval.federated_parameter import FederationSummaryTape
from ptranking.ltr_federation.federated_debiasing.federated_pdgd import Federated_PDGDParameter

from ptranking.ltr_federation.util.plot_fpdgd import draw_line

LTR_Federation_MODEL = ['Federated_PDGD']

class FederationLTREvaluator(LTREvaluator):
    def __init__(self, frame_id=LTRFRAME_TYPE.Federation, cuda=None):
        super(FederationLTREvaluator, self).__init__(frame_id=frame_id, cuda=cuda)

    def setup_eval(self, data_dict, eval_dict, sf_para_dict, model_para_dict):
        """
        Finalize the evaluation setting correspondingly
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """

        # print("data_dict:{}これを探しています".format(data_dict))

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

    def load_federated_server(self, sf_para_dict, model_para_dict, federation_para_dict, click_model=None):
        # todo: click_modelをここで追加する
        federated_server = FederatedServer(sf_para_dict=sf_para_dict, model_para_dict=model_para_dict,
                                           federation_para_dict=federation_para_dict, click_model=click_model)
        return federated_server

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

    def set_federation_setting(self, debug=False, federation_id=None, para_json=None):
        if para_json is not None:
            self.federation_parameter = FederationParameter(para_json=para_json)
        else:
            self.federation_parameter = FederationParameter(debug=debug, federation_id=federation_id)

    def get_default_federation_setting(self):
        return self.federation_parameter.default_para_dict()

    def load_data_federation(self, eval_dict, data_dict, fold_k):
        """
        Load the dataset correspondingly.
        :param eval_dict:
        :param data_dict:
        :param fold_k:
        :param model_para_dict:
        :return:
        """

        # 現在のfold
        #print("fold_k:{}".format(fold_k))

        file_train, file_vali, file_test = self.determine_files(data_dict, fold_k=fold_k)

        input_eval_dict = eval_dict if eval_dict['mask_label'] else None # required when enabling masking data

        train_data = LTRDataset(file=file_train, split_type=SPLIT_TYPE.Train, presort=data_dict['train_presort'],
                                 data_dict=data_dict, eval_dict=input_eval_dict)

        #test_data = LTRDataset(file=file_test, split_type=SPLIT_TYPE.Test, data_dict=data_dict, presort=data_dict['test_presort'])
        _test_data = LTRDataset(file=file_test, split_type=SPLIT_TYPE.Test, data_dict=data_dict, presort=data_dict['test_presort'])
        test_letor_sampler = LETORSampler(data_source=_test_data, rough_batch_size=data_dict['test_rough_batch_size'])
        test_data = torch.utils.data.DataLoader(_test_data, batch_sampler=test_letor_sampler, num_workers=0)

        if eval_dict['do_validation'] or eval_dict['do_summary']: # vali_data is required
            vali_data = LTRDataset(file=file_vali, split_type=SPLIT_TYPE.Validation, data_dict=data_dict, presort=data_dict['validation_presort'])
        else:
            vali_data = None

        return train_data, test_data, vali_data

    def kfold_cv_eval(self, data_dict=None, eval_dict=None, sf_para_dict=None, model_para_dict=None, federation_para_dict=None):

        self.display_information(data_dict, model_para_dict)
        self.check_consistency(data_dict, eval_dict, sf_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, model_para_dict)

        model_id = model_para_dict['model_id']
        fold_num, label_type, max_label = data_dict['fold_num'], data_dict['label_type'], data_dict['max_rele_level']

        #if model_id in LTR_Federation_MODEL: # federated LTR methods
        #    click_model = PBM(max_label=4, gpu=self.gpu, device=self.device, user_model="PERFECT")
        #    federated_server = self.load_federated_server(sf_para_dict=sf_para_dict, model_para_dict=model_para_dict,
        #                                                  federation_para_dict=federation_para_dict,
        #                                                  click_model=click_model)
        #else:
        #    raise NotImplementedError

        # for quick access of common evaluation settings
        vali_k, log_step, cutoffs = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        do_vali, vali_metric, do_summary = eval_dict['do_validation'], eval_dict['vali_metric'], eval_dict['do_summary']

        interaction_budget, num_clients, interactions_per_feedback =\
            federation_para_dict["interaction_budget"], federation_para_dict["num_clients"], federation_para_dict["interactions_per_feedback"]
        # fed_epochs ... Number of server side updates
        fed_epochs = interaction_budget // num_clients // interactions_per_feedback

        if model_id in LTR_Federation_MODEL:
            cv_tape = CVTape(model_id=model_id, fold_num=fold_num, cutoffs=cutoffs, do_validation=False)
        else:
            raise NotImplementedError

        #todo add tape objects for federated LTR

        #print("fold_num: {}".format(fold_num))

        if model_id in LTR_Federation_MODEL: # federated LTR methods
            # user model
            user_model_list=["PERFECT", "NAVIGATIONAL", "INFORMATIAL"]
            for user_model in user_model_list:
                click_model = PBM(max_label=4, gpu=self.gpu, device=self.device, user_model=user_model)
                federated_server = self.load_federated_server(sf_para_dict=sf_para_dict, model_para_dict=model_para_dict,
                                                              federation_para_dict=federation_para_dict,
                                                              click_model=click_model)

                # foldごとのリストをまとめるリスト
                es_list_size = 4
                all_fold_list = torch.zeros(fold_num, es_list_size, fed_epochs)
                '''
                fold_num: foldの数
                es_list_size: パラメータの数
                fed_epochs: サーバーのupdate回数
                '''

                for fold_k in range(1, fold_num + 1):  # evaluation over k-fold data
                    print("Fold-{} の作業".format(fold_k))
                    if model_id in LTR_Federation_MODEL:
                        train_data, test_data, _ = self.load_data_federation(data_dict=data_dict, eval_dict=eval_dict, fold_k=fold_k)
                        #print("length of train data: {}".format(len(train_data)))
                        #print("length of test data: {}".format(type(test_data)))
                    else:
                        raise NotImplementedError

                    if do_summary:
                        #print("do summary:{}".format(do_summary))
                        summary_tape = FederationSummaryTape(cutoffs=cutoffs, label_type=label_type,
                                                             test_presort=data_dict['test_presort'], gpu=self.gpu)

                    # epsilon & sensitivity のリスト
                    e_s_list = torch.tensor([[1.2, 3], [2.3, 3], [4.5, 5], [10, 5]])

                    # esごとのndcgをまとめて管理するリスト
                    ndcg_es_list = torch.zeros(4, fed_epochs)

                    # index も同時に取得する
                    for es_index, p in enumerate(e_s_list):
                        epsilon = p[0]
                        sensitivity = p[1]

                        # conduct necessary initialization for federated learning, e.g., the initial global ranker
                        federated_server.init(train_data, test_data, data_dict, epsilon, sensitivity)  # initialize or reset

                        # serverでのアップデート毎にndcgを記録するリスト
                        server_ndcg_list = torch.zeros(fed_epochs)

                        print("--total fed_epochs: {}".format(fed_epochs))
                        for epoch_k in range(fed_epochs):
                            if model_id in LTR_Federation_MODEL:
                                trend_avg_client_ndcg = copy.deepcopy(federated_server.federated_train())
                                # ndcgの値を格納する
                                server_ndcg_list[epoch_k] = trend_avg_client_ndcg
                                print("fed_epoch_{}_server:{}".format(epoch_k, trend_avg_client_ndcg))

                            else:
                                raise NotImplementedError

                            # todo implement more complex SGD
                            #ranker.scheduler_step()  # adaptive learning rate with step_size=40, gamma=0.5

                            if do_summary and (epoch_k % log_step == 0 or epoch_k == 1):  # stepwise check
                                summary_tape.epoch_summary(ranker=federated_server.global_ranker, test_data=test_data)

                        ndcg_es_list[es_index] = server_ndcg_list

                    # 確認
                    print("fold_{}でのupdateの様子:{}".format(fold_k, ndcg_es_list))

                    # fold_k での結果を格納
                    all_fold_list[fold_k-1] = ndcg_es_list

                    if do_summary:  # track
                        #todo how to average among k-folds
                        summary_tape.fold_summary(fold_k=fold_k, dir_run=self.dir_run, train_data_length=train_data.__len__())

                    # buffer the model after a fixed number of training-epoches if no validation is deployed
                    fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                    federated_server.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')

                    if model_id in LTR_Federation_MODEL:
                        #todo only pointwise computation?
                        cv_tape.fold_evaluation(model_id=model_id, ranker=federated_server.global_ranker, test_data=test_data, max_label=max_label, fold_k=fold_k)
                    else:
                        raise NotImplementedError

                print("User_Model:{}での平均".format(user_model))

                # todo: deepcopyする
                # 平均をとる
                all_fold_mean = torch.mean(all_fold_list, dim=0)
                print(all_fold_mean)

                # todo: 可視化する
                # user model毎に、foldの平均を plot
                draw_line(fed_epochs, ndcg_es_list, user_model)

        else:
            raise NotImplementedError


        ndcg_cv_avg_scores = cv_tape.get_cv_performance()
        return ndcg_cv_avg_scores

    def grid_run(self, debug=True, model_id=None, sf_id=None, data_id=None, dir_data=None, dir_output=None,
                 dir_json=None):
        """
        Perform diversified ranking based on grid search of optimal parameter setting
        """
        #print("grid_run")

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

    def point_run(self, debug=False, federation_id=None, model_id=None, sf_id=None, data_id=None, dir_data=None, dir_output=None):
        """
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """

        #print("point_run")

        self.set_eval_setting(debug=debug, dir_output=dir_output)
        self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()

        self.set_scoring_function_setting(debug=debug, sf_id=sf_id)
        sf_para_dict = self.get_default_scoring_function_setting()

        self.set_model_setting(debug=debug, model_id=model_id)
        model_para_dict = self.get_default_model_setting()

        # todo federation_parameter's implementation
        self.set_federation_setting(debug=debug, federation_id=federation_id)
        federation_para_dict = self.get_default_federation_setting()

        # dataに関するパラメータ
        #print(data_dict)

        self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, sf_para_dict=sf_para_dict,
                           model_para_dict=model_para_dict, federation_para_dict=federation_para_dict)

    def run(self, debug=False, federation_id=None, model_id=None, sf_id=None, config_with_json=None, dir_json=None,
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
                self.point_run(debug=debug, federation_id=federation_id, model_id=model_id, sf_id=sf_id,
                               data_id=data_id, dir_data=dir_data, dir_output=dir_output)
