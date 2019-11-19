
import yaml
import pandas as pd
import argparse
import os

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from skmultiflow.data.data_stream import DataStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.meta.batch_incremental import BatchIncremental
from skmultiflow.trees import RegressionHoeffdingTree, RegressionHAT, HAT, HATT, HoeffdingTree
import skmultiflow.utils.constants as constants

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import sys
# sys.path.append('../..')


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = [10, 5]

PATH = os.getcwd()  # TODO: from yaml file
OUTPATH = os.sep.join([f'{PATH}', 'output'])
PREDPATH = f'{PATH}/predictions/'
MODELSPATH = f'{PATH}/models/'

# TODO: move to yaml file
algorithms = {
    'experiment_1': {
        'ht': {'grace_period': 400, 'tie_threshold': 0.2, 'leaf_prediction': 'perceptron',
               'learning_ratio_perceptron': 0.02, 'learning_ratio_decay': 0.1},
        'hat': {'grace_period': 400, 'tie_threshold': 0.2, 'leaf_prediction': 'perceptron',
                'learning_ratio_perceptron': 0.02, 'learning_ratio_decay': 0.1},
        'sgd': {'loss': 'squared_epsilon_insensitive', 'penalty': 'l2', 'alpha': 0.01, 'n_iter': 1000},
        'mlp': {'hidden_layer_sizes': 7, 'solver': 'sgd', 'activation': 'relu', 'learning_rate_init': 0.01,
                'momentum': 0.4, 'learning_rate': 'adaptive'},
        'passive_aggresive': {'C': 0.1, 'loss': 'epsilon_insensitive', 'epsilon': 0.05}
    }
}


def export_dataset(X, y, dataset, site, dataset_file):
    X.to_csv(site + '_features.csv', index=True, header=True)
    y.to_csv(site + '_prediction.csv', index=True, header=True)
    dataset.to_csv(dataset_file, index=True, header=True)


def export_model_info(experiment_id, master_name, model, name):
    out = open(f'{OUTPATH}{experiment_id.split("_")[1]}/{experiment_id}_model_info.txt', "w")
    # print('\nResult for ' + name + ' ' + master_name + ': \n')
    if callable(getattr(model, "get_info", None)):
        out.writelines('\n\nModel info:')
        out.writelines(model.get_info())
        out.writelines('------------------\n')
        # print(model.get_info())
        # print('------------------\n')
    if callable(getattr(model, "get_model_description", None)):
        out.writelines('\n\nFinal model description:')
        out.writelines(model.get_model_description())
        out.writelines('------------------\n')
        # print(model.get_model_description())
        # print('------------------\n')
    if callable(getattr(model, "feature_importances_", None)):
        out.writelines('\n\nFinal feature importance:')
        out.writelines(model.feature_importances_)
        out.writelines('------------------\n')
        # print(model.feature_importances_)
        # print('------------------\n')
    out.close()


def load_config():
    """ This just loads the config file. """
    with open(os.path.join(PATH, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f.read())
        config['database']['server_user'] = os.environ['JOYCE3_USER']
        config['database']['private_key'] = os.environ['JOYCE3_AUTH_KEY']
    return config


def get_values_in_rows(sample, ws_days):
    ws = int(ws_days * 24 * 60 / sample)
    test_each_n = int(24 * 60 / sample)
    return test_each_n, ws


def set_evaluator(pretrain_size, test_each_n, test_name, classification_task):
    mode = 'incremental'
    # prequential evaluation = testing then training with each instance
    evaluator = EvaluatePrequential(show_plot=False,
                                    pretrain_size=pretrain_size,
                                    max_samples=sys.maxsize,
                                    batch_size=1 if mode == 'incremental' else test_each_n,
                                    n_wait=test_each_n,  # the number of samples to process between each test.
                                    restart_stream=True,
                                    data_points_for_classification=True,
                                    output_file=os.sep.join([OUTPATH,
                                                             f'{test_name}_tests_output.csv']),
                                    metrics=['accuracy',
                                             'kappa',
                                             'kappa_t',
                                             'kappa_m',
                                             'true_vs_predicted',
                                             'precision',
                                             'recall',
                                             'f1',
                                             'gmean', # binary-classification only]
                                             'running_time',
                                             'model_size'] if classification_task else
                                    ['mean_square_error',
                                     'mean_absolute_error',
                                     'true_vs_predicted',  # it may not represent the actual learner performance
                                     'running_time',
                                     'model_size'])
    return evaluator


def get_models(ws, classification_task: bool = True):

    # Create dictionary of base learners
    regressions_models = {
        # 1 Regression Hoeffding trees known as Fast incremental model tree with drift detection (FIMT-DD).
        # The trees below  uses ADWIN to detect drift and PERCEPTRON or MEAN to make predictions
        'ht': RegressionHoeffdingTree(**algorithms['experiment_1']['ht']),
        # 2 The main different between HT and AHT is that aht does more pruning overtime to maintain itself 'clean'
        'hat': RegressionHAT(**algorithms['experiment_1']['hat']),
        # 3. Stochastic Gradient Descent for regression from scikit-learn
        # 'sgd': SGDRegressor(**algorithms['experiment_1']['sgd']),
        # 4 Multi-layer Perceptron regressor from scikit-learn
        'mlp': MLPRegressor(**algorithms['experiment_1']['mlp']),
        # 5 Passive Aggressive Regressor from scikit-learn
        'passive_aggresive': PassiveAggressiveRegressor(**algorithms['experiment_1']['passive_aggresive']),
        # # X. Static scikit-learn models to be learned in a mini-batch fashion
        'dt': DecisionTreeRegressor(criterion='mae', splitter='best', max_depth=None,
                                    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                    max_features=None, random_state=None, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
    }

    classification_models = {
        'ht': HoeffdingTree(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                            split_criterion='info_gain', split_confidence=1e-07, tie_threshold=0.05,
                            binary_split=False, stop_mem_management=False, remove_poor_atts=False, no_preprune=False,
                            leaf_prediction='nba', nb_threshold=0, nominal_attributes=None),  # random_state/seed?
        'hat': HAT(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                   split_criterion='info_gain', split_confidence=1e-07, tie_threshold=0.05, binary_split=False,
                   stop_mem_management=False, remove_poor_atts=False, no_preprune=False,
                   leaf_prediction='nba', nb_threshold=0, nominal_attributes=None),
        'hatt': HATT(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                     min_samples_reevaluate=20, split_criterion='info_gain', split_confidence=1e-07,
                     tie_threshold=0.05, binary_split=False, stop_mem_management=False, leaf_prediction='nba',
                     nb_threshold=0, nominal_attributes=None),
        'dt': DecisionTreeClassifier()  # default-config
        # TODO: add more inc classifiers from sklearn and adaptive ones from skmultiflow
    }

    # Assign dictionary of models depending on prediction task
    models = classification_models if classification_task else regressions_models

    # Now add models that have dependencies with the base learners added already..

    # Mini batch looks like still work in progress in scikit-multiflow.
    # It looks like not working properly yet at least for regression (it's actually classifying)
    # models.update({'idt': BatchIncremental(base_estimator=models['dt'], window_size=ws, n_estimators=15)})

    # Dictionary of selected combinations and their names for evaluation
    selected_models = {
        'HOEFT': models['ht'],
        # 'HAT': models['hat'],
        # 'SGD': models['sgd'],
        # 'MLP': models['mlp'],
        # 'PassiveAggressive': models['passive_aggresive']
        # , 'Mini batch ensemble of DT': models['idt']
    }
    return selected_models


grid_search_models = {
    'regression': {
        # SGDRegressor: {'loss': ['squared_loss', 'squared_epsilon_insensitive', 'epsilon_insensitive'], # 'huber' tried
        #                'penalty': ['l1', 'l2', 'elasticnet'],
        #                'alpha': [0.00001, 0.0001, 0.001, 0.01],
        #                'n_iter': [50, 100, 500, 1000]},
        MLPRegressor: {'hidden_layer_sizes': [3, 7, 10, 50],
                       'solver': ['adam', 'sgd'],
                       'activation': ['relu', 'logistic', 'tanh'],
                       'learning_rate_init': [0.001, 0.01, 0.1],  # 0.2, 0.3 also tried
                       'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
                       'learning_rate': ['adaptive']},  # also tried constant, invscaling
        PassiveAggressiveRegressor: {'C': [0.1, 0.5, 1.0],
                                     'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                                     'epsilon': [0.05, 0.1, 0.2]},
        RegressionHoeffdingTree: {'grace_period': [100, 200, 300, 400],
                                  'tie_threshold': [0.01, 0.05, 0.1, 0.2],
                                  'leaf_prediction': ['mean', 'perceptron'],
                                  'learning_ratio_perceptron': [0.02],
                                  'learning_ratio_decay': [0.001]}
        # params for Hoeffding Tree are also used for RegressionHAT, as it's just another version of it more adaptive
    },
    'classification': {} # TODO
}


def run_modelling():

    """
    Load and prepare features for modeling
    :return:
    """

# ['rsi_10', 'mom_10', 'ema_10', 'sma_10', 'wma_10', 'trima_10', 'roc_10', 'rocr_10', 'ppo_10', 'label']  55%
    print(OUTPATH)
    # 1. Read a dataframe, create an stream and initialize (if it exists, read from directory, otherwise create it).
    datasets = dict()
     #'rsi_10', 'mom_10', 'ema_10', 'sma_10', 'wma_10', 'roc_10'
    files = {
        'AAPL': 'APPLE_[2018-08-01_to_2018-09-11]_5min_indicators.csv' ,
        'BTC': 'BITCOIN_[2019-07-01_to_2019-07-15]_5min_indicators.csv',
        'DIA': 'DOWJONES_[2015-08-01_to_2015-08-31]_market_hours_indicators.csv',
        'XRP': 'RIPPLE_[2019-07-01_to_2019-08-01]_5min_indicators.csv'
    }
    task = 'classification'  # 'classification' , 'regression'
    for file in files.values():
        file_path = os.sep.join(['C:', 'Users', 'suare', 'Workspace',
                                 'phd_cetrulin', 'moa-2017.06-sources', 'data', 'real',
                                 'Synthethic_TS_new', 'raw', file])
        full_df = pd.read_csv(file_path, sep=';')
        full_df.drop(['close', 'close_t-1', 'close_t-2', 'close_t-3', 'close_t-4'],
                     axis=1, inplace=True, errors='ignore')
        batch_size = 10
        # TODO: check why PPO was here and now by adding open,close,low,high not. weird........
        subsets = [
            (0, ['rsi_10','mom_10','ema_10','ema_20','ema_30','sma_10','sma_20','sma_30','wma_10','wma_20','wma_30','trima_10','trima_20','trima_30','roc_10','rocr_10','label']),
            (1, ['rsi_10','mom_10','ema_10','ema_20','sma_10','sma_20','wma_10','wma_20','trima_10','trima_20','roc_10','rocr_10','label']),
            (2, ['rsi_10','mom_10','ema_10','sma_10','wma_10','trima_10','roc_10','rocr_10','label']),
            (3, ['rsi_10','mom_10','ema_10','sma_10','wma_10','roc_10','label']),
            (4, ['rsi_10','mom_10','ema_10','sma_10','roc_10','label']),  #,
            (5, ['rsi_10','mom_10','ema_10','ema_20','ema_30','sma_10','sma_20','sma_30','wma_10','wma_20','wma_30','trima_10','trima_20','trima_30','roc_10','label']),
            (6, ['rsi_10','mom_10','ema_10','ema_20','ema_30','sma_10','sma_20','sma_30','trima_10','trima_20','trima_30','roc_10','label']),
            (7, ['rsi_10','mom_10','ema_10','ema_20','ema_30','sma_10','sma_20','sma_30','wma_10','wma_20','wma_30','roc_10','label']),
            (8, ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','sma_20','sma_30','wma_5','wma_10','wma_20','wma_30','ema_5','ema_10','ema_20','ema_30','trima_5','trima_10','trima_20','trima_30','adx_10','bbands_upperband','bbands_middleband','bbands_lowerband','roc_10','rocr_10','stochf_fastd','stochf_fastk','aroon_aroondown','aroon_aroonup','medprice_10','typprice_10','wclprice_10','atr_10','macdfix_macd','sar_10','label']),
            (9, ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','sma_20','sma_30','wma_5','wma_10','wma_20','wma_30','ema_5','ema_10','ema_20','ema_30','trima_5','trima_10','trima_20','trima_30','adx_10','bbands_upperband','bbands_middleband','bbands_lowerband','roc_10','aroon_aroondown','aroon_aroonup','medprice_10','typprice_10','wclprice_10','atr_10','macdfix_macd','sar_10','label']),
            (10, ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','wma_10','ema_10','trima_10','adx_10','bbands_upperband','bbands_middleband','bbands_lowerband','roc_10','rocr_10','stochf_fastd','stochf_fastk','aroon_aroondown','aroon_aroonup','medprice_10','typprice_10','wclprice_10','atr_10','macdfix_macd','sar_10','label']),
            (11, ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','wma_10','ema_10','trima_10','adx_10','bbands_upperband','bbands_lowerband','roc_10','aroon_aroondown','aroon_aroonup','medprice_10','typprice_10','wclprice_10','atr_10','sar_10','label']),
            (12, ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','wma_10','ema_10','trima_10','adx_10','bbands_upperband','bbands_lowerband','roc_10','aroon_aroondown','aroon_aroonup','label']),
            (13, ['rsi_10','willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','wma_10','roc_10','label']),
            (14, ['rsi_10','mom_10','sma_5','sma_10','sma_20','wma_5','wma_10','wma_20','trima_5','trima_10','trima_20','roc_10', 'macd_macd', 'label']),
            (15, ['rsi_10','willr_10','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','wma_5','wma_10','ema_5','ema_10','adx_10','roc_10','aroon_aroondown','aroon_aroonup','atr_10','macdfix_macd','label']),
            (16, ['willr_10','macd_macd','cci_10','mom_10','stoch_slowk','stoch_slowd','sma_5','sma_10','wma_5','wma_10','ema_5','ema_10','trima_10','adx_10','bbands_upperband','bbands_lowerband','roc_10','aroon_aroondown','aroon_aroonup','label']),
            (17, ['rsi_10','mom_10','sma_5','sma_10','sma_20','wma_5','wma_10','wma_20','ema_5','ema_10','ema_20','roc_10', 'label']),
            (18, ['rsi_10','willr_10','cci_10','mom_10','sma_5','sma_10','sma_20','ema_5','ema_10','ema_20','roc_10', 'label']),
        ]
        datasets.update({'all': full_df})  # just one dataset



        # best ripple: 16, 12
        # dia: 6
        # apple: 18, 6
        # btc: 10

        dataset_counter = 0
        # for dataset in datasets.values():
        #     dataset_counter = dataset_counter + 1
        test_name = file.split('_')[0] + '_' + str(file.split("_")[0])

        print('start')
        results = []
        for id, subset in subsets:
            df = full_df[subset]
            stream = DataStream(df)
            stream.prepare_for_use()
            print(f'\n\n\nRUNNING SUBSET #{id} - {file.split("_")[0]}\n==================\n')

            # 2. Instantiate and select the regression learners (vanilla setup configuration)
            selected_models = get_models(ws=batch_size,
                                         classification_task=(task == 'classification'))

            # 3. Setup the evaluator
            pretrain_size = 0  # int(len(dataset) * .8)  # 1000 10000 int(len(dataset) * .8)
            evaluator = set_evaluator(pretrain_size=pretrain_size, test_each_n=batch_size,
                                      test_name=test_name + f'_subset{id}',
                                      classification_task=(task == 'classification'))

            # 4. Run evaluation
            print(f'Running model.')
            print('Turn-off scientific mode in pycharm to see the results overtime and plots properly.')
            evaluator.evaluate(stream=stream, model=list(selected_models.values()),
                               model_names=list(selected_models.keys()),)  # results saved at the end
            print(f'Subset: {subset}')
            measurements = {'id': id,
                            'accuracy': evaluator._data_buffer.get_data(metric_id=constants.ACCURACY,
                                                                        data_id=constants.MEAN)[0]}
            results.append(measurements)

        print(f'\n\n\n=================\n\n\nResults for dataset: {file.split("_")[0]}')
        print(pd.DataFrame(results))

        print("Class distribution: ")
        label_zero = len(df[df['label'] == 0])
        label_one = len(df[df['label'] == 1])
        print("0 in "+str(float(label_zero)/(label_one+label_zero))+"%")
        print("1 in "+str(float(label_one)/(label_one+label_zero))+"%")

        print('\n\n\n=======================\n=======================\n\n\n')

        # 5. Show results and save models
        # for name, model in selected_models.items():
        #     experiment_id = name + '_' + test_name
            # export_model_info(experiment_id, master_name, model, name)
            # TODO: some models crash when dumping the result. fix this
            # dump(model, f'{MODELSPATH}{experiment_id.split("_")[1]}/{experiment_id}.joblib')  # TODO: add date


def run_all_combinations(export_flag, num_closest_farms: int, nearest_farm_mode: str, clustering: bool):
    from itertools import combinations
    print('\n\n -- RUNNING ALL COMBINATIONS AS THE ARGUMENT -a IS ENABLED.-- \n\n ')

    # Listing all combinations
    feature_list = [('base',)]
    feature_options = []
    for i in range(len(feature_options)):
        feature_list += list(combinations(feature_options, i + 1))
    sites = args.site.split(',')
    frequencies = [10, 30, 60]
    modes = ['mini-batch']  # 'incremental'

    # Iterating through them
    for s in sites:
        for f in frequencies:
            for m in modes:
                for fts in feature_list:
                    if len(fts) > 1:
                        fts_ = '-'.join(fts)
                    else:
                        fts_ = fts[0]
                    column_name = '_'.join([s, fts_, str(f)])
                    save_path = os.path.join(args.path, 'base' if len(fts) == 0 else fts_)

                    print(f"Running model for combination: {column_name}")
                    run_modelling(path=save_path, site=s, frequency=f,
                                  lags='lags' in fts,
                                  movingaverage='moving_average' in fts,
                                  momentum='momentum' in fts,
                                  mode=m, export=export_flag,
                                  grid_search=False, hourly_models=False,
                                  num_closest_farms=num_closest_farms, nearest_farm_mode=nearest_farm_mode,
                                  clustering=clustering)  # hourly models hardcoded as false here


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--grid_search', action='store_true',
                        help='Performing grid search only rather than training models and predicting.')
    args = parser.parse_args()
    print(args)

    run_modelling()


