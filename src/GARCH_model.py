import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import math
import random
from enum import Enum
from sklearn.preprocessing import MinMaxScaler
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from dataclasses import dataclass
import logging
import multiprocessing
from functools import partial

# RPY packages to run rugarch in python
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# Set values to R libs
base = importr('base')
rugarch = importr("rugarch", lib_loc=yaml.safe_load(open("config.yaml", 'r') )['env']['r_libs_path'])
TIME_HORIZON = 1  # this is fixed. other values would require minor development
MODEL_DICT_NAMES = 'fitted_'

# Logger
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class Switch(Enum):
    NONE = -1
    GRADUAL = 0
    ABRUPT = 1


@dataclass
class Model:
    # Model attributes and data
    id: int
    raw_input_path: float
    input_ts: pd.Series()
    probability: float

    # Define fitted params
    p = 0
    o = 0
    q = 0
    ARMAGARCH = None


def tsplot(y: pd.Series(), lags: list() = None, figsize: tuple = (15, 10), style: str = 'bmh'):
    """
    Function that plot correlation accross time steps / lags of the time series to allow exploration.
    :param y: time series fed
    :param lags: list of lags
    :param figsize: size
    :param style: plot style
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()


def plot_input(df: pd.DataFrame(), title: str):
    """ This plots the DF passed """
    df.plot(title=title)
    plt.show()


def plot_results(simulations: pd.Series()):
    """
    This function plots the series generated.
    :param simulations: TS generated
    """
    # Plot simulations
    lines = plt.plot(simulations.values[-1, :, :].T, color='blue', alpha=0.01)
    lines[0].set_label('Simulated paths')
    plt.show()

    print(np.percentile(simulations.values[-1, :, -1].T, 5))

    # Plot histogram of simulation
    plt.hist(simulations.values[-1, :, -1], bins=50)
    plt.title('Distribution of prices')
    plt.show()


def parse_yaml():
    """ This function parses the config file and returns options, paths, etc."""
    # Read YAML file
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        data_config = config['datasets']
        global_params = config['params']
        model_params = config['datasets']['files']
        paths = config['paths']
        plot = config['plot']
        print(config)
    return data_config, global_params, model_params, paths, plot


def single_thread(config, plot, file_config):
    """
    This handles each thread in 'load_all_series'
    :param config: from YAML file
    :param plot: plot series?
    :param file_config: list of ids, files and probabilities.
    :return: model and desc tuple
    """
    counter, file, prob = file_config
    print(f'counter: {counter}')
    df = pd.read_csv(os.path.join(config['path'], file), header=None)
    df.columns = config['cols']
    df.set_index(keys=config['index_col'], drop=True, inplace=True)

    # Clean nulls and select series
    raw_series = df[[config['sim_col']]]  # .dropna()
    raw_returns_series = 100 * df[[config['sim_col']]].pct_change()  # .dropna()

    # the returns later are calculated in a different way and they use log scale
    if plot:  # Plot initial df and returns
        plot_input(df, 'Raw dataset')
        plot_input(raw_series, 'Prices')
        plot_input(raw_returns_series, 'Returns')

    # Prepare Model and return it to be added to a dictionary
    return Model(id=counter, raw_input_path=os.path.join(config['path'], file),
                 input_ts=prepare_raw_series(config['parsing_mode'], raw_series), probability=prob), \
        f'{MODEL_DICT_NAMES}{counter}'


def load_all_series(config: dict(), plot: bool = True):
    """
    This function loads the intial series to feed them to models.
    :param config: from YAML file
    :param plot: plot series?
    :return: dict of series
    """
    # Load raw time series for the pre-training of the models
    logging.info('Load models...')
    pool = multiprocessing.Pool(processes=len(config['files']))
    mapped = pool.map(partial(single_thread, config, plot), config['files'])
    series_dict = dict(map(reversed, tuple(mapped)))
    return series_dict


def prepare_raw_series(mode: str, ts: pd.DataFrame()):
    """
    This function computes returns from the price series at logarithmic scale.
    Raw prices, if selected, are standarized using maxmin.
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(ts)
    print(f'Min: {min_max_scaler.data_min_[0]}, Max: {min_max_scaler.data_max_[0]}')

    if mode == 'returns':  # log returns
        ts = pd.DataFrame(np.log(ts / ts.shift(1))).reset_index(drop=True)
    else:  # standardization the dataset
        ts = pd.DataFrame(min_max_scaler.transform(ts)).reset_index(drop=True)

    return ts[ts.columns[0]].apply(lambda x: 0 if math.isnan(x) else x)  # the first value = NaN due to the returns


def get_best_parameters(ts: list(), config: dict()):
    """
    This list returns the best ARMA model for the current pre-training period.
    @:param TS: time series of returns used for pre-training
    """
    best_aic = np.inf
    best_order = None
    best_mdl = None

    for i in range(config['pq_rng']):   # [0,1,2,3,4]
        for d in range(config['d_rng']):  # [0] # we'll use arma-garch, so not d (= 0)
            for j in range(config['pq_rng']):
                try:
                    tmp_mdl = smt.ARIMA(ts, order=(i, d, j)).fit(
                        method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except:
                    continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl


def fit(current_series, p_, q_):
    """
    This function uses rugarch to fit an ARMA-GARCH model for p, 0, q (d!=0 only in ARIMA-GARCH)
    :param current_series:
    :param p_: p lags of the best fitted ARMA model.
    :param q_: q lags of the best fitted ARMA model.
    :return: trained/fitted model
    """
    # Initialize R GARCH model
    # TODO: test that calling the same rugarch does not make models to be static and reset when creating another...
    garch_spec = rugarch.ugarchspec(
        mean_model=robjects.r(f'list(armaOrder=c({p_},{q_}), include.mean=T)'),
        # Using student T distribution usually provides better fit
        variance_model=robjects.r('list(garchOrder=c(1,1))'),
        distribution_model='sged')  # 'std'

    # Train R GARCH model on returns as %
    numpy2ri.activate()  # Used to convert training set to R list for model input
    model = rugarch.ugarchfit(
        spec=garch_spec,
        data=np.array(current_series),
        out_sample=1  # : remember: the 1st forecast should be reproducible in the reconst., as the real value exists
    )
    numpy2ri.deactivate()
    #TODO IMP:
    # assert model.beta + model.alpha <= 1
    return model


def pre_train_models(dict_of_series: dict(), config: dict(), plot: bool = False):
    """
    This function triggers the selection of the best parameters and fitting of n models
     (one model per dataset added to the YAML config file).
    :param dict_of_series - datasets as a single column DF to fit the models
    :param config - dictionary from YAML with datasets-related configuration
    :param plot - plot model?
    :return list of fitted models
    """
    # Fit models in parallel
    logging.info('Fitting models...')
    pool = multiprocessing.Pool(processes=len(config['files']))
    mapped = pool.map(partial(pretraining_single_thread, plot, config), dict_of_series.items())
    return dict(map(reversed, tuple(mapped)))


def pretraining_single_thread(plot: bool, config: dict(), series_model: Model):
    """
    This handles each thread in 'load_all_series'
    :param config:
    :param plot: plot series?
    :param series_model: list of series as an object of Model.
    :return: fitted model and description to be added to dictionary
    """
    name_series, current_model = series_model
    logging.info(f'\n\nStart fitting process for {name_series}')
    # TODO: uncomment line below
    # _, ARMA_order, ARMA_model = get_best_parameters(ts=list(current_model.input_ts), config=config)  #
    ARMA_order, ARMA_model = (4, 0, 4), None
    # TODO: maybe this should get the best ARMAGARCH instead.
    print(current_model.id)
    print('Best parameters are: ')
    current_model.p, current_model.o, current_model.q = ARMA_order[0], ARMA_order[1], ARMA_order[2]
    print(f'{current_model.p}, {current_model.o}, {current_model.q}')

    if plot:
        tsplot(ARMA_order.resid, lags=30)
        tsplot(ARMA_order.resid ** 2, lags=30)

    # Now we can fit the arch model using the best fit ARIMA model parameters
    current_model.ARMAGARCH = fit(current_model.input_ts, current_model.p, current_model.q)  # 'o' not in ARMAGARCH
    return current_model, name_series  # f'{MODEL_DICT_NAMES}{counter}'


def get_forecast(model, ts):
    """
    This function calls the R rugarch library to produce a the ARMA-GARCH forecast.
    :param model - current selected model
    :param ts - current series
    :return forecast or the next time horizon
    """
    # print(ts)  # TODO: see why this is always the same
    garch_forecast = rugarch.ugarchforecast(model, data=ts, n_ahead=1, n_roll=1)
    return np.array(garch_forecast.slots['forecast'].rx2('seriesFor')).flatten()[0]


def reconstruct(new_model_forecast: float, init_value: float):
    """
    This function reconstructs the initial series, as the models are trained with returns/deltas in log scale.
    :param new_model_forecast - forecast as a return and in log scale
    :param init_value - initial value of the current series for the reconstruction
    :return: forecast reconstructed.
    """
    # TODO: reconstruction of price? talk to David.
    # TODO: Test this properly.
    #  It may be the case that we need to add as first row the last price of the series that
    #   belongs to the current series (or the model being used) so we do the cumsum.
    #   Here it does not apply as there is only one row, but I'm not sure if it's implemented correctly.
    # reconstructing forecasting (rec[rec_col_name].values[-TIME_HORIZON])
    return init_value * np.exp(new_model_forecast[TIME_HORIZON - 1] * -1)


def update_weights(w, switch_sharpness):
    """
    This function updates weights each iteration depending on the sharpness of the current switch.
    :param w: tuple of weights
    :param switch_sharpness: speed of changes
    :return: tuple of weights updated.
    """
    #TODO: use sigmoid?
    if switch_sharpness < 1:
        print('Minimum switch abrupcy is 0.1, so this is the value being used. ')
        switch_sharpness = 1
    incr = 0.1 * switch_sharpness

    # see for reference get_weight and reset_weights.
    w = (w[0] - incr, w[1] + incr)
    w = (0 if w[0] <= 0 else w[0], 1 if w[1] >= 1 else w[1])  # deal with numbers out of range
    return w


def reset_weights():
    """
    This function init weights (or different, depending of gradual or abrupt drifts)
    :return default/initial weight.
    """
    w = (1, 0)  # Initialize
    return w


def starts_switch(switch_prob: float, abrupt_prob: float):
    """
    This function flips coins and return if a drift should be triggered and its type.
    :param switch_prob
    :param abrupt_prob
    :return switch event that takes in place. integer represented by an enum.
    """
    if random.random() < switch_prob:
        # Switch ?
        if random.random() < abrupt_prob:
            return Switch.ABRUPT
        else:
            return Switch.GRADUAL
    else:
        # Otherwise
        return Switch.NONE


def get_new_model(current_id: int, config: dict()):
    """
    This function picks a new model based in their probability to be selected (equal for all by now)
    :param current_id - so there is an actual drift and the id is not repeated.
    :param config - for probabilities
    :return new model id
    """
    # NOT TO BE DEVELOPED (YET)
    # Just here in the case of having different probabilities of transitioning per series.
    # for i in range(len(config)):
    #     config[i][1]  # TOD: ENUMERATOR SO TRANSITION_PROBABILITIES_POS == 1
    new_model_id = random.randrange(1, len(config)+1)
    return get_new_model(current_id, config) if current_id == new_model_id else new_model_id


def switching_process(tool_params: dict(), models: dict(), data_config: dict(), plot: bool):
    """
    This function computes transitions between time series and returns the resulting time series.
    :param tool_params: info regarding to stitches from yaml file
    :param models: fitted models
    :param data_config: datasets info from yaml file
    :param plot: plot resulting ts?
    :return: ts - series generated
    :rerurn: rc - dataframe of events (switches flagged, models used and weights)
    """
    # Init params
    switch_shp = (tool_params['gradual_drift_sharpness'], tool_params['abrupt_drift_sharpness'])
    switch_type = Switch.NONE

    # Start with model A as initial model
    # current_id = 1
    current_model = models[f'{MODEL_DICT_NAMES}{1}']  # first model -> current_model = A (randomly chosen)
    new_model = None

    # Initialize main series
    ts = list()
    # rec_ts = pd.Series() TODO
    rc = list()
    counter = 0
    w = reset_weights()  # tuple (current, new) of model weights.
    # rec_value = TODO
    logging.info('Start of the context-switching generative process:')
    for counter in range(tool_params['periods']):
        # 1 Start forecasting in 1 step horizons using the current model
        old_model_forecast = get_forecast(current_model.ARMAGARCH, list(current_model.input_ts)
                                          if counter < current_model.p or counter < current_model.q else list(ts))
        new_switch_type = starts_switch(tool_params['switching_probability'], tool_params['abrupt_drift_prob'])

        # 2 In case of switch, select a new model and reset weights: (1.0, 0.0) at the start (no changes) by default.
        if new_switch_type.value >= 0:
            logging.info(f'There is a {new_switch_type.name} switch.')
            switch_type = new_switch_type
            new_model = models[f'{MODEL_DICT_NAMES}{get_new_model(current_model.id, data_config["files"])}']
            print(w)
            w = update_weights(w=reset_weights(), switch_sharpness=switch_shp[switch_type.value])
            print(w)
            # TODO: see why w does not change. does it reset or the issue is when updating?

        # 3 Log switches and events
        rc.append({'n_row': counter, 'new_switch': new_switch_type.name,
                   'cur_switch': switch_type.name, 'weights': w,
                   'current_model_id': current_model.id,   # Add p,o,q to this?
                   'new_model_id': -1 if new_model is None else new_model.id})
        assert w[0] + w[1] == 1

        # 4 if it's switching (started now or in other iteration), then forecast with new model and get weighted average
        if 0 < w[1] < 1:
            print('Update weights:')
            # Forecast and expand current series (current model is the old one, this becomes current when weight == 1)
            new_model_forecast = get_forecast(new_model.ARMAGARCH, list(ts))
            ts.append(old_model_forecast * w[0] + new_model_forecast * w[1])
            # TODO: TEST reconstruction once the rest works
            # rec_ts.append(reconstruct(new_model_forecast, rec_value) * w[1] +
            #              reconstruct(old_model_forecast, rec_value) * w[0])
            w = update_weights(w, switch_shp[switch_type.value])

            if w[1] == 1:
                current_model = new_model
                new_model = None
                w = reset_weights()
                switch_type = Switch.NONE

        # 3. Otherwise, use the current forecast
        else:
            # ts.append(reconstruct(old_model_forecast, rec_value)) # TODO
            ts.append(old_model_forecast)

        logging.info(f'Period {counter}: {ts[-1]}')

    # 4 Plot simulations
    if plot:
        plot_results(ts)
    # non-reconstructed ts is not returned  -TODO
    return pd.Series(ts),  pd.DataFrame(rc)  # TODO return pd.Series(rec_ts), pd.DataFrame(rc)


def add_noise(noise_level: float, ts: list()):
    """
    This function adds noise to the time series passed as a parameter.
    :param noise_level: percentage representing level of noise to be added
    :param ts: time series generated
    :return time series with added noise
    """
    logging.info('Adding noise...')

    t = np.linspace(-20, 20, len(ts))
    snr = 10 * np.log(1 + noise_level)  # SNR = 0.487 for noise_level = 0.05

    # Random N - length vector of Gaussian numbers
    r = np.random.randn(1, len(t))

    # 6.1 Add noise using (noise_level)% Gaussian Noise Method
    ts_n1 = pd.Series((ts + noise_level * r * ts)[0])  # Noisy signal

    # 6.2 Add noise using (10 * np.log(1 + noise_level)) SNR and White Gaussian Noise
    ts_n2 = pd.Series((ts + np.sqrt((10 ** (-snr / 10))) * r)[0])  # Noisy signal

    return ts_n1, ts_n2


def compute():
    """
    This function coordinates the whole process.
    1. It loads examples series and pre-train ad many models as series received.
    2. It triggers the switching and generation process and plots the resulting series.
    3. It adds white and gaussian noise.
    4. It exports the resulting time series without and with noise, and the events/switches to a CSV.
    """
    # 0 Read  from YAML file
    data_config, global_params, model_params, paths, plot = parse_yaml()

    # 1 Get dict of series and their probabilities. These are loaded from the yaml file.
    models_dict = load_all_series(config=data_config, plot=plot)  # these series is the series of returns on log scale

    # 2 Then, pre-train GARCH models by looking at different series
    models_dict = pre_train_models(dict_of_series=models_dict, config=data_config, plot=plot)

    # 3 Once the models are pre-train, these are used for simulating the final series.
    # At every switch, the model that generates the final time series will be different.
    ts, rc = switching_process(tool_params=global_params, models=models_dict, data_config=data_config, plot=plot)
    # TODO: reconstruction at this level? see notes from David

    # 4 Plot simulations
    if plot:
        plot_results(ts)

    # 5 Add noise (gaussian noise and SNR)
    noise_level = global_params['white_noise_level']
    ts_gn, ts_snr = add_noise(noise_level, list(ts))

    # 6 Final simulation (TS created) and a log of the regime changes (RC) to CSV files
    rc['ts'] = ts
    rc['ts_n1'] = ts_gn  # Gaussian noise
    rc['ts_n2'] = ts_snr  # SNR and White Gaussian Noise
    rc.to_csv(os.sep.join([paths['output_path'], paths['ts_export_name']]))  # TODO: add config desc to name?


if __name__ == '__main__':
    compute()
