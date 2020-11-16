import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy.stats as scs
import multiprocessing
from sklearn.preprocessing import MinMaxScaler

TIME_HORIZON = 1  # fixed / static TODO


def tsplot(y: pd.Series(), lags: list() = None, figsize: tuple = (15, 10), style: str = 'bmh'):
    """
    Function that plots correlation accross time steps / lags of the time series to allow exploration.
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


def prepare_raw_series(mode: str, ts: pd.DataFrame()):
    """
    This function computes returns from the price series at logarithmic scale.
    Raw prices, if selected, are standarized using maxmin.
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler = min_max_scaler.fit(ts)
    print(f'Min: {min_max_scaler.data_min_[0]}, Max: {min_max_scaler.data_max_[0]}')

    if mode == 'returns':  # log returns
        ts = pd.DataFrame(np.log(ts / ts.shift(1))).dropna().reset_index(drop=True)
    else:  # standardization the dataset
        ts = pd.DataFrame(min_max_scaler.transform(ts)).dropna().reset_index(drop=True)

    # Returning series of ret
    return ts[ts.columns[0]]  #.apply(lambda x: 0 if math.isnan(x) else x)  # the first value = NaN due to the returns


def sigmoid(x):
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def get_sigmoid():
    x = np.linspace(-5, 5, 100)  # for sigmoid
    return sigmoid(x)


def reconstruct(ts: float, init_val: float):
    """
    This function reconstructs the initial series, as the models are trained with returns/deltas in log scale.
    :param new_model_forecast - forecast as a return and in log scale
    :param new_model_forecast - forecast as a return and in log scale
    :param init_value - initial value of the current series for the reconstruction
    :return: forecast reconstructed.
    """
    # return init_value * np.exp(ts[TIME_HORIZON - 1] * -1)
    return init_val * np.exp(np.cumsum(ts))  # * -1))


def add_noise(noise_level: float, ts: list()):
    """
    This function adds noise to the time series passed as a parameter.
    :param noise_level: percentage representing level of noise to be added
    :param ts: time series generated
    :return time series with added noise
    """
    t = np.linspace(-20, 20, len(ts))
    snr = 10 * np.log(1 + noise_level)  # SNR = 0.487 for noise_level = 0.05

    # Random N - length vector of Gaussian numbers
    r = np.random.randn(1, len(t))

    # 6.1 Add noise using (noise_level)% Gaussian Noise Method
    ts_n1 = pd.Series((ts + noise_level * r * ts)[0])  # Noisy signal

    # 6.2 Add noise using (10 * np.log(1 + noise_level)) SNR and White Gaussian Noise
    ts_n2 = pd.Series((ts + np.sqrt((10 ** (-snr / 10))) * r)[0])  # Noisy signal

    return ts_n1, ts_n2


# Helper wrapper to create non-daemonic processes in the pool of parallel processes so these can still be split.
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

