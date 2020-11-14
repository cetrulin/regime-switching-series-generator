import numpy as np
import pandas as pd
import itertools
import multiprocessing
from functools import partial
from dataclasses import dataclass

# RPY packages to run rugarch in python
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# For logging
import calendar
import time
import logging
import os

timestamp = calendar.timegm(time.gmtime())
log_filename = f"logs/model_{timestamp}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, filemode='w', level=logging.INFO)
file_handler = logging.FileHandler(log_filename, mode="w", encoding=None, delay=False)


@dataclass
class Model:
    rugarch_lib_instance = None

    # Model attributes and data
    id: int
    raw_input_path: float
    input_ts: pd.Series()
    rec_price: float  # init price for reconstruction from returns
    probability: float
    ARMAGARCH_preconf: list()

    # Define fitted params
    p = 0
    o = 0
    q = 0
    g_p = 0
    g_q = 0
    ARMAGARCHspec = None

    def get_lags(self):
        """ This function returns a list of lags (p, o, q) for the current model. """
        return [self.p, self.o, self.q]

    def set_lags(self, p_, o_, q_, g_p_=1, g_q_=1):
        """ This function sets a list of lags (p, o, q) for the current model and lags also for GARCH g_pq."""
        self.o = o_
        self.p = p_
        self.q = q_
        self.g_p = g_p_
        self.g_q = g_q_

    # def load_preconf(self):
    #     """ This function sets p, o, q for ARMA and GARCH if preloaded (distint from 0). """
    #     if len(self.ARMAGARCH_preconf) == 5 & self.ARMAGARCH_preconf[0] > -1:
    #         self.o, self.p, self.q, self.g_p, self.g_q = self.ARMAGARCH_preconf
    #         return True
    #     return False

    def fit(self, current_series, lib_conf,  p_=1, q_=1, garch_param1=1, garch_param2=1):
        """
        This function uses rugarch to fit an ARMA-GARCH model for p, 0, q (d!=0 only in ARIMA-GARCH)
        :param current_series:
        :param p_: p lags of the best fitted ARMA model.
        :param q_: q lags of the best fitted ARMA model.
        :param garch_param1: garch order 0
        :param garch_param2: garch order 1
        :param lib_conf: TSpackage for R library to use
        :return: trained/fitted model
        """

        try:
            # Initialize R GARCH model
            self.rugarch_lib_instance = importr(lib_conf['lib'], lib_conf['env'])
            spec = self.rugarch_lib_instance.ugarchspec(
                mean_model=robjects.r(f'list(armaOrder=c({p_},{q_}), include.mean=T)'),
                # Using student T distribution usually provides better fit
                variance_model=robjects.r(f'list(garchOrder=c({garch_param1},{garch_param2}))'),
                distribution_model='sged')  # 'std'

            # Train R GARCH model on returns as %
            numpy2ri.activate()  # Used to convert training set to R list for model input
            model = self.rugarch_lib_instance.ugarchfit(
                spec=spec,
                data=np.array(current_series),
                out_sample=1  # remember: the 1st forecast should be reproducible in the reconst., as the real value exists
            )
            numpy2ri.deactivate()
            # Checks
            coef = model.slots['fit'].rx2('coef')  # this gives a vector of coefficients
            # [mu, ar1, ar2, ar3, ar4, ma1, ma2, ma3, omega, alpha, beta, skw, shape]
            omega, alpha, beta = coef[-5], coef[-4], coef[-3]
            assert omega > 0 and alpha > 0 and beta > 0  # this ensures that the predicted variance > 0 at all times.
            assert alpha + beta < 1  # this ensures that the predicted variance always returns to the long run variance

            self.ARMAGARCHspec = self.get_spec(model)
            self.rugarch_lib_instance = None
        except Exception:
            print(f'Model{model} does not fit for the desired params.')

        return model

    def get_best(self, current_series, conf, lib_conf):
        """
        This function uses rugarch to find the best params for ARMA-GARCH for p, 0, q (d!=0 only in ARIMA-GARCH)
        :param current_series:
        :param conf: tool params
        :param lib_conf: TSpackage for R library to use
        :return: trained/fitted model
        """
        # If best config pre-loaded, do not search
        if (len(self.ARMAGARCH_preconf) == 5) & (self.ARMAGARCH_preconf[0] > -1):
            p, o, q, p_g, q_g = self.ARMAGARCH_preconf
            return 0.0, self.ARMAGARCH_preconf, self.fit(current_series, lib_conf,
                                                         p_=p, q_=q, garch_param1=p_g, garch_param2=q_g)
        else:
            # Fitting in parallel according to the ARMA value 'p'.
            pool = multiprocessing.Pool(processes=conf['pq_rng'])
            mapped = pool.map(partial(self.param_search, conf, current_series, lib_conf), range(1, conf['pq_rng'] + 1))
            best_models_dict = dict(map(reversed, tuple(mapped)))

            # Retrieving the best result across all threads (each value of p)
            best_aic, best_mdl, best_order = self.compute_intermediate_results(best_models_dict)
            return best_aic, best_order, best_mdl

    def compute_intermediate_results(self, best_models_dict):
        """
        This function gets the best model from the intermediate results (one thread per value of p in ARMA tried).
        """
        best_aic = np.inf
        for i in best_models_dict.values():  # small loop to reduce results from map function
            if i['aic'] < best_aic:  # This must be consistent with the same comparison in each thread
                best_aic = i['aic']
                best_order = i['order']
                best_mdl = i['mdl']
        return best_aic, best_mdl, best_order

    def param_search(self, conf, current_series, lib_conf, p):
        best_aic, best_order, best_mdl = np.inf, None, None
        self.rugarch_lib_instance = importr(lib_conf['lib'], lib_conf['env'])

        for q, g_p, g_q in itertools.product(range(1, conf['pq_rng'] + 1),
                                             range(1, conf['garch_pq_rng'] + 1),
                                             range(1, conf['garch_pq_rng'] + 1)):
            try:
                # print(f'Trying params: {(i, 0, j, k, h)} on model {self.id}.')
                # Initialize R GARCH model
                spec = self.rugarch_lib_instance.ugarchspec(
                    mean_model=robjects.r(f'list(armaOrder=c({p},{q}), include.mean=T)'),
                    # Using student T distribution usually provides better fit
                    variance_model=robjects.r(f'list(garchOrder=c({g_p},{g_q}))'),
                    distribution_model='sged')  # 'std'

                # Train R GARCH model on returns as %
                numpy2ri.activate()  # Used to convert training set to R list for model input
                tmp_mdl = self.rugarch_lib_instance.ugarchfit(
                    spec=spec,
                    data=np.array(current_series),
                    out_sample=1  # remember: the 1st forecast should be reproducible in the reconst.,
                    # as the real value exists
                )
                numpy2ri.deactivate()
                # Checks - see description of checks in fit function
                coef = tmp_mdl.slots['fit'].rx2('coef')
                omega, alpha, beta = coef[-5], coef[-4], coef[-3]
                cond = omega > 0 and alpha > 0 and beta > 0 and alpha + beta < 1  # print(cond)
                assert cond

                # [0 AIC, 1 BIC, 2 Shibata, 3 Hannan - Quinn ]
                tmp_aic, tmp_bic, tmp_sic, tmp_hic = self.get_infocrit(tmp_mdl)
                print(f'Trying params: {(p, 0, q, g_p, g_q)} on model {self.id} - '
                      f'AIC: {tmp_aic:6.5f} | BIC: {tmp_bic:6.5f} | SIC: {tmp_sic:6.5f} | HQIC: {tmp_hic:6.5f}')
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (p, 0, q, g_p, g_q)  # o = 0 in ARMAGARCH
                    best_mdl = tmp_mdl
            except Exception:
                continue
        print(f'////////////\nBEST Model {self.id} p={p} -> aic: {best_aic:6.5f} | order: {best_order}\n////////////')
        logging.info(f'////////////\nBEST Model {self.id} p={p} -> aic: {best_aic:6.5f} | order: {best_order}\n////////////')

        self.rugarch_lib_instance = None
        return {'aic': best_aic, 'mdl': best_mdl, 'order': best_order}, p

    def forecast(self, ts: list(), lib_conf, roll: int = 1000, n_steps: int = 1):
        """
        This function calls the R rugarch library to produce a the ARMA-GARCH forecast.
        :param self - current selected model
        :param ts - current series
        :param lib_conf: TSpackage for R library to use
        :param roll: max-size of rolling window fed to forecast
        :param n_steps: number of steps ahead forecasted
        out_sample: Optional.
            If a specification object is supplied, indicates how many data points to keep for out of sample testing.
        n.roll argument which controls how many times to roll the n.ahead forecast.
            The default argument of n.roll = 0 denotes no rolling and returns the standard n.ahead forecast.
            Critically, since n.roll depends on data being available from which to base the rolling forecast,
            the ugarchfit function needs to be called with the argument out.sample being at least as large as
            the n.roll argument, or in the case of a specification being used instead of a fit object,
            the out.sample argument directly in the forecast function.
        :return forecast or the next time horizon
        """
        self.rugarch_lib_instance = importr(lib_conf['lib'], lib_conf['env'])
        forecast = self.rugarch_lib_instance.ugarchforecast(self.ARMAGARCHspec,
                                                            # Rolling window of latest 1000 values to avoid huge values
                                                            #  in forecasts when switching some models.
                                                            data=ts[-roll:] if len(ts) > roll else ts,
                                                            n_ahead=n_steps, n_roll=0, out_sample=0)
        # print(f'len: {len(ts)}')
        # print(f'len: {len(ts)}  - {np.array(forecast.slots["forecast"].rx2("seriesFor")).flatten()[0]}')

        self.rugarch_lib_instance = None
        return np.array(forecast.slots['forecast'].rx2('seriesFor')).flatten()  # n_steps

    def set_spec_from_model(self, model):
        """ This function sets an spec of a fitted model into the object."""
        self.ARMAGARCHspec = self.get_spec(model)

    """ 
    This function sets the fitted parameters in a rugarch spec. 
    @:param fitted: fitted model
    """
    get_spec = robjects.r('''function(fitted) {
                                spec=getspec(fitted)
                                setfixed(spec) <- as.list(coef(fitted))
                                return(spec)
                             }''')

    """
    This function gets the information criteria of a given fitted model.
    @:return vector of 4 pos containing 4 metrics: [Akaike (AIC), Bayes (BIC), Shibata, Hannan - Quinn ]
    """
    get_infocrit = robjects.r('''function(fitted) {
                                    return(infocriteria(fitted))
                                 }''')
