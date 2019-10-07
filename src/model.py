import numpy as np
import pandas as pd
from dataclasses import dataclass

# RPY packages to run rugarch in python
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


@dataclass
class Model:
    rugarch_lib_instance = None

    # Model attributes and data
    id: int
    raw_input_path: float
    input_ts: pd.Series()
    rec_price: float  # init price for reconstruction from returns
    probability: float

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

    def get_best(self, current_series, conf, lib_conf):
        """
        This function uses rugarch to find the best params for ARMA-GARCH for p, 0, q (d!=0 only in ARIMA-GARCH)
        :param current_series:
        :param conf: tool params
        :param lib_conf: TSpackage for R library to use
        :return: trained/fitted model
        """
        self.rugarch_lib_instance = importr(lib_conf['lib'], lib_conf['env'])
        best_aic = np.inf
        best_order = None
        best_mdl = None

        for i in range(conf['pq_rng']):
            for j in range(conf['pq_rng']):
                for k in range(conf['garch_pq_rng']):
                    for h in range(conf['garch_pq_rng']):
                        try:
                            # print(f'Trying params: {(i, 0, j, k, h)} on model {self.id}.')
                            # Initialize R GARCH model
                            spec = self.rugarch_lib_instance.ugarchspec(
                                mean_model=robjects.r(f'list(armaOrder=c({i},{j}), include.mean=T)'),
                                # Using student T distribution usually provides better fit
                                variance_model=robjects.r(f'list(garchOrder=c({k},{h}))'),
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
                            assert omega > 0 and alpha > 0 and beta > 0 and alpha + beta < 1

                            tmp_aic = self.get_infocrit(tmp_mdl)[0]  # [0 AIC, 1 BIC, 2 Shibata, 3 Hannan - Quinn ]
                            # print(f'AIC: {tmp_aic}\n')
                            print(f'Trying params: {(i, 0, j, k, h)} on model {self.id}. AIC is: {tmp_aic}')
                            if tmp_aic < best_aic:
                                best_aic = tmp_aic
                                best_order = (i, 0, j, k, h)  # o = 0 in ARMAGARCH
                                best_mdl = tmp_mdl
                        except:
                            continue
        print('model {} -> aic: {:6.5f} | order: {}'.format(self.id, best_aic, best_order))
        self.rugarch_lib_instance = None
        return best_aic, best_order, best_mdl

    def forecast(self, ts: list(), lib_conf):
        """
        This function calls the R rugarch library to produce a the ARMA-GARCH forecast.
        :param self - current selected model
        :param ts - current series
        :param lib_conf: TSpackage for R library to use
        :return forecast or the next time horizon
        """
        self.rugarch_lib_instance = importr(lib_conf['lib'], lib_conf['env'])
        forecast = self.rugarch_lib_instance.ugarchforecast(self.ARMAGARCHspec, data=ts,
                                                            n_ahead=1, n_roll=0, out_sample=0)
        print(f'len: {len(ts)}')
        print(np.array(forecast.slots['forecast'].rx2('seriesFor')).flatten())

        self.rugarch_lib_instance = None
        return np.array(forecast.slots['forecast'].rx2('seriesFor')).flatten()[0]

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
