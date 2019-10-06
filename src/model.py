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
    ARMAGARCHspec = None

    def get_lags(self):
        """ This function returns a list of lags (p, o, q) for the current model. """
        return [self.p, self.o, self.q]

    def set_lags(self, p_, o_, q_):
        """ This function sets a list of lags (p, o, q) for the current model. """
        self.o = o_
        self.p = p_
        self.q = q_

    def fit(self, current_series, p_, q_, lib_conf):
        """
        This function uses rugarch to fit an ARMA-GARCH model for p, 0, q (d!=0 only in ARIMA-GARCH)
        :param current_series:
        :param p_: p lags of the best fitted ARMA model.
        :param q_: q lags of the best fitted ARMA model.
        :param lib_conf: TSpackage for R library to use
        :return: trained/fitted model
        """
        # Initialize R GARCH model
        self.rugarch_lib_instance = importr(lib_conf['lib'], lib_conf['env'])
        spec = self.rugarch_lib_instance.ugarchspec(
            mean_model=robjects.r(f'list(armaOrder=c({p_},{q_}), include.mean=T)'),
            # Using student T distribution usually provides better fit
            variance_model=robjects.r('list(garchOrder=c(1,1))'),
            distribution_model='sged')  # 'std'

        # Train R GARCH model on returns as %
        numpy2ri.activate()  # Used to convert training set to R list for model input
        model = self.rugarch_lib_instance.ugarchfit(
            spec=spec,
            data=np.array(current_series),
            out_sample=1  # remember: the 1st forecast should be reproducible in the reconst., as the real value exists
        )
        numpy2ri.deactivate()
        # TODO IMP:  assert self.model.beta + self.model.alpha <= 1
        self.ARMAGARCHspec = self.get_spec(model)
        self.rugarch_lib_instance = None

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
