import numpy as np
import scipy.stats as stats
import torch
from einops import rearrange, reduce

from models.cfrnn import CFRNN

# === HEURISTIC BASELINE INTERVALS ===
# In all cases the shapes are
# forecast  :(batch, n_sampes, horizon)
# output    :(batch, interval, horizon)

# I'll add more
def naive_quantile_int(forecast, alpha):
    b, n, h = forecast.shape
    ints = torch.quantile(forecast, dim = 1, q = torch.tensor([alpha, 1. - alpha]).to(forecast))
    return rearrange(ints, 'i b h -> b i h')

def naive_gaussian(forecast, alpha):
    b, n, h = forecast.shape
    mu = forecast.mean(dim = 1)
    se = forecast.std(dim = 1) / np.sqrt(n)
    z = stats.norm.ppf(1 - alpha)
    return rearrange([mu - z*se, mu + z*se], 'i b h -> b i h')

def bonferroni(forecast, alpha, other_f, **other_f_kwargs):
    # Wrapps another interval function applying Bonferroni correction
    # (just devide alpha by the horizon size)
    b, n, h = forecast.shape
    other_f = other_f(forecast, alpha = alpha / h, **other_f_kwargs)
    return other_f


class HeuristicChronosWrapper:
    # For "classical"/heuristic (non-conformal) interval methods to be used as baselines

    def __init__(self, horizon, coverage, pipeline, pred_kwargs, int_func, **kwargs):

        self.horizon = horizon
        self.coverage = coverage

        self.pipeline = pipeline
        self.pred_kwargs = pred_kwargs
        self.int_func = int_func

    def fit(*args, **kwargs): pass # Calibration could happen here for the subclass

    def predict(self, X_test, alpha):
        # X_test is "shaped" [n_samples, max_seq_len, n_features]
        # but its a list of length n_samples, each element is a tensor of shape [max_seq_len, n_features]
        # Note: n_features is always 1 (as far as I can tell)
        assert X_test[0].shape[1] == 1, "n_features must be 1?"

        # Adapt to feed to chronos: must be [n_samples, seq_len]
        X = rearrange(X_test, 'n_samples seq_len 1 -> n_samples seq_len')

        # [n_samples, num_samples, horizon] (where num_samples is set by us in pred_kwargs)
        forecast = self.pipeline.predict(
            context = X,
            prediction_length = self.horizon,
            **self.pred_kwargs
        )

        # Point forecast is the mean(?)
        y_pred = reduce(forecast, 'n_samples num_samples horizon -> n_samples horizon', 'mean')
        ints = self.int_func(forecast, alpha = alpha)
        y_l_approx, y_u_approx = rearrange(ints, 'b i h -> i b h')
        return y_pred, y_l_approx, y_u_approx
    

class _ChronosPipelineWrapper:
    # Wraps a Chronos pipeline to be used as an "auxiliary forecaster" from CFRNN
    # Required to match the CFRNN interface

    def __init__(self, pipeline, horizon, pred_kwargs):
        self.pipeline = pipeline
        self.horizon = horizon
        self.pred_kwargs = pred_kwargs
    
    @torch.no_grad()
    def __call__(self, x, _ = None):
        # x is "shaped" [n_samples, seq_len, n_features]
        # but its a list of length n_samples, each element is a tensor of shape [seq_len, n_features]
        # Note: n_features is always 1 (as far as I can tell)
        assert x[0].shape[1] == 1, "n_features must be 1?"

        x = rearrange(x, 'n_samples seq_len 1 -> n_samples seq_len')

        # Expects output [n_samples, ], state
        forecast = self.pipeline.predict(
            context = x,
            prediction_length = self.horizon,
            **self.pred_kwargs
        )
        y_pred = reduce(forecast, 'n_samples num_samples horizon -> n_samples horizon', 'mean')

        # Need to add trailing dimention to match expected shape
        y_pred = rearrange(y_pred, 'n_samples horizon -> n_samples horizon 1')
        return y_pred, None # expects state, but we don't care

    def eval(self): pass

    # Chronos should not be fit -- although we could try fine-tuning here
    def fit(self, *args, **kwargs): raise NotImplementedError

    
class CFChronos(CFRNN):
    # CFRNN subclass that uses Chronos as the auxiliary forecaster

    def __init__(self, pipeline, pred_kwargs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxiliary_forecaster = _ChronosPipelineWrapper(pipeline, self.horizon, pred_kwargs)
        self.requires_auxiliary_fit = False