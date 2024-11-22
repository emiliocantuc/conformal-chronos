import numpy as np
import scipy.stats as stats
import torch
from einops import rearrange, reduce

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


# For "classical" (non-conformal) interval methods to be used as baselines
# maybe subclass for the conformal wrapper?
class ChronosWrapper:

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