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
    # We've also added RAG as in: https://arxiv.org/abs/2411.08249

    def __init__(self, pipeline, horizon, pred_kwargs): # False
        self.pipeline = pipeline
        self.horizon = horizon
        self.pred_kwargs = {k:v for k,v in pred_kwargs.items() if k != 'rag'}
        self.rag = pred_kwargs.get('rag', False)
    
    @torch.no_grad()
    def __call__(self, x, _ = None):
        # x is "shaped" [n_samples, seq_len, n_features]
        # but its a list of length n_samples, each element is a tensor of shape [seq_len, n_features]
        # Note: n_features is always 1 (as far as I can tell)
        assert x[0].shape[1] == 1, "n_features must be 1?"

        x = rearrange(x, 'n_samples seq_len 1 -> n_samples seq_len')
        if self.rag: x = self.augment_context(x)

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
    
    def augment_context(self, x):
        # Retrieves similar sequences from the training set and appends them to the context

        # Embed the series in the context
        query_embeddings, _ = self.pipeline.embed(x)
        aug_x = []

        # For every series
        for series, emb in zip(x, query_embeddings):

            # Find the closest embeddings in the training set
            diffs = emb - self.embeddings # [n_train, seq_len, emb_dim]
            dists = torch.norm(diffs, dim = -1, p = 2).sum(-1)
            retrieved_ix = dists.argmin()
            retrieved_seq = self.sequences[retrieved_ix]

            # Center and scale retrieved to query
            retrieved_seq = (retrieved_seq - retrieved_seq.mean()) / retrieved_seq.std()
            retrieved_seq = retrieved_seq * series.std() + series.mean()

            # Make sure the retrieved ends where the query starts (and del duplicate element)
            retrieved_seq += series[0] - retrieved_seq[-1]
            retrieved_seq = retrieved_seq[:-1]

            aug_x.append(torch.concat([retrieved_seq, series], dim = 0))

        return aug_x

    # Chronos is not "fit" but we do "RAG" here (note: could also fine-tune, etc.)
    def fit(self, train_dataset, batch_size, *args, **kwargs):

        if not self.rag: return

        self.embeddings, self.sequences = [], []
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

        for x, y, lengths in train_loader:
            x = rearrange(x, 'n_samples seq_len 1 -> n_samples seq_len')
            embs, _ = self.pipeline.embed(x)
            self.embeddings.append(embs)
            self.sequences.append(torch.concat([x, y.squeeze()], dim = 1))
        
        self.embeddings = torch.cat(self.embeddings, dim = 0)
        self.sequences  = torch.cat(self.sequences, dim = 0)

    def eval(self): pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state = {k:v for k,v in state.items() if k in ['horizon', 'calibration_scores']}
        return state


class CFChronos(CFRNN):
    # CFRNN subclass that uses Chronos as the auxiliary forecaster

    def __init__(self, pipeline, pred_kwargs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxiliary_forecaster = _ChronosPipelineWrapper(pipeline, self.horizon, pred_kwargs)
        self.requires_auxiliary_fit = pred_kwargs.get('rag', False)