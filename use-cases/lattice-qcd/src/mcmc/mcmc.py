# Copyright (c) 2021-2022 Javad Komijani


import torch
import numpy as np
import copy

from ..lib.combo import estimate_logz, fmt_val_err
from ..lib.stats import Resampler

seize = lambda var: var.detach().cpu().numpy()


# =============================================================================
class MCMCSampler:
    """Perform Markov chain Monte Carlo simulation."""

    def __init__(self, model):
        self._model = model
        self.history = MCMCHistory()
        self._ref = dict(sample=None, logq=None, logp=None, logqp=None)

    @torch.no_grad()
    def sample(self, batch_size=1, **kwargs):
        return self.sample__(batch_size=batch_size, **kwargs)[0]

    @torch.no_grad()
    def sample_(self, batch_size=1, **kwargs):
        return self.sample__(batch_size=batch_size, **kwargs)[:2]

    @torch.no_grad()
    def sample__(self, batch_size=1, bookkeeping=False):
        """Return a batch of Monte Carlo Markov Chain samples generated using
        independence Metropolis method.
        Acceptances/rejections occur proportionally to how well/poorly
        the model density matches the desired density.

        The calculations are done by
            1) Drawing raw samples as proposed samples
            2) Apply Metropolis accept/reject to the proposed samples
        """
        y, logq, logp = self._model.posterior.sample__(batch_size=batch_size)

        if bookkeeping:
            self.history.bookkeeping(raw_logq=logq, raw_logp=logp)

        mydict = dict(bookkeeping=bookkeeping)
        y, logq, logp = self._accept_reject_step(y, logq, logp, **mydict)

        if bookkeeping:
            self.history.bookkeeping(logq=logq, logp=logp)

        return y, logq, logp

    @torch.no_grad()
    def _accept_reject_step(self, y, logq, logp, bookkeeping=False):
        # Return (y, logq, logp) after Metropolis accept/reject step to the
        # proposed ones in the input

        ref = self._ref
        logqp_ref = ref['logqp']

        # 2.1) Calculate the accept/reject status of the samples
        accept_seq = Metropolis.calc_accept_status(seize(logq - logp), logqp_ref)

        # 2.2) Handle the first item separately
        if accept_seq[0] == False:
            y[0], logq[0], logp[0] = ref['sample'], ref['logq'], ref['logp']

        # 2.3) Handle the rest items by calculating accept_ind
        accept_ind = Metropolis.calc_accept_indices(accept_seq)

        accept_ind_torch = torch.LongTensor(accept_ind).to(y.device)
        func = lambda x: x.index_select(0, accept_ind_torch)
        y, logq, logp = func(y), func(logq), func(logp)

        # Update '_ref' dictionary for the next round
        ref['sample'] = y[-1]
        ref['logq'] = logq[-1].item()
        ref['logp'] = logp[-1].item()
        ref['logqp'] = ref['logq'] - ref['logp']

        self.history.bookkeeping(accept_rate=np.mean(accept_seq))  # always save
        if bookkeeping:
            self.history.bookkeeping(accept_seq=accept_seq, accept_ind=accept_ind)

        return y, logq, logp

    @torch.no_grad()
    def serial_sample_generator(self, n_samples, batch_size=16):
        """Generate Monte Carlo Markov Chain samples one by one"""
        unsqz = lambda a, b, c: (a.unsqueeze(0), b.unsqueeze(0), c.unsqueeze(0))
        for i in range(n_samples):
            ind = i % batch_size  # the index of the batch
            if ind == 0:
                y, logq, logp = self.sample__(batch_size)
            yield unsqz(y[ind], logq[ind], logp[ind])

    @torch.no_grad()
    def calc_accept_rate(self, n_samples=1024, batch_size=None,
            n_resamples=10, method='shuffling'):
        """Calculate acceptance rate from logqp = log(q) - log(p)"""

        # First, draw (raw) samples
        if batch_size is None or batch_size > n_samples:
            batch_size = n_samples
        n_batches = np.ceil(n_samples/batch_size).astype(int)
        logqp = np.zeros(n_batches * batch_size)
        for k in range(n_batches):
            _, logq, logp = self._model.posterior.sample__(batch_size=batch_size)
            logqp[k*batch_size: (k+1)*batch_size] = seize(logq - logp)

        # Now calculate the mean and std of acceptance rate (by shuffling)
        mean, std = self.estimate_accept_rate(logqp)
        return mean, std

    @staticmethod
    @torch.no_grad()
    def estimate_accept_rate(logqp, n_resamples=10, method='shuffling'):
        """Estimate acceptance rate from shuffling logqp"""
        calc_rate = lambda logqp: np.mean(Metropolis.calc_accept_status(logqp))
        resampler = Resampler(method)
        mean, std = resampler.eval(logqp, fn=calc_rate, n_resamples=n_resamples)
        return mean, std

    def log_prob(self, y, action_logz=0):
        """Returns log probability up to an additive constant."""
        return -self._model.action(y) - action_logz


# =============================================================================
class BlockedMCMCSampler(MCMCSampler):
    """Perform Markov chain Monte Carlo simulation with blocking."""

    @torch.no_grad()
    def sample(self, batch_size=1, **kwargs):
        return self.sample__(batch_size=batch_size, **kwargs)[0]

    @torch.no_grad()
    def sample_(self, batch_size=1, **kwargs):
        return self.sample__(batch_size=batch_size, **kwargs)[:2]

    @torch.no_grad()
    def sample__(self, batch_size=1, n_blocks=1, bookkeeping=False):
        """Return a batch of mcmc samples."""

        prior = self._model.prior
        net_ = self._model.net_
        action = self._model.action

        try:
            x = net_.backward(self._ref['sample'].unsqueeze(0))[0]
            logqp_ref = self._ref['logqp']
        except:
            print("Starting from scratch & setting logqp_ref to None")
            x = prior.sample(1)
            logqp_ref = None

        nvar = prior.nvar
        if isinstance(n_blocks, int):
            block_len = nvar // n_blocks
            assert block_len * n_blocks == nvar
        else:
            block_len = nvar
            n_blocks = 1

        prior.setup_blockupdater(block_len)

        cfgs = torch.empty((batch_size, *prior.shape))
        logq = torch.empty((batch_size,))
        logp = torch.empty((batch_size,))
        accept_seq = np.empty((batch_size, n_blocks), dtype=bool)

        for ind in range(batch_size):
            accept_seq[ind], logqp_ref = self.sweep(x, n_blocks, logqp_ref)  # in-place sweeper
            y, logJ = net_(x)
            logq[ind] = prior.log_prob(x) - logJ
            logp[ind] = -action(y)
            cfgs[ind] = y

        # update '_ref' dictionary for the next round
        self._ref['sample'] = y[-1]
        self._ref['logq'] = logq[-1].item()
        self._ref['logp'] = logp[-1].item()
        self._ref['logqp'] = (logq[-1] - logp[-1]).item()

        self.history.bookkeeping(accept_rate=np.mean(accept_seq))  # always save
        if bookkeeping:
            self.history.bookkeeping(logq=logq, logp=logp)
            self.history.bookkeeping(accept_seq=accept_seq.ravel())

        return cfgs, logq, logp

    @torch.no_grad()
    def sweep(self, x, n_blocks=1, logqp_ref=None):
        """In-place sweeper."""
        prior = self._model.prior
        net_ = self._model.net_
        action = self._model.action

        accept_seq = np.empty(n_blocks, dtype=bool)
        lrand_arr = np.log(np.random.rand(n_blocks))

        for ind in range(n_blocks):
            prior.blockupdater(x, ind)  # in-place updater
            y, logJ = net_(x)
            logq = prior.log_prob(x) - logJ
            logp = -action(y)
            # Metropolis acceptance condition:
            if ind == 0 and logqp_ref is None:
                accept_seq[ind] = True
            else:
                accept_seq[ind] = lrand_arr[ind] < logqp_ref - (logq - logp)[0]
            if accept_seq[ind]:
                logqp_ref = (logq - logp).item()
            else:
                prior.blockupdater.restore(x, ind)

        return accept_seq, logqp_ref


# =============================================================================
class MCMCHistory:
    """For bookkeeping of Perform Markov chain Monte Carlo simulation."""

    def __init__(self):
        self.reset_history()

    def reset_history(self):
        self.logq = []
        self.logp = []
        self.raw_logq = []
        self.raw_logp = []
        self.accept_seq = []
        self.accept_ind = []
        self.accept_rate = []

    def report_summary(self, since=0, asstr=False):

        if asstr:
            fmt = lambda mean, std: fmt_val_err(mean, std, err_digits=2)
        else:
            fmt = lambda mean, std: (mean, std)

        logqp = torch.tensor(self.logq[-1] - self.logp[-1])  # estimate_logz
        accept_rate = torch.tensor(self.accept_rate)
        mean_std = lambda t: (t.mean().item(), t.std().item())

        report = {'logqp': fmt(*mean_std(logqp)),
                  'logz': fmt(*estimate_logz(logqp)),
                  'accept_rate': fmt(*mean_std(accept_rate))
                  }
        return report

    def bookkeeping(self,
            logq=None,
            logp=None,
            raw_logq=None,
            raw_logp=None,
            accept_seq=None,
            accept_rate=None,
            accept_ind=None
            ):

        if raw_logq is not None:
            # make a copy of the raw one in case it is manually changed
            self.raw_logq.append(copy.copy(seize(raw_logq)))

        if raw_logp is not None:
            # make a copy of the raw one in case it is manually changed
            self.raw_logp.append(copy.copy(seize(raw_logp)))

        if logq is not None:
            self.logq.append(seize(logq))

        if logp is not None:
            self.logp.append(seize(logp))

        if accept_rate is not None:
            self.accept_rate.append(accept_rate)

        if accept_seq is not None:
            self.accept_seq.append(accept_seq)

        if accept_ind is not None:
            self.accept_ind.append(accept_ind)

    @property
    def logqp(self):
        return [(logq - logp) for (logq, logp) in zip(self.logq, self.logp)]

    @property
    def raw_logqp(self):
        return [(logq - logp) for (logq, logp) in zip(self.raw_logq, self.raw_logp)]


# =============================================================================
class Metropolis:
    """
    To perform Metropolis-Hastings accept/reject step in Markov chain Monte
    Carlo simulation.
    """

    @staticmethod
    @torch.no_grad()
    def calc_accept_status(logqp, logqp_ref=None):
        """Returns accept/reject using Metropolis algorithm."""
        # Much faster if inputs are np.ndarray & python number (NO tensor)
        if logqp_ref is None:
            logqp_ref = logqp[0]
        status = np.empty(len(logqp), dtype=bool)
        rand_arr = np.log(np.random.rand(logqp.shape[0]))
        for i, logqp_i in enumerate(logqp):
            status[i] = rand_arr[i] < (logqp_ref - logqp_i)
            if status[i]:
                logqp_ref = logqp_i
        return status  # also called accept_seq

    def calc_accept_indices(accept_seq):
        """Return indices of output of Metropolis-Hasting accept/reject step."""
        indices = np.arange(len(accept_seq))
        cntr = 0
        for ind, accept in enumerate(accept_seq):
            if accept:
                cntr = ind  # update `cntr`
            else:
                indices[ind] = cntr  # reduce the index to the `cntr` value
        return indices

    @staticmethod
    def calc_accept_count(accept_seq):
        """Count how many repetition till next accepted configuration."""
        ind = np.where(accept_seq)[0]  # index of True ones
        multiplicity = ind[1:] - ind[:-1]  # count except for the last
        return multiplicity
        # return multiplicity, [ind[0], len(accept_seq) - ind[-1]]

    @staticmethod
    def calc_tau_rejections_prob(accept_seq, max_tau=100):
        """Return the probability of tau rejections in a rows"""

        p_tau = np.zeros(max_tau)

        rej_seq = ~accept_seq
        tau_rej_seq = rej_seq

        p_tau[0] = np.mean(tau_rej_seq)
        for i in range(1, max_tau):
            tau_rej_seq = tau_rej_seq[:-1] & rej_seq[i:]
            p_tau[i] = np.mean(tau_rej_seq)

        return p_tau


class ModifiedMetropolis(Metropolis):
    """
    To perform a modified version of Metropolis-Hastings accept/reject step in
    Markov chain Monte Carlo simulation.
    """

    @staticmethod
    @torch.no_grad()
    def calc_accept_status(logqp, logqp_ref=None, tau=0):
        """Returns accept/reject using Metropolis algorithm."""
        # Much faster if inputs are np.ndarray & python number (NO tensor)
        if logqp_ref is None:
            logqp_ref = logqp[0]
        status = np.empty(len(logqp), dtype=bool)
        lrand_arr = np.log(np.random.rand(logqp.shape[0]))
        for i, logqp_i in enumerate(logqp):
            x = logqp_ref - logqp_i
            status[i] = lrand_arr[i] < -(tau * x**2 + (-x if x < 0 else 0))
            if status[i]:
                logqp_ref = logqp_i
        return status  # also called accept_seq


# =============================================================================
