"""
drift_core.py  --  EXTENSION 2: DRIFT-AWARE cascades (works for synthetic AND real data)

PROBLEM
    OPT-ALG (new_algorithm.optimize + the lambda probability update of sim.py) is designed with
    ONE fixed description of the stream: exit probabilities p_i and persistence lambda_i measured
    at design time.  If the stream drifts (users change, activity regime changes, the sensor
    population changes ...), those numbers become stale and the cascade it builds is no longer
    the optimal one.

OUR METHOD  --  DA-OPT  (Drift-Adaptive OPT-ALG)
    Instead of a threshold-triggered rebuild, we keep OPT-ALG untouched (same optimize(), same
    explore_OA) and only change WHERE ITS PROBABILITIES COME FROM:

      * The probability that the next input is classified by stage j, given that the previous one
        was classified by stage r, is represented by a mixture of three experts
            E0  design model       : the original (p_i, lambda_i) model of OPT-ALG (frozen)
            E1  slow adaptive model: transition counts with long memory  (gamma = 0.998)
            E2  fast adaptive model: transition counts with short memory (gamma = 0.96)
        E1/E2 are Dirichlet-smoothed transition rows  N[r][j]  shrunk toward the design model.
      * Mixture weights = exp(-discounted predictive log-loss) of each expert (Bayesian model
        averaging with forgetting).  While the stream looks like the design-time stream, E0 wins and
        DA-OPT behaves exactly like OPT-ALG (no price to pay when there is no drift).  When the
        stream drifts, E0's loss grows, the weight moves to E1/E2 within tens of inputs, and the
        cascade is re-optimized with the CURRENT statistics.
      * DRIFT DETECTION is a by-product: an alarm is raised when the weight of the design model
        falls below 0.2 (and re-armed when it recovers above 0.6).
      * The cascade is rebuilt for every input with the ORIGINAL optimize() (OptBuild), exactly as
        OPT-ALG already does for every input.  The only extra work is updating three small count
        matrices: about 35 microseconds per input in this Python/numpy implementation (measured,
        independent of the number of classifiers), i.e. small but not zero.

BASELINE  --  DRIFT-v1 (re-implementation of the description given by the team)
    window monitoring of classifier-transition statistics, transition-based lambda estimation,
    threshold-conditional cascade rebuilding with the original OptBuild, paired one-sided Student
    t-test (95 %) and a minimum 1 % improvement requirement before a candidate is deployed.
"""
import numpy as np
import exploration, new_algorithm
from scipy import stats


# ----------------------------------------------------------------------------
# Helpers shared by all methods (same maths as the original sim.py)
# ----------------------------------------------------------------------------
def cdf_to_pi(cdf):
    return np.diff(np.concatenate([[0.0], np.asarray(cdf, dtype=float)]))


def with_cdf(cascade, cdf):
    """Same cascade structure ({'id','c','p'}), with the exit CDF replaced."""
    return [{'id': x['id'], 'c': x['c'], 'p': float(cdf[i])} for i, x in enumerate(cascade)]


def opt_alg_update(cdf, prev_id, lam):
    """The ORIGINAL OPT-ALG probability update (identical to sim.py's cascade_prob_update)."""
    if prev_id is None:
        return list(cdf)
    l = lam[prev_id]
    return [(1 - l) * p if i < prev_id else l + (1 - l) * p for i, p in enumerate(cdf)]


def cost_with_cdf(cascade, cdf, true_id):
    """Optimize the cascade for this CDF (Algorithm 1) and execute it on the real exit stage."""
    c, _ = exploration.explore_OA(with_cdf(cascade, cdf), None, true_id)
    return c


# ----------------------------------------------------------------------------
# OUR METHOD
# ----------------------------------------------------------------------------
class _Expert:
    def __init__(self, prior_rows, prior_marg, gamma, alpha):
        n = len(prior_marg)
        self.g, self.a = gamma, alpha
        self.prior, self.pm = prior_rows, prior_marg
        self.T = np.zeros((n, n))
        self.M = np.zeros(n)

    def row(self, r):
        if r is None:
            return (self.a * self.pm + self.M) / (self.a + self.M.sum())
        return (self.a * self.prior[r] + self.T[r]) / (self.a + self.T[r].sum())

    def update(self, prev, cur):
        self.T *= self.g
        self.M *= self.g
        if prev is not None:
            self.T[prev, cur] += 1
        self.M[cur] += 1


class DriftAdaptiveOptAlg:
    """DA-OPT: supplies OPT-ALG with continuously re-estimated exit probabilities."""

    def __init__(self, cdf0, lam0, g_slow=0.998, g_fast=0.96, a_slow=5.0, a_fast=3.0,
                 disc=0.98, eta=1.0, floor=1e-3, alarm_low=0.2, alarm_high=0.6):
        pi0 = cdf_to_pi(cdf0)
        n = len(pi0)
        prior = np.zeros((n, n))
        for r in range(n):
            prior[r] = (1 - lam0[r]) * pi0
            prior[r, r] += lam0[r]
        self.experts = [_Expert(prior, pi0, 1.0, 1e12),        # E0 design model (frozen)
                        _Expert(prior, pi0, g_slow, a_slow),    # E1 slow adaptive
                        _Expert(prior, pi0, g_fast, a_fast)]    # E2 fast adaptive
        self.L = np.zeros(3)
        self.disc, self.eta, self.floor = disc, eta, floor
        self.alarm_low, self.alarm_high = alarm_low, alarm_high
        self.prev, self.t, self.alarms, self._armed = None, 0, [], True

    def weights(self):
        w = np.exp(-self.eta * (self.L - self.L.min()))
        return w / w.sum()

    def cdf(self, prev):
        """Current estimate of P(next exit stage <= i | previous exit stage = prev)."""
        w = self.weights()
        q = sum(wk * e.row(prev) for wk, e in zip(w, self.experts))
        c = np.cumsum(q)
        c[-1] = 1.0
        return c

    def update(self, cur):
        losses = np.array([-np.log(max(e.row(self.prev)[cur], self.floor)) for e in self.experts])
        self.L = self.disc * self.L + losses
        for e in self.experts:
            e.update(self.prev, cur)
        w0 = self.weights()[0]
        if w0 < self.alarm_low and self._armed:
            self.alarms.append(self.t)
            self._armed = False
        elif w0 > self.alarm_high:
            self._armed = True
        self.prev = cur
        self.t += 1


# ----------------------------------------------------------------------------
# BASELINE: re-implementation of the described "drift v1" extension
# ----------------------------------------------------------------------------
class DriftV1:
    def __init__(self, cascade, cdf0, lam0, window=50, tv_threshold=0.10, confidence=0.95, min_improvement=0.01):
        self.cascade, self.W = cascade, window
        self.tv_thr, self.conf, self.min_imp = tv_threshold, confidence, min_improvement
        self.base_cdf = np.asarray(cdf0, dtype=float)       # deployed exit probabilities
        self.ref_pi = cdf_to_pi(cdf0)                       # reference usage distribution
        self.lam0 = np.asarray(lam0, dtype=float)
        self.lam_hat = np.asarray(lam0, dtype=float).copy() # transition-based lambda estimates
        self.win_ids, self.win_prev = [], []
        self.deployments, self.t = [], 0
        self.n = len(cascade)

    def _lambda_from_window(self):
        ids = np.array(self.win_ids)
        pi = np.bincount(ids, minlength=self.n) / len(ids)
        lam = self.lam0.copy()
        for r in range(self.n):
            nxt = [c for c, p in zip(self.win_ids, self.win_prev) if p == r]
            if len(nxt) >= 5 and pi[r] < 1.0:
                stay = np.mean(np.array(nxt) == r)
                lam[r] = np.clip((stay - pi[r]) / (1 - pi[r]), 0.0, 0.99)
        return lam

    def update(self, cur, prev):
        self.win_ids.append(cur)
        self.win_prev.append(prev)
        self.t += 1
        if len(self.win_ids) < self.W:
            return
        ids = np.array(self.win_ids)
        usage = np.bincount(ids, minlength=self.n) / len(ids)
        self.lam_hat = self._lambda_from_window()                  # transition-based lambda estimation
        if 0.5 * np.abs(usage - self.ref_pi).sum() > self.tv_thr:  # window-based drift monitor
            cand_cdf = np.cumsum(usage)
            cand_cdf[-1] = 1.0
            cur_casc = new_algorithm.optimize(with_cdf(self.cascade, self.base_cdf))
            cand_casc = new_algorithm.optimize(with_cdf(self.cascade, cand_cdf))   # original OptBuild
            a = np.array([exploration.explore_FIRST(cur_casc, i)[0] for i in ids])
            b = np.array([exploration.explore_FIRST(cand_casc, i)[0] for i in ids])
            d = a - b                                              # paired differences (>0: candidate better)
            if d.std(ddof=1) > 0:
                t_stat, p_two = stats.ttest_1samp(d, 0.0)
                p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2   # paired one-sided Student t-test
            else:
                p_one = 0.0 if d.mean() > 0 else 1.0
            if p_one < (1 - self.conf) and d.mean() >= self.min_imp * a.mean():
                self.base_cdf, self.ref_pi = cand_cdf, usage       # deploy the new cascade
                self.deployments.append(self.t)
        self.win_ids, self.win_prev = [], []


# ----------------------------------------------------------------------------
# Run all strategies on ONE stream of real/synthetic exit stages
# ----------------------------------------------------------------------------
METHODS = ["FIRST", "LAST", "OPTALG", "DRIFTV1", "DAOPT", "ORACLE"]


def run_stream(cascade, ids, cdf0, lam0, oracle=None):
    """cascade: [{'id','c','p'}]   ids: exit stage of each incoming input (the observed stream)
       cdf0, lam0: DESIGN-TIME exit CDF and per-classifier lambda (what the original OPT-ALG knows)
       oracle: optional function t -> (true_cdf_t, true_lambda_t)   (synthetic only)
    Returns ({method: cost per input}, DA-OPT alarms, DRIFT-v1 deployments)."""
    L = len(ids)
    cost = {m: np.zeros(L) for m in METHODS}
    da = DriftAdaptiveOptAlg(cdf0, lam0)
    v1 = DriftV1(cascade, cdf0, lam0)
    prev = None
    for t in range(L):
        cur = int(ids[t])
        cost["FIRST"][t] = exploration.explore_FIRST(cascade, cur)[0]
        cost["LAST"][t] = exploration.explore_LAST(cascade, None if prev is None else {'id': prev}, cur)
        # original OPT-ALG: design-time p and lambda, never updated
        cost["OPTALG"][t] = cost_with_cdf(cascade, opt_alg_update(cdf0, prev, lam0), cur)
        # DRIFT-v1: deployed p (changed only after a passed t-test), windowed lambda estimates
        cost["DRIFTV1"][t] = cost_with_cdf(cascade, opt_alg_update(v1.base_cdf, prev, v1.lam_hat), cur)
        # DA-OPT (ours)
        cost["DAOPT"][t] = cost_with_cdf(cascade, da.cdf(prev), cur)
        if oracle is not None:
            true_cdf, true_lam = oracle(t)
            cost["ORACLE"][t] = cost_with_cdf(cascade, opt_alg_update(true_cdf, prev, true_lam), cur)
        da.update(cur)
        v1.update(cur, prev)
        prev = cur
    if oracle is None:
        del cost["ORACLE"]
    return cost, da.alarms, v1.deployments
