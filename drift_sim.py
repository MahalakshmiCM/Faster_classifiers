"""
drift_sim.py -- EXTENSION 2 experiments: drift-aware cascades on SYNTHETIC and REAL (UCI HAR) data.

    python drift_sim.py              # full run (about 25 min)
    python drift_sim.py --quick      # small run (about 2 min) to check everything works
    python drift_sim.py --trials N   # N cascades per scenario (default 300 synthetic / 200 real)

Strategies compared on every stream (cost = execution time per input, same units as the other extensions):
    FIRST    no optimisation                      LAST    BOUNDARY heuristic
    OPTALG   original OPT-ALG (design-time p, lambda, never updated)
    DRIFTV1  re-implementation of the team's first drift extension (window + t-test gated rebuild)
    DAOPT    OUR drift-adaptive OPT-ALG
    ORACLE   OPT-ALG given the TRUE current parameters (synthetic only; an unreachable bound)
"""
import os, sys, csv
import numpy as np
import drift_core
from drift_core import METHODS, run_stream, cdf_to_pi

SEED = 12345
L_SYN = 1000                  # stream length, synthetic scenarios
SEG_REAL = 600                # segment length (regime scenario) for real data
OUT = "results_drift"


# ----------------------------------------------------------------------------
# SYNTHETIC streams (cascade generator = the ORIGINAL build_random_cascade of sim.py)
# ----------------------------------------------------------------------------
def _gen_stream(pis, lams, L):
    """Same generative model as the original sim.py: stay with prob lambda[prev], else draw from p."""
    ids, prev = [], None
    for t in range(L):
        pi, lam = pis(t), lams(t)
        if prev is not None and np.random.random() < lam[prev]:
            cur = prev
        else:
            cur = int(np.random.choice(len(pi), p=pi))
        ids.append(cur)
        prev = cur
    return ids


def _random_pi(n):
    """Exit distribution drawn with the original generator's recipe (Step 3 of build_random_cascade)."""
    prev_p, cdf = 0.0, []
    for _ in range(n - 1):
        prev_p += (1 - prev_p) * np.random.random()
        cdf.append(prev_p)
    cdf.append(1.0)
    return cdf_to_pi(cdf)


def synthetic_trial(scenario, L=L_SYN, severity=1.0):
    from sim import build_random_cascade            # ORIGINAL synthetic generator, untouched
    cascade = build_random_cascade()
    n = len(cascade)
    cdf0 = np.array([x['p'] for x in cascade])
    pi0 = cdf_to_pi(cdf0)
    lam0 = np.random.rand(n)                        # per-classifier lambda, as in the original multi-lambda experiment
    pi1 = _random_pi(n)
    lam1 = np.random.rand(n)
    rev = pi0[::-1].copy()
    lams = lambda t: lam0
    if scenario == "abrupt":                        # new random exit distribution at L/2
        pis = lambda t: pi0 if t < L // 2 else pi1
        t_drift = L // 2
    elif scenario == "strong":                      # easy <-> hard swap at L/2
        pis = lambda t: pi0 if t < L // 2 else rev
        t_drift = L // 2
    elif scenario == "severity":                    # partial swap, strength = severity
        mix = (1 - severity) * pi0 + severity * rev
        pis = lambda t: pi0 if t < L // 2 else mix
        t_drift = L // 2
    elif scenario == "gradual":                     # linear change between L/3 and 2L/3
        a, b = L // 3, 2 * L // 3
        pis = lambda t: pi0 if t < a else (pi1 if t >= b else pi0 + (pi1 - pi0) * (t - a) / (b - a))
        t_drift = a
    elif scenario == "recurring":                   # p0, p1, p0, p1 in blocks of L/4
        pis = lambda t: pi0 if (t // (L // 4)) % 2 == 0 else pi1
        t_drift = L // 4
    elif scenario == "lambda":                      # persistence of the stream changes at L/2
        pis = lambda t: pi0
        lams = lambda t: lam0 if t < L // 2 else lam1
        t_drift = L // 2
    elif scenario == "none":                        # control: stationary stream
        pis = lambda t: pi0
        t_drift = None
    else:
        raise ValueError(scenario)
    ids = _gen_stream(pis, lams, L)
    oracle = lambda t: (np.cumsum(pis(t)), lams(t))
    return dict(cascade=cascade, ids=ids, cdf0=cdf0, lam0=lam0, oracle=oracle,
                t_drift=t_drift, post_start=(t_drift if t_drift is not None else L // 2))


# ----------------------------------------------------------------------------
# REAL streams (UCI HAR): cascades made of the real classifiers of extension 1
# ----------------------------------------------------------------------------
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        import real_data
        _POOL = real_data.load_pool()
    return _POOL


def real_trial(scenario):
    import real_data
    pool = _pool()
    n = np.random.randint(2, 11)
    members = pool.random_members(n)
    cascade, exit_val, exit_test = pool.build_cascade(members)
    cdf0 = np.array([x['p'] for x in cascade])                       # design-time p  (validation subjects)
    lam0, _ = real_data.estimate_lambda(exit_val, pool.subj_val, cascade, 1)   # design-time lambda
    if scenario == "subject":        # deployment on NEW people: the 9 test subjects, in random order
        subs = np.random.permutation(np.unique(pool.subj_test))
        blocks = [exit_test[pool.subj_test == s] for s in subs]
        ids = np.concatenate(blocks)
        post = len(blocks[0])
        t_drift = post
    elif scenario == "regime":       # activity regime changes: static (sit/stand/lay) <-> dynamic (walking)
        static = np.flatnonzero(np.isin(pool.y_test, [3, 4, 5]))
        dynamic = np.flatnonzero(np.isin(pool.y_test, [0, 1, 2]))
        segs = []
        for idx in (static, dynamic):
            s = np.random.randint(0, len(idx) - SEG_REAL + 1)
            segs.append(exit_test[idx[s:s + SEG_REAL]])
        if np.random.random() < 0.5:
            segs = segs[::-1]
        ids = np.concatenate(segs)
        post = SEG_REAL
        t_drift = post
    elif scenario == "none":         # control: deployment = design (validation subjects, same people)
        subs = np.random.permutation(np.unique(pool.subj_val))
        ids = np.concatenate([exit_val[pool.subj_val == s] for s in subs])
        post = len(ids) // 2
        t_drift = None
    else:
        raise ValueError(scenario)
    return dict(cascade=cascade, ids=ids, cdf0=cdf0, lam0=np.array(lam0), oracle=None,
                t_drift=t_drift, post_start=post)


# ----------------------------------------------------------------------------
# Experiment loop + statistics
# ----------------------------------------------------------------------------
def run_scenario(domain, scenario, trials, seed, severity=1.0, verbose=True):
    np.random.seed(seed)
    per_trial = {m: [] for m in METHODS}
    per_trial_post = {m: [] for m in METHODS}
    series_sum, series_n = {}, 0
    det = dict(da_delay=[], da_false=[], v1_delay=[], v1_false=[])
    for k in range(trials):
        tr = synthetic_trial(scenario, severity=severity) if domain == "synthetic" else real_trial(scenario)
        cost, alarms, deploys = run_stream(tr['cascade'], tr['ids'], tr['cdf0'], tr['lam0'], tr['oracle'])
        base = cost["OPTALG"].mean()
        for m in cost:
            per_trial[m].append(cost[m].mean())
            per_trial_post[m].append(cost[m][tr['post_start']:].mean())
            series_sum[m] = series_sum.get(m, 0) + cost[m] / base
        series_n += 1
        td, L = tr['t_drift'], len(tr['ids'])
        for name, ev in (("da", alarms), ("v1", deploys)):
            if td is None:
                det[name + "_false"].append(len(ev) / L * 1000)
            else:
                pre = [e for e in ev if e < td]
                post = [e for e in ev if e >= td]
                det[name + "_false"].append(len(pre) / td * 1000)
                det[name + "_delay"].append(post[0] - td if post else np.nan)
        if verbose and (k + 1) % 50 == 0:
            print("    %s/%s: %d/%d trials" % (domain, scenario, k + 1, trials), flush=True)
    return per_trial, per_trial_post, {m: v / series_n for m, v in series_sum.items()}, det


def summarize(domain, scenario, per_trial, per_trial_post):
    rows = []
    base, base_post = np.array(per_trial["OPTALG"]), np.array(per_trial_post["OPTALG"])
    for m in per_trial:
        if len(per_trial[m]) == 0:           # ORACLE does not exist on real data
            continue
        c, cp = np.array(per_trial[m]), np.array(per_trial_post[m])
        imp = 100 * (1 - c / base)                 # % execution-time reduction vs original OPT-ALG (per cascade)
        imp_post = 100 * (1 - cp / base_post)
        ci = 1.96 * imp.std(ddof=1) / np.sqrt(len(imp)) if len(imp) > 1 else 0.0
        ci_post = 1.96 * imp_post.std(ddof=1) / np.sqrt(len(imp)) if len(imp) > 1 else 0.0
        rows.append([domain, scenario, m, len(c), c.mean(), cp.mean(), imp.mean(), ci, imp_post.mean(), ci_post,
                     float((c < base - 1e-9).mean())])
    return rows


SYN_SCENARIOS = ["abrupt", "strong", "gradual", "recurring", "lambda", "none"]
REAL_SCENARIOS = ["subject", "regime", "none"]
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    args = sys.argv[1:]
    quick = "--quick" in args
    n_syn = 30 if quick else 300
    n_real = 20 if quick else 200
    n_sev = 20 if quick else 150
    if "--trials" in args:
        n_syn = n_real = n_sev = int(args[args.index("--trials") + 1])
    os.makedirs(OUT, exist_ok=True)

    summary = [["domain", "scenario", "method", "n_trials", "mean_cost", "mean_cost_after_drift",
                "improve_vs_OPTALG_pct", "ci95", "improve_after_drift_pct", "ci95_after_drift", "win_rate"]]
    detection = [["domain", "scenario", "detector", "detected_pct", "mean_delay_inputs", "false_alarms_per_1000"]]

    jobs = [("synthetic", s, n_syn) for s in SYN_SCENARIOS] + [("real", s, n_real) for s in REAL_SCENARIOS]
    for j, (domain, scen, trials) in enumerate(jobs):
        print("[%d/%d] %s / %s (%d cascades)" % (j + 1, len(jobs), domain, scen, trials), flush=True)
        pt, ptp, series, det = run_scenario(domain, scen, trials, SEED + 1000 * j)
        summary += summarize(domain, scen, pt, ptp)
        for name, label in (("da", "DA-OPT"), ("v1", "DRIFT-v1")):
            d = np.array(det[name + "_delay"], dtype=float)
            detection.append([domain, scen, label,
                              "" if scen == "none" else 100 * np.mean(~np.isnan(d)),
                              "" if scen == "none" or np.all(np.isnan(d)) else np.nanmean(d),
                              np.mean(det[name + "_false"])])
        # smoothed time series (moving average, window 25) of cost relative to OPT-ALG
        w = 25
        keys = list(series.keys())
        sm = {m: np.convolve(series[m], np.ones(w) / w, mode="valid") for m in keys}
        with open("%s/timeseries_%s_%s.csv" % (OUT, domain, scen), "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["t"] + keys)
            for t in range(len(sm[keys[0]])):
                cw.writerow([t + w - 1] + [sm[m][t] for m in keys])
        for r in summary[-sum(len(v) > 0 for v in pt.values()):]:
            if r[2] in ("OPTALG", "DRIFTV1", "DAOPT", "ORACLE"):
                print("    %-8s cost %.1f | vs OPT-ALG: %+.2f%% (+-%.2f) | after drift %+.2f%%"
                      % (r[2], r[4], r[6], r[7], r[8]), flush=True)

    # severity sweep (synthetic): how much does DA-OPT gain as the drift gets stronger?
    sev_rows = [["severity", "method", "improve_vs_OPTALG_pct", "ci95"]]
    for j, s in enumerate(SEVERITIES):
        print("[severity %.2f] (%d cascades)" % (s, n_sev), flush=True)
        pt, ptp, _, _ = run_scenario("synthetic", "severity", n_sev, SEED + 50000 + j, severity=s, verbose=False)
        for r in summarize("synthetic", "severity", pt, ptp):
            if r[2] in ("DRIFTV1", "DAOPT", "ORACLE"):
                sev_rows.append([s, r[2], r[6], r[7]])
    csv.writer(open(OUT + "/drift_summary.csv", "w", newline="")).writerows(summary)
    csv.writer(open(OUT + "/drift_detection.csv", "w", newline="")).writerows(detection)
    csv.writer(open(OUT + "/drift_severity.csv", "w", newline="")).writerows(sev_rows)
    print("Done. Results in", OUT)


if __name__ == "__main__":
    main()
