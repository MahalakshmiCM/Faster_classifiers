"""
real_data.py  --  REAL-WORLD DATA LAYER (UCI HAR) replacing the synthetic generator.

The original simulator (sim.py) invented everything at random:
    * the WCET c_i of every classifier              (build_random_cascade, step 2)
    * the exit probabilities p_i of every classifier (build_random_cascade, step 3)
    * which classifier "really" classifies each input (pick_classifier)
    * the temporal correlation lambda                (random coin in execute_simulation)

Here every one of those quantities is MEASURED from the UCI HAR dataset
(smartphone accelerometer/gyroscope windows, 6 human activities):

    * A pool of M=20 real classifiers is trained. Classifier m is a multinomial
      logistic regression that sees the k_m most informative features
      (k grows 2 -> 561), so a bigger classifier is costlier AND more accurate,
      exactly the cascade structure the algorithms assume.
    * c_m  = WCET of classifier m   (multiply-accumulate count, or measured time)
    * A window is "classified" by stage m if the stage confidence >= CONF_THRESHOLD.
      The last stage of a cascade always classifies (p = 1), as in the original.
    * p_m  = cumulative fraction of windows that exit at or before stage m,
             estimated on VALIDATION subjects (never seen by the classifiers).
    * The sequence of "which classifier classifies the input" is the REAL
      time-ordered stream of test windows (official UCI HAR test subjects).
    * lambda_i = persistence of the exit stage between consecutive windows,
      estimated on the validation subjects (model-consistent estimator,
      see estimate_lambda).

Data split (all by SUBJECT, so there is no subject leakage):
    train subjects (21)  ->  15 fit the classifiers, 6 are validation (p, lambda)
    official test subjects (9) -> the stream that the simulator replays
"""
import os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("HAR_DIR", os.path.join(HERE, "UCI HAR Dataset"))   # folder with train/ test/
DATA_ZIP = os.path.join(HERE, "UCI_HAR_Dataset.zip")      # the dataset zip shipped with the project
CACHE_FILE = "results_realdata/real_cascade_cache.npz"

POOL_SIZE = 20                 # size of the pool of real classifiers (= NR_CASCADES in sim.py)
K_MIN, K_MAX = 2, 561          # number of features used by smallest / largest classifier
CONF_THRESHOLD = 0.90          # classifier "classifies" a window if max softmax >= this
VAL_EVERY_NTH_SUBJECT = 4      # subjects[::4] of the training subjects become validation
NR_ACTIVITY_CLASSES = 6

# WCET model: "macs"     -> deterministic multiply-accumulate count (default, reproducible)
#             "measured" -> 99th percentile of measured single-window inference time (microseconds)
WCET_SOURCE = os.environ.get("WCET_SOURCE", "macs")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def ensure_dataset():
    """If the folder 'UCI HAR Dataset' does not exist yet, extract it automatically
    from UCI_HAR_Dataset.zip (shipped next to this file). Runs only once."""
    if os.path.exists(os.path.join(DATA_DIR, "train", "X_train.txt")):
        return
    import zipfile
    if not os.path.exists(DATA_ZIP):
        raise FileNotFoundError("Neither the folder '%s' nor the zip '%s' was found." % (DATA_DIR, DATA_ZIP))
    print("Extracting dataset from %s ..." % os.path.basename(DATA_ZIP))
    with zipfile.ZipFile(DATA_ZIP) as z:
        for name in z.namelist():
            # raw 'Inertial Signals' and macOS junk are not needed (we use the 561 provided features)
            if "Inertial Signals" in name or ".DS_Store" in name or name.startswith("__MACOSX"):
                continue
            z.extract(name, os.path.dirname(DATA_DIR))
    print("Dataset extracted to", DATA_DIR)


def _load_split(split):
    ensure_dataset()
    d = os.path.join(DATA_DIR, split)
    X = np.loadtxt(os.path.join(d, "X_%s.txt" % split))
    y = np.loadtxt(os.path.join(d, "y_%s.txt" % split), dtype=int) - 1
    s = np.loadtxt(os.path.join(d, "subject_%s.txt" % split), dtype=int)
    return X, y, s


def _measure_wcet_us(W, b, idx, n_runs=3000):
    """99th percentile of single-window inference time (us) of one stage."""
    x = np.random.RandomState(0).uniform(-1, 1, 561)
    ts = []
    for _ in range(n_runs):
        t = time.perf_counter()
        z = x[idx] @ W.T + b
        z = np.exp(z - z.max())
        z /= z.sum()
        ts.append(time.perf_counter() - t)
    return float(np.percentile(ts, 99) * 1e6)


def build_pool():
    """Train the pool of real classifiers and store everything the simulator needs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import f_classif

    print("Loading UCI HAR from '%s' ..." % DATA_DIR)
    Xtr, ytr, str_ = _load_split("train")
    Xte, yte, ste = _load_split("test")

    subjects = np.unique(str_)
    val_subjects = subjects[::VAL_EVERY_NTH_SUBJECT]
    vm = np.isin(str_, val_subjects)
    Xf, yf = Xtr[~vm], ytr[~vm]
    Xv, yv, sv = Xtr[vm], ytr[vm], str_[vm]
    print("  fit: %d windows (%d subjects) | validation: %d windows (%d subjects) | test: %d windows (%d subjects)"
          % (len(yf), len(subjects) - len(val_subjects), len(yv), len(val_subjects), len(yte), len(np.unique(ste))))

    # Feature ranking (fit data only): ANOVA F-score. Stage m uses the k_m best features.
    F, _ = f_classif(Xf, yf)
    order = np.argsort(-np.nan_to_num(F))
    ks = np.unique(np.round(np.geomspace(K_MIN, K_MAX, POOL_SIZE)).astype(int))
    assert len(ks) == POOL_SIZE, "K_MIN/K_MAX/POOL_SIZE give duplicate k values"

    conf_val, conf_test, pred_val, pred_test, cost, cost_us = [], [], [], [], [], []
    for k in ks:
        idx = order[:k]
        t = time.time()
        m = LogisticRegression(max_iter=500).fit(Xf[:, idx], yf)
        pv, pt = m.predict_proba(Xv[:, idx]), m.predict_proba(Xte[:, idx])
        conf_val.append(pv.max(1)); pred_val.append(pv.argmax(1))
        conf_test.append(pt.max(1)); pred_test.append(pt.argmax(1))
        cost.append(k * NR_ACTIVITY_CLASSES + NR_ACTIVITY_CLASSES)       # MACs of one inference
        cost_us.append(_measure_wcet_us(m.coef_, m.intercept_, idx) if WCET_SOURCE == "measured" else 0.0)
        print("  stage k=%3d features | val acc %.3f | test acc %.3f | %.1fs"
              % (k, (pred_val[-1] == yv).mean(), (pred_test[-1] == yte).mean(), time.time() - t))

    cost = np.array(cost, dtype=float)
    if WCET_SOURCE == "measured":
        cost = np.maximum.accumulate(np.array(cost_us))   # enforce c_0 <= c_1 <= ... (cascade assumption)

    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    np.savez_compressed(CACHE_FILE,
                        ks=ks, cost=cost, wcet_source=WCET_SOURCE,
                        conf_val=np.array(conf_val), conf_test=np.array(conf_test),
                        pred_val=np.array(pred_val), pred_test=np.array(pred_test),
                        y_val=yv, y_test=yte, subj_val=sv, subj_test=ste)


class RealPool:
    """The measured pool + helpers to build real cascades from it."""

    def __init__(self, cache=CACHE_FILE):
        z = np.load(cache, allow_pickle=True)
        self.ks = z["ks"]
        self.cost = z["cost"]
        self.wcet_source = str(z["wcet_source"])
        self.accept_val = z["conf_val"] >= CONF_THRESHOLD       # (M, Nval)
        self.accept_test = z["conf_test"] >= CONF_THRESHOLD     # (M, Ntest)
        self.correct_val = z["pred_val"] == z["y_val"][None, :]
        self.correct_test = z["pred_test"] == z["y_test"][None, :]
        self.subj_val = z["subj_val"]
        self.subj_test = z["subj_test"]
        self.y_val = z["y_val"]          # activity labels 0..5 (WALKING, UP, DOWN, SITTING, STANDING, LAYING)
        self.y_test = z["y_test"]
        self.M = len(self.ks)

    # -- cascade construction ------------------------------------------------
    def random_members(self, nr_classifiers):
        """Pick the stages of a cascade: always the largest classifier last (it must be
        able to classify everything), the others chosen at random from the pool.
        Returned sorted by cost, so ids are increasing in WCET like the original generator."""
        if nr_classifiers == 1:
            return [self.M - 1]
        others = np.random.choice(self.M - 1, nr_classifiers - 1, replace=False)
        return sorted(others.tolist()) + [self.M - 1]

    @staticmethod
    def _exit_ids(accept, members):
        """Index (inside the cascade) of the first stage that classifies each window.
        The last stage always classifies."""
        a = accept[members].copy()
        a[-1, :] = True
        return a.argmax(axis=0)

    def build_cascade(self, members):
        """Return (cascade, exit_val, exit_test) for a given list of pool members.
        cascade = [{'id','c','p'}, ...]   same data structure as the original generator."""
        n = len(members)
        exit_val = self._exit_ids(self.accept_val, members)
        exit_test = self._exit_ids(self.accept_test, members)
        counts = np.bincount(exit_val, minlength=n)
        p = np.cumsum(counts) / counts.sum()          # measured cumulative exit probabilities
        p[-1] = 1.0
        cascade = [{'id': i, 'c': float(self.cost[m]), 'p': float(p[i])} for i, m in enumerate(members)]
        return cascade, exit_val, exit_test

    def cascade_accuracy(self, members):
        """Accuracy of the real cascade (sanity check) on the test stream."""
        ex = self._exit_ids(self.accept_test, members)
        corr = self.correct_test[members][ex, np.arange(len(ex))]
        return float(corr.mean())

    def monotonicity_violation(self, members):
        """Fraction of test windows for which some LATER stage would NOT have accepted a window
        that an earlier stage accepted. The simulator (like the original model) assumes
        'if stage j classifies it, every later stage does too'; this measures how true that is."""
        a = self.accept_test[members].copy()
        a[-1, :] = True
        ex = a.argmax(axis=0)
        bad = 0
        for t in range(a.shape[1]):
            if not a[ex[t]:, t].all():
                bad += 1
        return bad / a.shape[1]


# ----------------------------------------------------------------------------
# Temporal persistence (lambda) estimation from real sequences
# ----------------------------------------------------------------------------
def _pairs(exit_ids, subj, stride):
    """Pairs (exit_t, exit_{t+stride}) of the real sequence, never crossing subjects.
    stride=None -> windows randomly shuffled (no temporal structure)."""
    if stride is None:
        rs = np.random.RandomState(0)
        perm = rs.permutation(len(exit_ids))
        return exit_ids[perm][:-1], exit_ids[perm][1:]
    a, b = exit_ids[:-stride], exit_ids[stride:]
    ok = subj[:-stride] == subj[stride:]
    return a[ok], b[ok]


def estimate_lambda(exit_val, subj_val, cascade, stride=1):
    """Estimate the per-classifier lambda vector and the global lambda of a cascade
    from the real VALIDATION sequence.

    The original generator says: next window keeps the same classifier with prob.
    lambda_i, otherwise it is drawn from the exit distribution pi.  Hence
        P(next = i | cur = i) = lambda_i + (1 - lambda_i) * pi_i
    and we solve it for lambda_i (clipped to [0, 0.99]).
    """
    n = len(cascade)
    p = np.array([x['p'] for x in cascade])
    pi = np.diff(np.concatenate([[0.0], p]))
    a, b = _pairs(exit_val, subj_val, stride)

    lam = np.zeros(n)
    for i in range(n):
        vis = (a == i)
        if vis.sum() == 0 or pi[i] >= 1.0:
            continue
        stay = (b[vis] == i).mean()
        lam[i] = np.clip((stay - pi[i]) / (1.0 - pi[i]), 0.0, 0.99)

    S = (a == b).mean() if len(a) else 0.0
    q = float((pi ** 2).sum())
    glob = 0.0 if q >= 1.0 else float(np.clip((S - q) / (1.0 - q), 0.0, 0.99))
    return lam.tolist(), glob


def strided_stream(exit_test, stride, nr_runs):
    """A real sequence of `nr_runs` consecutive (strided) test windows: the thing the
    simulator replays instead of tossing random numbers."""
    if stride is None:                              # shuffled windows
        seq = exit_test[np.random.permutation(len(exit_test))]
    else:
        off = np.random.randint(0, stride)
        seq = exit_test[off::stride]
    start = np.random.randint(0, len(seq) - nr_runs + 1)
    return seq[start:start + nr_runs].tolist()


def load_pool(rebuild=False):
    if rebuild or not os.path.exists(CACHE_FILE):
        build_pool()
    return RealPool()


if __name__ == "__main__":
    # Build the cache and print a summary of the real cascade.
    import csv
    build_pool()
    pool = RealPool()
    print("\nPool summary (WCET source = %s)" % pool.wcet_source)
    rows = [["stage", "nr_features", "wcet", "val_exit_rate", "test_exit_rate", "val_acc", "test_acc"]]
    for m in range(pool.M):
        rows.append([m, int(pool.ks[m]), pool.cost[m], pool.accept_val[m].mean(), pool.accept_test[m].mean(),
                     pool.correct_val[m].mean(), pool.correct_test[m].mean()])
        print(" stage %2d | k=%3d | c=%7.1f | classifies %.1f%% val / %.1f%% test | acc %.3f / %.3f"
              % tuple(rows[-1][:1] + rows[-1][1:3] + [100 * rows[-1][3], 100 * rows[-1][4], rows[-1][5], rows[-1][6]]))
    csv.writer(open("results_realdata/real_cascade_pool.csv", "w", newline="")).writerows(rows)
    full = list(range(pool.M))
    print("\nFull 20-stage cascade: accuracy on test stream = %.3f" % pool.cascade_accuracy(full))
    print("Monotonicity-assumption violations (test)        = %.3f" % pool.monotonicity_violation(full))
    cas, ev, et = pool.build_cascade(full)
    lam, glob = estimate_lambda(ev, pool.subj_val, cas, 1)
    print("Empirical global lambda (stride 1, validation)   = %.3f" % glob)
