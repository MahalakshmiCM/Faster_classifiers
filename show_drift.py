"""
show_drift.py -- live demo of the drift-aware extension on ONE synthetic stream and ONE real (UCI HAR) stream.

    python show_drift.py

Prints, for each stream: where the drift happens, how the weight of the design-time model collapses
(= drift detected), when DA-OPT raises its alarm, and the average cost of every strategy
before / after the drift.
"""
import numpy as np
import drift_core, drift_sim as ds
from drift_core import DriftAdaptiveOptAlg


def demo(title, tr, seed_note):
    print("=" * 74)
    print(title)
    print("=" * 74)
    cas = tr['cascade']
    print("cascade (n=%d)  c_i = %s" % (len(cas), [int(x['c']) for x in cas]))
    print("design-time p_i = %s" % np.round(tr['cdf0'], 2).tolist())
    L, post = len(tr['ids']), tr['post_start']
    print("stream length = %d inputs | drift happens at input %s" % (L, tr['t_drift']))
    print("observed exit stages (every 40th input):", [int(v) for v in tr['ids'][::40]])

    cost, alarms, deploys = drift_core.run_stream(cas, tr['ids'], tr['cdf0'], tr['lam0'], tr['oracle'])

    # replay DA-OPT to show how the design-model weight evolves
    da = DriftAdaptiveOptAlg(tr['cdf0'], tr['lam0'])
    print("\nWeight of the DESIGN model inside DA-OPT (1.0 = stream looks like design time, ~0 = drift):")
    marks = list(range(0, L, max(L // 12, 1)))
    line = []
    for t, cur in enumerate(tr['ids']):
        if t in marks:
            line.append("t=%4d: %.2f" % (t, da.weights()[0]))
        da.update(cur)
    for i in range(0, len(line), 4):
        print("   " + " | ".join(line[i:i + 4]))
    print("\nDA-OPT drift alarms at inputs:", da.alarms[:8], "..." if len(da.alarms) > 8 else "")
    print("Drift-v1 rebuild deployments  :", deploys[:8], "..." if len(deploys) > 8 else "")

    print("\nAverage cost per input (lower is better)")
    print("%-26s %12s %12s %12s" % ("strategy", "before drift", "after drift", "overall"))
    names = {"FIRST": "FIRST", "LAST": "BOUNDARY", "OPTALG": "OPT-ALG (original)", "DRIFTV1": "Drift v1 (t-test rebuild)",
             "DAOPT": "DA-OPT (ours)", "ORACLE": "Oracle (true params)"}
    for m, c in cost.items():
        print("%-26s %12.1f %12.1f %12.1f" % (names[m], c[:post].mean(), c[post:].mean(), c.mean()))
    imp = 100 * (1 - cost["DAOPT"][post:].mean() / cost["OPTALG"][post:].mean())
    print("\n>>> after the drift DA-OPT needs %.1f%% less time per input than the original OPT-ALG\n" % imp)


def batch(domain, scenario, trials=40, seed=2024):
    """Mini experiment: average over `trials` random cascades (the full study uses 200-300)."""
    pt, ptp, _, _ = ds.run_scenario(domain, scenario, trials, seed, verbose=False)
    rows = {r[2]: r for r in ds.summarize(domain, scenario, pt, ptp)}
    print("AVERAGE OVER %d RANDOM CASCADES  (reduction of execution time vs original OPT-ALG, mean +- 95%% CI)" % trials)
    print("%-26s %20s %20s" % ("strategy", "overall", "after the drift"))
    names = {"DRIFTV1": "Drift v1 (t-test rebuild)", "DAOPT": "DA-OPT (ours)", "ORACLE": "Oracle (true params)"}
    for m in ("DRIFTV1", "DAOPT", "ORACLE"):
        if m in rows:
            r = rows[m]
            print("%-26s %+9.2f%% +-%-6.2f %+10.2f%% +-%-6.2f" % (names[m], r[6], r[7], r[8], r[9]))
    print()


def first_with_size(make, lo=5, hi=8, seed=1):
    """Take the first generated stream whose cascade has between lo and hi classifiers
    (a fixed rule that does NOT look at the results, so the demo is not cherry-picked)."""
    np.random.seed(seed)
    while True:
        tr = make()
        if lo <= len(tr['cascade']) <= hi:
            return tr


if __name__ == "__main__":
    demo("1) SYNTHETIC stream with abrupt drift (easy<->hard swap at input 500)",
         first_with_size(lambda: ds.synthetic_trial("strong")), "")
    batch("synthetic", "strong")
    demo("2) REAL stream (UCI HAR): activity regime changes from static (sit/stand/lay) to dynamic (walking)",
         first_with_size(lambda: ds.real_trial("regime")), "")
    batch("real", "regime")
    print("NOTE: these are single streams. The reliable numbers are the averages over 300 synthetic / 200 real")
    print("cascades in results_drift/drift_summary.csv (python plot_drift.py prints them).")
