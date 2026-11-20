"""
show_realdata.py -- prints the REAL values (from the UCI HAR dataset) that replace the synthetic ones.

Run:  python show_realdata.py
Needs the folder 'UCI HAR Dataset' next to this file (or env HAR_DIR=path).
It retrains the 20 real classifiers from the raw files, so everything you see is computed live.
"""
import numpy as np
import real_data
import sim_realdata as sim
import os

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

print("=" * 72)
print(" 1. REAL DATASET: UCI HAR (smartphone accelerometer + gyroscope)")
print("=" * 72)
Xtr, ytr, str_ = real_data._load_split("train")
Xte, yte, ste = real_data._load_split("test")
print("train windows:", Xtr.shape, "| subjects:", len(np.unique(str_)))
print("test  windows:", Xte.shape, "| subjects:", len(np.unique(ste)))
print("activities   :", LABELS)
print("first 30 real activity labels in the test stream:", [LABELS[i][:4] for i in yte[:30]])

print("\n" + "=" * 72)
print(" 2. TRAINING THE REAL CLASSIFIER POOL (replaces random c_i and p_i)")
print("=" * 72)
real_data.build_pool()
pool = real_data.RealPool()

print("\n" + "=" * 72)
print(" 3. MEASURED VALUES PER CLASSIFIER (c = cost in MACs, exit rate = measured p)")
print("=" * 72)
print("%5s %10s %10s %14s %12s %10s" % ("stage", "features", "c (WCET)", "classifies(%)", "val acc", "test acc"))
for m in range(pool.M):
    print("%5d %10d %10.0f %14.1f %12.3f %10.3f" % (
        m, pool.ks[m], pool.cost[m], 100 * pool.accept_val[m].mean(),
        pool.correct_val[m].mean(), pool.correct_test[m].mean()))

full = list(range(pool.M))
cas, ev, et = pool.build_cascade(full)
lam, glob = real_data.estimate_lambda(ev, pool.subj_val, cas, 1)
print("\nFull cascade accuracy on real test stream : %.3f" % pool.cascade_accuracy(full))
print("Measured global lambda (real persistence) : %.3f" % glob)
print("Measured p_i (cumulative exit prob.)      :", np.round([x['p'] for x in cas], 3).tolist())

print("\n" + "=" * 72)
print(" 4. ONE SIMULATION ON REAL DATA (random 6-classifier cascade, 100 real windows)")
print("=" * 72)
np.random.seed(sim.SEED)
c, ev, et, members = sim.build_real_cascade(pool, 6)
lamb, g = real_data.estimate_lambda(ev, pool.subj_val, c, 1)
stream = real_data.strided_stream(et, 1, sim.NR_RUNS_PER_CASCADE)
print("cascade uses pool classifiers :", members)
print("real c_i                      :", [x['c'] for x in c])
print("real p_i                      :", [round(x['p'], 3) for x in c])
print("real per-classifier lambda_i  :", np.round(lamb, 3).tolist())
print("real exit stages (first 30)   :", stream[:30])
res = sim.execute_simulation(c, lamb, stream)
print("\nAverage execution time (MACs) per strategy on the real stream:")
names = {"FIRST": "FIRST (no optimization)", "LAST": "BOUNDARY", "OPTBUILD": "FIRST (optimized cascade)",
         "OPTALG": "OPT-ALG (ours)", "SOA0": "SKIP(0)", "SOA1": "SKIP(1)", "SOA2": "SKIP(2)", "SOA3": "SKIP(3)"}
for k, v in res.items():
    print("  %-28s %8.1f" % (names[k], v))

print("(NOTE: this is ONE random cascade, so the ranking can vary; the averages over 2000 cascades are below.)")

print("\n" + "=" * 72)
print(" 5. FULL RESULTS (run 'python sim_realdata.py' to regenerate, then the plot scripts)")
print("=" * 72)
for f in ["results_realdata/multiple_lambda.csv"]:
    if os.path.exists(f):
        import csv
        print("Number of classifiers vs average execution time (MACs):")
        print("%4s %8s %9s %9s %8s" % ("n", "FIRST", "BOUNDARY", "OPT-ALG", "SKIP"))
        for r in csv.reader(open(f)):
            v = [float(x) for x in r]
            print("%4d %8.0f %9.0f %9.0f %8.0f" % (v[0], v[5], v[3], v[7], min(v[9], v[11], v[13], v[15])))
    else:
        print("(run 'python sim_realdata.py' first to produce", f, ")")
