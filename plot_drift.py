"""
plot_drift.py -- figures of the drift-aware extension (run after drift_sim.py).
Writes PDFs to outfigs_drift/ and prints the result tables.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN, OUT = "results_drift", "outfigs_drift"
os.makedirs(OUT, exist_ok=True)
LABEL = {"OPTALG": "OPT-ALG (original)", "DRIFTV1": "Drift v1 (t-test rebuild)", "DAOPT": "DA-OPT (ours)", "ORACLE": "Oracle (true params)"}
COLOR = {"OPTALG": "#7f7f7f", "DRIFTV1": "#d97706", "DAOPT": "#15803d", "ORACLE": "#1d4ed8"}
STYLE = {"OPTALG": "-", "DRIFTV1": "--", "DAOPT": "-", "ORACLE": ":"}
NICE = {"abrupt": "abrupt", "strong": "strong\nabrupt", "gradual": "gradual", "recurring": "recurring", "lambda": "persistence\n($\\lambda$) drift",
        "none": "no drift\n(control)", "subject": "new users\n(subject drift)", "regime": "activity regime\nchange"}

rows = list(csv.DictReader(open(IN + "/drift_summary.csv")))
get = lambda dom, scen, m, key: float([r for r in rows if r["domain"] == dom and r["scenario"] == scen and r["method"] == m][0][key])


def bars(ax, dom, scens, methods, key="improve_vs_OPTALG_pct", cikey="ci95"):
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        v = [get(dom, s, m, key) for s in scens]
        e = [get(dom, s, m, cikey) for s in scens]
        ax.bar(np.arange(len(scens)) + (j - (len(methods) - 1) / 2) * w, v, w, yerr=e, capsize=2,
               label=LABEL[m], color=COLOR[m])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(scens)))
    ax.set_xticklabels([NICE[s] for s in scens], fontsize=9)
    ax.grid(axis="y", alpha=0.3)


# ---- Figure 1: improvement over original OPT-ALG, synthetic + real ----
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [6, 3]})
bars(axs[0], "synthetic", ["abrupt", "strong", "gradual", "recurring", "lambda", "none"], ["DRIFTV1", "DAOPT", "ORACLE"])
axs[0].set_title("Synthetic data (original cascade generator)")
axs[0].set_ylabel("execution-time reduction vs OPT-ALG (%)")
axs[0].legend(fontsize=8, loc="upper left")
bars(axs[1], "real", ["subject", "regime", "none"], ["DRIFTV1", "DAOPT"])
axs[1].set_title("Real data (UCI HAR)")
plt.tight_layout()
plt.savefig(OUT + "/drift_improvement.pdf", bbox_inches="tight")
plt.close()

# ---- Figure 2: adaptation curves ----
def curve(ax, dom, scen, title, drift_at, methods):
    """cost of every method RELATIVE to original OPT-ALG at the same input (OPT-ALG = 1.0)."""
    r = list(csv.DictReader(open("%s/timeseries_%s_%s.csv" % (IN, dom, scen))))
    t = [int(x["t"]) for x in r]
    base = np.array([float(x["OPTALG"]) for x in r])
    for m in methods:
        if m in r[0]:
            ax.plot(t, np.array([float(x[m]) for x in r]) / base, STYLE[m], color=COLOR[m], label=LABEL[m], lw=1.8)
    if drift_at is not None:
        ax.axvline(drift_at, color="red", lw=1, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("input number")
    ax.grid(alpha=0.3)

fig, axs = plt.subplots(2, 3, figsize=(14, 7))
M = ["DRIFTV1", "DAOPT", "ORACLE"]
curve(axs[0, 0], "synthetic", "abrupt", "synthetic: abrupt drift", 500, M)
curve(axs[0, 1], "synthetic", "strong", "synthetic: strong abrupt drift", 500, M)
curve(axs[0, 2], "synthetic", "gradual", "synthetic: gradual drift", 333, M)
curve(axs[1, 0], "synthetic", "recurring", "synthetic: recurring drift", 250, M)
curve(axs[1, 1], "real", "regime", "real (UCI HAR): activity regime change", 600, M)
curve(axs[1, 2], "real", "subject", "real (UCI HAR): new users", None, M)
axs[0, 0].set_ylabel("cost relative to OPT-ALG (1.0 = same)")
axs[1, 0].set_ylabel("cost relative to OPT-ALG (1.0 = same)")
axs[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "/drift_adaptation_curves.pdf", bbox_inches="tight")
plt.close()

# ---- Figure 3: gain vs drift severity ----
sev = list(csv.DictReader(open(IN + "/drift_severity.csv")))
fig, ax = plt.subplots(figsize=(6, 4))
for m in ["DRIFTV1", "DAOPT", "ORACLE"]:
    x = [float(r["severity"]) for r in sev if r["method"] == m]
    y = [float(r["improve_vs_OPTALG_pct"]) for r in sev if r["method"] == m]
    e = [float(r["ci95"]) for r in sev if r["method"] == m]
    ax.errorbar(x, y, yerr=e, fmt="o" + STYLE[m] if STYLE[m] != ":" else "o:", color=COLOR[m], label=LABEL[m], capsize=3)
ax.set_xlabel("drift severity (0 = none, 1 = easy$\\leftrightarrow$hard swap)")
ax.set_ylabel("reduction vs OPT-ALG (%)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "/drift_vs_severity.pdf", bbox_inches="tight")
plt.close()

# ---- tables ----
print("\nExecution-time reduction vs original OPT-ALG (%, mean +- 95%% CI over cascades; after-drift = stream part after the change)")
print("%-10s %-10s %-10s %14s %18s %9s" % ("domain", "scenario", "method", "overall", "after drift", "win rate"))
for r in rows:
    if r["method"] in ("DRIFTV1", "DAOPT", "ORACLE"):
        print("%-10s %-10s %-10s %+7.2f +-%-5.2f %+9.2f +-%-5.2f %8.0f%%" % (
            r["domain"], r["scenario"], r["method"], float(r["improve_vs_OPTALG_pct"]), float(r["ci95"]),
            float(r["improve_after_drift_pct"]), float(r["ci95_after_drift"]), 100 * float(r["win_rate"])))
print("\nFigures saved in", OUT)
