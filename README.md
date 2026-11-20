# Drift-Aware Classifier Cascades for Real-Time Sensor Streams

Extension of the IDK-cascade scheduling artifact of Agrawal et al. (*Faster Classification of Time-Series Input Streams*, ECRTS 2025) with:

1. **Extension 1 – real-world data:** every synthetic quantity (execution costs, exit probabilities, temporal dependence) is measured from the UCI HAR dataset.
2. **Extension 2 – drift-aware cascades (DA-Opt):** the original algorithm (`Opt-Alg`) is kept unchanged, but its probabilities come from a mixture of the design-time model and two adaptive transition models.

> `new_algorithm.py` (Opt-Alg) and `exploration.py` (strategies) are unchanged from the original artifact. The only change to an original file: `sim.py` got an `if __name__ == "__main__":` guard so Extension 2 can import its cascade generator.

## Repository layout

| Part | Files | Output folders |
|---|---|---|
| Original (synthetic) | `sim.py`, `plot_lambda_single.py`, `plot_lambda_multiple.py`, `overhead.py`, `plot_overhead.py` | `results/`, `outfigs/` |
| Extension 1: real data | `real_data.py`, `sim_realdata.py`, `plot_realdata_single.py`, `plot_realdata_multiple.py`, `show_realdata.py` | `results_realdata/`, `outfigs_realdata/` |
| Extension 2: drift-aware | `drift_core.py`, `drift_sim.py`, `plot_drift.py`, `show_drift.py` | `results_drift/`, `outfigs_drift/` |

Precomputed results and figures for Extensions 1 and 2 are included, so you can open the figures without running anything.

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Dataset.** Download *Human Activity Recognition Using Smartphones* from
<https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
and save the zip in the repo root as `UCI_HAR_Dataset.zip`. It is extracted automatically the first time it is needed
(alternatively extract it yourself to a folder named `UCI HAR Dataset/`, or point the `HAR_DIR` environment variable to it).
The dataset is distributed under CC BY 4.0.

## How to run

```bash
# Extension 2: drift
python show_drift.py            # ~30 s   live demo (synthetic + real drift)
python drift_sim.py --quick     # short run to check that everything works
python drift_sim.py             # ~9 min  full study (300 synthetic / 200 real cascades per scenario)
python plot_drift.py            # figures -> outfigs_drift/, prints result tables

# Extension 1: real data
python show_realdata.py         # ~30 s   prints the measured values
python sim_realdata.py          # ~7 min  full real-data simulation
python plot_realdata_single.py && python plot_realdata_multiple.py

# Original synthetic code (unchanged)
python sim.py                   # long, as in the original; then plot_lambda_single.py / plot_lambda_multiple.py
```

Run times are approximate and depend on your machine.

## Extension 1: real-world data (UCI HAR)

A pool of 20 real classifiers (logistic regression on the *k* = 2…561 most informative features) provides the WCETs `c_i` (multiply-accumulate counts).
Exit probabilities `p_i` are measured on held-out validation subjects, the sequence of "which classifier classifies the input" is the real time-ordered
test stream, and λ is estimated from it (λ is swept by sub-sampling the real stream every *s* windows).
Subject split: 15 fit / 6 validation / 9 test subjects. The full 20-stage cascade reaches 95.7 % accuracy on the test subjects.

Limitations: the assumption "if stage *j* classifies an input, every later stage does too" holds only approximately (~18.5 % of test windows violate it; kept as in the original),
and WCET is an operation count, not a certified hardware WCET. `overhead.py` is unchanged (it benchmarks up to 1000 classifiers, which a pool of 20 cannot provide).

## Extension 2: drift-aware cascade (DA-Opt)

**Problem.** Opt-Alg is designed once from design-time exit probabilities `p_i` and persistence `λ_i`. When the stream drifts (new users, a different activity regime, …)
these numbers become stale and the cascades are no longer optimal.

**Method.** Opt-Alg and OptBuild stay untouched; only the source of their probabilities changes. The probability that the next input is classified by stage *j*, given that the
previous one was classified by stage *r*, is a **mixture of three experts**: the frozen design model, a slow adaptive transition model (γ = 0.998) and a fast adaptive
transition model (γ = 0.96). Weights are exp(−discounted predictive log-loss).

* No drift → the design model wins → DA-Opt behaves like Opt-Alg.
* Drift → the design model's loss grows, weight moves to the adaptive experts within tens of inputs, and the cascade is re-optimized for the current statistics.
* Drift detection is a by-product: alarm when the design model's weight falls below 0.2 (re-armed above 0.6).

**Baseline (Drift-v1).** A window-and-t-test rebuild strategy (window monitoring of transition statistics, transition-based λ, conditional OptBuild rebuild, paired one-sided
t-test at 95 % with ≥ 1 % improvement) is implemented as `DriftV1` in `drift_core.py`. It is our re-implementation from a description and may differ from the original authors' code.

**Experiments.** Synthetic streams from the original generator (`build_random_cascade`): abrupt, strong abrupt (easy↔hard swap), gradual, recurring and λ-drift, plus a no-drift control and a severity sweep.
Real streams (UCI HAR): *new users* (test subjects in random order), *activity regime change* (static → walking, or reverse) and a design-matched control.
Metric: % execution-time reduction vs. Opt-Alg per cascade, mean ± 95 % CI over cascades.

### Results (reduction vs. Opt-Alg, whole stream)

| Domain | Scenario | Drift-v1 | **DA-Opt** | Oracle |
|---|---|---|---|---|
| synthetic | abrupt | +0.76 % | **+1.91 %** | +2.72 % |
| synthetic | strong abrupt | +6.11 % | **+6.92 %** | +7.97 % |
| synthetic | gradual | +0.39 % | **+1.66 %** | +2.49 % |
| synthetic | recurring | +0.30 % | **+1.90 %** | +2.82 % |
| synthetic | λ-drift | −0.16 % | **+1.65 %** | +2.19 % |
| synthetic | no drift (control) | −1.56 % | **−0.09 %** | 0 |
| real | new users | −4.10 % | **+1.81 %** | n/a |
| real | activity regime | −0.52 % | **+2.00 %** | n/a |
| real | design-matched (control) | −2.83 % | **+2.04 %** | n/a |

After the drift point DA-Opt recovers about 68–87 % of the gap between Opt-Alg and the oracle. It beats Opt-Alg on 54–82 % of synthetic cascades (drift scenarios) and 86–92 % of real cascades.
Gains grow with drift severity (`outfigs_drift/drift_vs_severity.pdf`). Raw numbers: `results_drift/`.

### Notes and limitations

1. Gains are modest for random drifts (~2 %): the oracle itself only reaches ~2.5–2.8 %. They are large for strong drifts (~7 %).
2. On real data, part of the gain comes from learning the real transition structure, not only from drift: the design-matched control also gains ~2 %.
3. As a change *detector*, Drift-v1's window monitor is faster on real data (98 % detected, ~83 inputs on *new users*; 90 %, ~47 on *regime*) than DA-Opt's alarm (79 %, ~588; 42 %, ~156), but it raises more false alarms (7–13 per 1000 inputs vs. ~1.7–6) and acting on them costs execution time. See `results_drift/drift_detection.csv`.
4. Same assumption as the original Opt-Alg: the exit stage of each processed input can be observed afterwards.
5. Overhead: the DA-Opt estimator costs ~35 µs per input in this Python/NumPy implementation, more than one `optimize()` call (6–19 µs). It only pays off for classifiers with millisecond-scale execution times.
6. In the synthetic study the Opt-Alg baseline gets the exact design parameters (best case for the baseline). DA-Opt's hyper-parameters were checked on separate tuning seeds (changes < 0.1 %) and are not tuned per domain.
7. The oracle exists only for synthetic streams.

## Credits and citation

* Original algorithm and artifact: K. Agrawal, S. Baruah, Z. Guo, J. Li, F. Reghenzani, K. Yang, J. Zhao, "Faster Classification of Time-Series Input Streams", ECRTS 2025, doi:10.4230/LIPIcs.ECRTS.2025.13.
* Dataset: D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones", ESANN 2013.
* Add a `LICENSE` file before publishing (check the license of the original artifact first, since this repo builds on its code).
