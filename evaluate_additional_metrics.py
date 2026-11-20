"""
evaluate_additional_metrics.py
Additional evaluation for the DA-OPT drift-aware cascade on UCI HAR.

Place this file in the project root (next to drift_core.py and real_data.py), then run:
    python evaluate_additional_metrics.py

Optional operation-count deadline (NOT milliseconds):
    python evaluate_additional_metrics.py --deadline-macs 100000

Important:
- Accuracy is measured using the cached real UCI HAR predictions of the classifier
  selected by each strategy on each test window.
- The cascade-selection logic follows the same monotonic-stage assumption as
  drift_core.run_stream / the original simulator.
- "classifier_cost_macs" is the project's operation-count cost, NOT wall-clock time.
- "decision_wall_us" is measured Python time for the strategy's selection/re-optimization
  step. It is NOT full sensor-to-answer response time.
- "update_wall_us" measures the adaptive estimator / Drift-v1 update separately.
- Deadline misses are reported only when --deadline-macs is explicitly supplied.
  Such deadlines are operation-count budgets, not certified real-time deadlines.
"""
import argparse
import csv
import os
import time
import numpy as np

import real_data
import drift_core
import exploration
import new_algorithm


def choose_stage(optimized_cascade, true_exit_stage):
    """Return selected original stage ID and cumulative classifier cost."""
    total = 0.0
    for stage in optimized_cascade:
        total += float(stage["c"])
        if int(stage["id"]) >= int(true_exit_stage):
            return int(stage["id"]), total
    raise RuntimeError("No selected classifier can classify this test window")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-macs", type=float, default=None,
                    help="Optional per-input operation-count budget; not a time deadline.")
    ap.add_argument("--output-dir", default="results_additional")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pool = real_data.load_pool()
    members = list(range(pool.M))
    cascade, _, exit_test = pool.build_cascade(members)
    cdf0 = np.array([x["p"] for x in cascade], dtype=float)

    # Design-time persistence is estimated only from validation subjects.
    # The test stream is not used to initialise the design parameters.
    exit_val, _ = pool._exit_ids(pool.accept_val, members), None
    lam0, _ = real_data.estimate_lambda(exit_val, pool.subj_val, cascade, 1)
    lam0 = np.asarray(lam0, dtype=float)

    # Preserve the official test order. This is the recorded real-world stream.
    methods = ("OPTALG", "DRIFTV1", "DAOPT")
    rows = []
    correct = {m: 0 for m in methods}
    total_cost = {m: 0.0 for m in methods}
    decision_ns = {m: [] for m in methods}
    update_ns = {"DRIFTV1": [], "DAOPT": []}
    deadline_misses = {m: 0 for m in methods}
    da = drift_core.DriftAdaptiveOptAlg(cdf0, lam0)
    v1 = drift_core.DriftV1(cascade, cdf0, lam0)
    prev = None

    for t, true_stage in enumerate(exit_test):
        true_stage = int(true_stage)
        sample_correct = {}
        sample_cost = {}
        sample_selected = {}

        # Original OPT-ALG: fixed design probabilities.
        t0 = time.perf_counter_ns()
        cdf = drift_core.opt_alg_update(cdf0, prev, lam0)
        opt_cascade = new_algorithm.optimize(drift_core.with_cdf(cascade, cdf))
        selected, cost = choose_stage(opt_cascade, true_stage)
        decision_ns["OPTALG"].append(time.perf_counter_ns() - t0)
        sample_selected["OPTALG"], sample_cost["OPTALG"] = selected, cost

        # Drift v1: use its currently deployed CDF and lambda estimate.
        t0 = time.perf_counter_ns()
        cdf = drift_core.opt_alg_update(v1.base_cdf, prev, v1.lam_hat)
        v1_cascade = new_algorithm.optimize(drift_core.with_cdf(cascade, cdf))
        selected, cost = choose_stage(v1_cascade, true_stage)
        decision_ns["DRIFTV1"].append(time.perf_counter_ns() - t0)
        sample_selected["DRIFTV1"], sample_cost["DRIFTV1"] = selected, cost

        # DA-OPT: mixture-estimated transition probabilities.
        t0 = time.perf_counter_ns()
        cdf = da.cdf(prev)
        da_cascade = new_algorithm.optimize(drift_core.with_cdf(cascade, cdf))
        selected, cost = choose_stage(da_cascade, true_stage)
        decision_ns["DAOPT"].append(time.perf_counter_ns() - t0)
        sample_selected["DAOPT"], sample_cost["DAOPT"] = selected, cost

        # The chosen cascade stage ID maps to the same-index member in this full pool.
        for method in methods:
            member_id = members[sample_selected[method]]
            is_correct = bool(pool.correct_test[member_id, t])
            sample_correct[method] = is_correct
            correct[method] += int(is_correct)
            total_cost[method] += sample_cost[method]
            if args.deadline_macs is not None and sample_cost[method] > args.deadline_macs:
                deadline_misses[method] += 1

        # Updates happen after this input, matching drift_core.run_stream().
        t0 = time.perf_counter_ns()
        da.update(true_stage)
        update_ns["DAOPT"].append(time.perf_counter_ns() - t0)

        t0 = time.perf_counter_ns()
        v1.update(true_stage, prev)
        update_ns["DRIFTV1"].append(time.perf_counter_ns() - t0)
        prev = true_stage

        for method in methods:
            rows.append({
                "input_index": t,
                "method": method,
                "selected_stage_id": sample_selected[method],
                "classifier_cost_macs": sample_cost[method],
                "prediction_correct": int(sample_correct[method]),
                "decision_wall_us": decision_ns[method][-1] / 1000.0,
                "update_wall_us": (update_ns[method][-1] / 1000.0
                                   if method in update_ns else 0.0),
                "deadline_missed": (
                    int(sample_cost[method] > args.deadline_macs)
                    if args.deadline_macs is not None else ""
                ),
            })

    per_input = os.path.join(args.output_dir, "per_input_metrics.csv")
    with open(per_input, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(args.output_dir, "metrics_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "method", "test_inputs", "accuracy_pct", "mean_classifier_cost_macs",
            "p95_classifier_cost_macs", "mean_decision_wall_us",
            "p95_decision_wall_us", "mean_update_wall_us",
            "p95_update_wall_us", "deadline_miss_ratio_pct",
            "deadline_budget_macs"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            costs = np.array([r["classifier_cost_macs"] for r in rows if r["method"] == method])
            decisions = np.array(decision_ns[method], dtype=float) / 1000.0
            updates = (np.array(update_ns[method], dtype=float) / 1000.0
                       if method in update_ns else np.zeros(len(exit_test)))
            miss_pct = (100.0 * deadline_misses[method] / len(exit_test)
                        if args.deadline_macs is not None else "")
            writer.writerow({
                "method": method,
                "test_inputs": len(exit_test),
                "accuracy_pct": 100.0 * correct[method] / len(exit_test),
                "mean_classifier_cost_macs": float(costs.mean()),
                "p95_classifier_cost_macs": float(np.percentile(costs, 95)),
                "mean_decision_wall_us": float(decisions.mean()),
                "p95_decision_wall_us": float(np.percentile(decisions, 95)),
                "mean_update_wall_us": float(updates.mean()),
                "p95_update_wall_us": float(np.percentile(updates, 95)),
                "deadline_miss_ratio_pct": miss_pct,
                "deadline_budget_macs": args.deadline_macs if args.deadline_macs is not None else ""
            })

    print("\nUCI HAR additional evaluation")
    print("Test inputs:", len(exit_test))
    print("Accuracy is selected-classifier accuracy under the simulator's stage-selection assumption.")
    print("Classifier cost is MAC operation count, not milliseconds.")
    print("Decision/update wall time is Python overhead, not end-to-end inference latency.")
    print("\n{:<10} {:>10} {:>15} {:>17} {:>17}".format(
        "Method", "Accuracy", "Mean cost (MACs)", "Decision (us)", "Update (us)"))
    for method in methods:
        costs = np.array([r["classifier_cost_macs"] for r in rows if r["method"] == method])
        dec = np.array(decision_ns[method], dtype=float) / 1000.0
        upd = np.array(update_ns[method], dtype=float) / 1000.0 if method in update_ns else np.zeros(1)
        print("{:<10} {:>9.2f}% {:>15.1f} {:>17.2f} {:>17.2f}".format(
            method, 100.0 * correct[method] / len(exit_test),
            costs.mean(), dec.mean(), upd.mean()))

    if args.deadline_macs is None:
        print("\nDeadline miss ratio: not calculated (no --deadline-macs budget supplied).")
    else:
        print("\nOperation-budget deadline miss ratio (budget = %.1f MACs):" % args.deadline_macs)
        for method in methods:
            print("  %-8s %.2f%%" % (method, 100.0 * deadline_misses[method] / len(exit_test)))

    print("\nSaved:")
    print(" ", per_input)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
