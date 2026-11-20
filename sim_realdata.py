import os
import numpy as np
import exploration
import csv, math, copy, sys, time
import new_algorithm
import real_data            # <-- NEW: real-world data layer (UCI HAR)

# List of possible number of classifiers in the cascade generator
NR_CLASSIFIERS_MIN = 2
NR_CLASSIFIERS_MAX = 10

# (WCET_MIN / WCET_MAX removed: WCETs are now MEASURED from real classifiers, see real_data.py)

# Lambda exploration (single lambda case).
# On real data lambda is not a knob: it is a property of the recorded activity stream.
# We sweep it with real data by sub-sampling the real stream every `stride` windows
# (larger stride -> less temporal persistence -> smaller measured lambda);
# None = windows randomly shuffled (no temporal structure at all).
STRIDES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, None]

# Maximum number of cascades (for the multi lambda case)
NR_CASCADES = 20

# Number of runs for each cascade configuration
NR_RUNS_PER_CASCADE = 100

# Nuber of simulations
NR_SIMULATIONS = 2000

# Random seed
SEED = 12345


def optimize_cascade(cascade):
    P = []
    C = []
    ID = []
    for x in cascade:
        P.append(x['p'])
        C.append(x['c'])
        ID.append(x['id'])
    
    f       = [0] 
    prec    = [0] # "preceding classifier"
    for i  in range(1, len(P)):
        tmp = math.inf #minimum expected duration
        for j in range(i):
            if ( tmp> f[j] + (1 - P[j]) * C[i] ):
                tmp = f[j] + (1 - P[j]) * C[i]
                tmp_min = j #preceding classifier 
        f.append(tmp)
        prec.append(tmp_min)

    cascade_out = []

    ind = len(P) - 1
    while True:
        cascade_out.insert(0, {
             'c' : C[ind],
             'p' : P[ind],
             'id' : ID[ind]
            })

        if (ind == 0):
            break

        ind = prec[ind]
    
    return cascade_out


def build_real_cascade(pool, nr_classifiers=None):
    """REAL replacement of build_random_cascade().
    Picks which real classifiers form the cascade (same size range as before) and returns
    the cascade in the same data structure [{'id','c','p'}, ...] together with the real
    exit stage of every validation / test window, and the per-classifier lambda."""
    # Step 1 - Pick random number of classifiers
    if nr_classifiers == None:
        nr_classifiers = np.random.randint(NR_CLASSIFIERS_MIN, NR_CLASSIFIERS_MAX+1)

    # Step 2+3 - c (WCET) and p (exit probability) are MEASURED, not generated
    members = pool.random_members(nr_classifiers)
    cascade, exit_val, exit_test = pool.build_cascade(members)
    return cascade, exit_val, exit_test, members


def execute_simulation(cascade, lamb, stream):
    # stream = REAL sequence of 'true classifier' ids (one per input window), taken from the dataset

    prev_classifier = None  # Previously used classifier, for lambda behavior

    exec_time_FIRST = []
    exec_time_LAST = []
    exec_time_OPTBUILD = []
    exec_time_OPTALG   = []
    exec_time_vSOA0 = []
    exec_time_vSOA1 = []
    exec_time_vSOA2 = []
    exec_time_vSOA3 = []

    #optimized_cascade = optimize_cascade(cascade)
    optimized_OPTBUILD_cascade  = cascade # Will be copied and updated later
    optimized_OPTALG_cascade    = cascade # Will be copied and updated later

    # Explore the "epochs" of each input
    for i in range(0,NR_RUNS_PER_CASCADE):

        # lambda (probability to stay with the same classifier) used by OPT-ALG.
        # If it is the first round, then it's 0
        curr_lamb = 0
        if i != 0:
            curr_lamb = lamb[prev_classifier['id']]

        # REAL data: the classifier that classifies this window is the one observed
        # in the dataset (no random coin toss / random pick any more)
        curr_classifier = cascade[stream[i]]

        # Now let's execute all the algorithms

        et_FIRST, _ = exploration.explore_FIRST(cascade, curr_classifier['id'])

        et_OPTBUILD, better_cascade = exploration.explore_OA(cascade, prev_classifier, curr_classifier['id'])

        et_LAST = exploration.explore_LAST(cascade, prev_classifier, curr_classifier['id'])

        cascade_prob_update = copy.deepcopy(cascade)    # Copy the cascade to update the probabilities
        if prev_classifier != None:
            for j in range(0, len(cascade)):
                if cascade_prob_update[j]['id'] < prev_classifier['id']:
                    cascade_prob_update[j]['p'] = (1-curr_lamb) * cascade_prob_update[j]['p']
                else:
                    cascade_prob_update[j]['p'] = curr_lamb + (1-curr_lamb) * cascade_prob_update[j]['p']

        et_OPTALG, _ = exploration.explore_OA(cascade_prob_update, prev_classifier, curr_classifier['id'])

        # Execute SKIP algorithms
        et_vSOA0 = exploration.explore_vSOA(better_cascade, prev_classifier, curr_classifier['id'], 0)
        et_vSOA1 = exploration.explore_vSOA(better_cascade, prev_classifier, curr_classifier['id'], 1)
        et_vSOA2 = exploration.explore_vSOA(better_cascade, prev_classifier, curr_classifier['id'], 2)
        et_vSOA3 = exploration.explore_vSOA(better_cascade, prev_classifier, curr_classifier['id'], 3)

        exec_time_FIRST.append(et_FIRST)
        exec_time_LAST.append (et_LAST)
        exec_time_OPTBUILD.append(et_OPTBUILD)
        exec_time_OPTALG.append(et_OPTALG)
        exec_time_vSOA0.append(et_vSOA0)
        exec_time_vSOA1.append(et_vSOA1)
        exec_time_vSOA2.append(et_vSOA2)
        exec_time_vSOA3.append(et_vSOA3)

        prev_classifier = curr_classifier

    sim_result = {
        "FIRST"   : np.mean(exec_time_FIRST),
        "LAST"    : np.mean(exec_time_LAST),
        "OPTBUILD": np.mean(exec_time_OPTBUILD),
        "OPTALG"  : np.mean(exec_time_OPTALG),
        "SOA0"    : np.mean(exec_time_vSOA0),
        "SOA1"    : np.mean(exec_time_vSOA1),
        "SOA2"    : np.mean(exec_time_vSOA2),
        "SOA3"    : np.mean(exec_time_vSOA3)
    }

    return sim_result


RESULT_HEADER_COLS = ["FIRST", "LAST", "OPTBUILD", "OPTALG", "SOA0", "SOA1", "SOA2", "SOA3"]

def result_row(x, results):
    """[x, mean/std FIRST, mean/std LAST, ...]  -- identical CSV layout as the synthetic version."""
    row = [x]
    for k in RESULT_HEADER_COLS:
        row += [np.mean(results[k]), np.std(results[k])]
    return row


def sim_single_lambda(pool):
    # Real-data version of Figure 10: one (global) lambda shared by all classifiers.
    # The x axis is the MEASURED lambda of the real stream (see STRIDES above).

    filename = 'results_realdata/varying_lambda.csv'
    print("Results will be saved in:", filename)

    cascades = []
    print("Pre-generating cascades (real classifiers)...")
    for i in range(0,NR_SIMULATIONS):
        cascades.append(build_real_cascade(pool))

    print("Done. Running simulations...")

    rows = []
    settings = []
    for stride in STRIDES:
        name = "shuffled" if stride is None else "stride %d" % stride
        print("Simulating", name, "...")

        out = {k: [] for k in RESULT_HEADER_COLS}
        lambdas = []

        np.random.seed(SEED)    # Same seed for every setting, so all settings see comparable things

        # NR_SIMULATIONS different simulations on the NR_SIMULATIONS different (real) cascades
        for i in range(0,NR_SIMULATIONS):
            cascade, exit_val, exit_test, _ = cascades[i]
            _, lamb_global = real_data.estimate_lambda(exit_val, pool.subj_val, cascade, stride)
            lambdas.append(lamb_global)
            stream = real_data.strided_stream(exit_test, stride, NR_RUNS_PER_CASCADE)
            sim_results = execute_simulation(cascade, [lamb_global] * len(cascade), stream)
            for k in RESULT_HEADER_COLS:
                out[k].append(sim_results[k])

        rows.append(result_row(np.mean(lambdas), out))
        settings.append([name, np.mean(lambdas)])

    rows.sort(key=lambda r: r[0])       # increasing lambda, like the original sweep
    csv.writer(open(filename, 'w', newline='')).writerows(rows)
    csv.writer(open('results_realdata/varying_lambda_settings.csv', 'w', newline='')).writerows([["setting", "measured_lambda"]] + settings)


def sim_multiple_lambda(pool):
    # Real-data version of Figure 11: a different lambda per classifier (measured on the real
    # validation stream, stride 1) and a growing number of classifiers.

    filename = 'results_realdata/multiple_lambda.csv'
    print("Results will be saved in:", filename)
    csv_writer = csv.writer(open(filename, 'w', newline=''))

    for n_class in range(1,NR_CASCADES+1):
        print("Simulating nr. classifier ", n_class, "/", NR_CASCADES, "...")
        out = {k: [] for k in RESULT_HEADER_COLS}

        for i in range(0,NR_SIMULATIONS):
            cascade, exit_val, exit_test, _ = build_real_cascade(pool, n_class)
            lamb, _ = real_data.estimate_lambda(exit_val, pool.subj_val, cascade, 1)
            stream = real_data.strided_stream(exit_test, 1, NR_RUNS_PER_CASCADE)

            sim_results = execute_simulation(cascade, lamb, stream)
            for k in RESULT_HEADER_COLS:
                out[k].append(sim_results[k])

        csv_writer.writerow(result_row(n_class, out))


if __name__ == "__main__":
    # Usage:  python sim_realdata.py [--quick] [--sims N] [--rebuild]
    #   --quick    small run (200 cascades) to check everything works in a few minutes
    #   --sims N   number of cascades per point (paper setting: 2000)
    #   --rebuild  retrain the real classifier pool from the UCI HAR folder
    args = sys.argv[1:]
    if "--quick" in args:
        NR_SIMULATIONS = 200
    if "--sims" in args:
        NR_SIMULATIONS = int(args[args.index("--sims") + 1])

    os.makedirs('results_realdata', exist_ok=True)
    pool = real_data.load_pool(rebuild="--rebuild" in args)
    print("Real classifier pool loaded: %d classifiers, WCET source = %s, NR_SIMULATIONS = %d"
          % (pool.M, pool.wcet_source, NR_SIMULATIONS))

    np.random.seed(SEED)    # Note: seed will be reset also later to guarantee
                            # a fair comparison between the different approaches

    print("##############################################")
    print("Running single lambda simulation (Figure 10) - REAL DATA")
    print("##############################################")
    sim_single_lambda(pool)

    print("##############################################")
    print("Running multiple lambda simulation (Figure 11) - REAL DATA")
    print("##############################################")
    sim_multiple_lambda(pool)

    print("##############################################")
    print("Done! You can plot the results now.")
    print("##############################################")
