import numpy as np
import exploration
import csv, math, copy
import new_algorithm

# List of possible number of classifiers in the cascade generator
NR_CLASSIFIERS_MIN = 2
NR_CLASSIFIERS_MAX = 10

# WCET min / max in the cascade generator
WCET_MIN = 10
WCET_MAX = 100

# Lambda exploration parameters (for the single lambda case)
LAMBDA_MIN=0
LAMBDA_MAX=0.95
LAMBDA_STEP=0.05

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


def build_random_cascade(nr_classifiers=None):
    # Step 1 - Pick random number of classifiers
    if nr_classifiers == None:
        nr_classifiers = np.random.randint(NR_CLASSIFIERS_MIN, NR_CLASSIFIERS_MAX+1)

    result = []

    # Step 2 - Generate WCET vector
    prev_c = 0
    for i in range(0,nr_classifiers):
        prev_c = prev_c + np.random.randint(WCET_MIN, WCET_MAX+1)
        c = {'id': i, 
             'c' : prev_c,
             'p' : None
            }
        result.append(c)

    # Step 3 - Generate probability vector
    prev_p = 0
    for i in range(0,nr_classifiers-1):
        prev_p += (1-prev_p)*np.random.random()
        result[i]['p'] = prev_p

    result[nr_classifiers-1]['p'] = 1

    return result


def pick_classifier(cascade):       # Pick, randomly, the classifier that classify the data
    rand_nr = np.random.random()
    for x in cascade:
        if rand_nr <= x['p']:
            return x

    assert(False)   # This should not occur, it means that the last classifier has not 1 as probability

def execute_simulation(cascade, lamb):

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

        # Set the probability to stay with the same classifier.
        # If it is the first round, then it's 0
        curr_lamb = 0
        if i != 0:
            curr_lamb = lamb[prev_classifier['id']]
        
        # Toss the lambda biased coin
        keep_the_same = True if np.random.random() < curr_lamb else False

        if keep_the_same:
            curr_classifier = prev_classifier
        else:
            curr_classifier = pick_classifier(cascade)

        # Now let's execute all the algorithms

        et_FIRST, _ = exploration.explore_FIRST(cascade, curr_classifier['id'])

        et_OPTBUILD, better_cascade = exploration.explore_OA(cascade, prev_classifier, curr_classifier['id'])

        et_LAST = exploration.explore_LAST(cascade, prev_classifier, curr_classifier['id'])

        cascade_prob_update = copy.deepcopy(cascade)    # Copy the cascade to update the probabilities
        if prev_classifier != None:
            for i in range(0, len(cascade)):
                if cascade_prob_update[i]['id'] < prev_classifier['id']:
                    cascade_prob_update[i]['p'] = (1-curr_lamb) * cascade_prob_update[i]['p']
                else:
                    cascade_prob_update[i]['p'] = curr_lamb + (1-curr_lamb) * cascade_prob_update[i]['p']

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


def sim_single_lambda():

    filename = 'results/varying_lambda.csv'
    print("Results will be saved in:", filename)
    csv_writer = csv.writer(open(filename, 'w', newline=''))

    cascades = []
    print("Pre-generating cascades...")
    for i in range(0,NR_SIMULATIONS):
        cascades.append(build_random_cascade())

    print("Done. Running simulations...")

    LAMBDA_CURR = LAMBDA_MIN

    while LAMBDA_CURR <= LAMBDA_MAX:
        print("Simulating lambda ", LAMBDA_CURR, "/", LAMBDA_MAX, "...")

        out_FIRST  = []
        out_LAST   = []
        out_OPTBUILD = []
        out_OPTALG = []
        out_vSOA0= []
        out_vSOA1= []
        out_vSOA2= []
        out_vSOA3= []
    
        np.random.seed(SEED)    # Let's reset here the seed so that all lambdas have the same things happening

        # For we perform NR_SIMULATIONS different simulations on the NR_SIMULATIONS different cascades
        for i in range(0,NR_SIMULATIONS):
            cascade = cascades[i]
            sim_results = execute_simulation(cascade, [LAMBDA_CURR] * len(cascade))
            out_FIRST.append(sim_results['FIRST'])
            out_LAST.append(sim_results['LAST'])
            out_OPTBUILD.append(sim_results['OPTBUILD'])
            out_OPTALG.append(sim_results['OPTALG'])
            out_vSOA0.append(sim_results['SOA0'])
            out_vSOA1.append(sim_results['SOA1'])
            out_vSOA2.append(sim_results['SOA2'])
            out_vSOA3.append(sim_results['SOA3'])
        
        csv_writer.writerow([LAMBDA_CURR,
                             np.mean(out_FIRST),
                             np.std(out_FIRST),
                             np.mean(out_LAST),
                             np.std(out_LAST),
                             np.mean(out_OPTBUILD),
                             np.std(out_OPTBUILD),
                             np.mean(out_OPTALG),
                             np.std(out_OPTALG),
                             np.mean(out_vSOA0),
                             np.std(out_vSOA0),
                             np.mean(out_vSOA1),
                             np.std(out_vSOA1),
                             np.mean(out_vSOA2),
                             np.std(out_vSOA2),
                             np.mean(out_vSOA3),
                             np.std(out_vSOA3)
                             ])

        LAMBDA_CURR = math.floor((LAMBDA_CURR + LAMBDA_STEP)*100)/100    # Math ceil just to avoid rounding errors


def sim_multiple_lambda():

    filename = 'results/multiple_lambda.csv'
    print("Results will be saved in:", filename)
    csv_writer = csv.writer(open(filename, 'w', newline=''))

    for n_class in range(1,NR_CASCADES+1):
        print("Simulating nr. classifier ", n_class, "/", NR_CASCADES, "...")
        out_FIRST  = []
        out_LAST  = []
        out_OPTBUILD  = []
        out_OPTALG  = []
        out_vSOA0= []
        out_vSOA1= []
        out_vSOA2= []
        out_vSOA3= []

        for i in range(0,NR_SIMULATIONS):
            cascade = build_random_cascade(n_class)
            lamb = np.random.rand(len(cascade))
            
            sim_results = execute_simulation(cascade, lamb)
            out_FIRST.append(sim_results['FIRST'])
            out_LAST.append(sim_results['LAST'])
            out_OPTBUILD.append(sim_results['OPTBUILD'])
            out_OPTALG.append(sim_results['OPTALG'])
            out_vSOA0.append(sim_results['SOA0'])
            out_vSOA1.append(sim_results['SOA1'])
            out_vSOA2.append(sim_results['SOA2'])
            out_vSOA3.append(sim_results['SOA3'])

        csv_writer.writerow([n_class,
                             np.mean(out_FIRST),
                             np.std(out_FIRST),
                             np.mean(out_LAST),
                             np.std(out_LAST),
                             np.mean(out_OPTBUILD),
                             np.std(out_OPTBUILD),
                             np.mean(out_OPTALG),
                             np.std(out_OPTALG),
                             np.mean(out_vSOA0),
                             np.std(out_vSOA0),
                             np.mean(out_vSOA1),
                             np.std(out_vSOA1),
                             np.mean(out_vSOA2),
                             np.std(out_vSOA2),
                             np.mean(out_vSOA3),
                             np.std(out_vSOA3)
                             ])

if __name__ == "__main__":      # (only change to the original file: lets other scripts import build_random_cascade)
    np.random.seed(SEED)    # Note: seed will be reset also later to guarantee 
                            # a fair comparison between the different approaches

    print("##############################################")
    print("Running single lambda simulation (Figure 10)")
    print("##############################################")
    sim_single_lambda()

    print("##############################################")
    print("Running multiple lambda simulation (Figure 11)")
    print("##############################################")
    sim_multiple_lambda()

    print("##############################################")
    print("Done! You can plot the results now.")
    print("##############################################")

