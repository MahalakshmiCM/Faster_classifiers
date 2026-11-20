from time import process_time
import new_algorithm
import numpy as np
import csv

# Which "point of the x axis" to test 
NR_CLASSIFIER = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# WCET min / max
WCET_MIN = 1
WCET_MAX = 100

# Number of averaging runs
NR_RUNS = 10000

# Random seed
SEED = 1234

np.random.seed(SEED)

def build_random_cascade(nr_classifiers):
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

csv_writer = csv.writer(open('results/overhead.csv', 'w', newline=''))

for nr_class in NR_CLASSIFIER:

    print("Doing ", nr_class, "...")

    cascade = build_random_cascade(nr_class)

    meas = []

    for i in range(0, NR_RUNS):
        t = process_time()
        new_algorithm.optimize(cascade)
        elapsed_time = process_time() - t
        meas.append(elapsed_time)
    
    csv_writer.writerow([nr_class, np.mean(meas), np.std(meas)])
