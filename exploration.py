import copy
import new_algorithm

def explore_FIRST(cascade, id_true_classifier):
    # Start from scratch until the first classifier classify it

    total_sum = 0
    last_class = None
    for x in cascade:
        total_sum += x['c']
        match_id = x['orig_id'] if 'orig_id' in x else x['id'] 
        if match_id >= id_true_classifier:
            last_class = x
            break

    if last_class == None:   # If this fails, it means that id_true_classifier does not correspond to an existing classifier
        assert(False)
    return total_sum, last_class

def explore_LAST(cascade, last_used_classifier, id_true_classifier):
    total_sum = 0

    if last_used_classifier != None:    # Try to use LAST
        match_id = None
        for i in range(0, len(cascade)):
            if cascade[i]['id'] >= last_used_classifier['id']:
                match_id = i
                break
        assert(match_id != None)

        total_sum += cascade[match_id]['c']

        if cascade[match_id]['id'] >= id_true_classifier:
            # It classifies it, so let's return it
            return total_sum
    
        remaining_cascade = []
        for c in cascade:
            if c['id'] > cascade[match_id]['id']:
                remaining_cascade.append(c)
    else:
        remaining_cascade = cascade

    cost, id_last = explore_FIRST(remaining_cascade, id_true_classifier)
    return total_sum + cost

def explore_OA(cascade, prev_real_classifier, id_true_classifier):
    # Algorithm 2

    if prev_real_classifier == None:
        prev_real_classifier = {'id':-1}

    # Algorithm 1
    better_cascade = new_algorithm.optimize(cascade)

    et_optimized, _ = explore_FIRST(better_cascade, id_true_classifier)

    return et_optimized, better_cascade

def explore_vSOA(cascade, prev_real_classifier, id_true_classifier, skip_back):

    starting_point = 0

    if prev_real_classifier != None:
        for idx in range(0,len(cascade)):
            if cascade[idx]['id'] >= prev_real_classifier['id']:
                break
        starting_point = max(idx-skip_back, 0)

    total_sum = 0
    for i in range(starting_point, len(cascade)):
        total_sum += cascade[i]['c']
        if cascade[i]['id'] >= id_true_classifier:
            return total_sum

    assert(False)