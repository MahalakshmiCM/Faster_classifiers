from collections import deque

MAX = 1e99

def optimize(cascade):
    P = [0]
    C = [0]

    # Change the cascade with the same data structures
    # used in the article
    N = len(cascade)
    for x in cascade:
        P.append(x['p'])
        C.append(x['c'])
    
    # Initialize helper deque and array
    Q = deque([(0,0)])
    A = [(0,-1)]        # "halt" is -1 in position 0, others starts from 1

    # Compute the optimized cascade
    f = {}
    for i in range(1,N+1):
        while len(Q) > 1 and C[i] >= Q[1][1]:
            Q.popleft()

        h = Q[0][0]
        f[h] = A[h][0]
        f[i] = f[h] + C[i] * (1-P[h])
        A.append((f[i], h))

        x = 0


        while len(Q) > 0:
            t = len(Q)-1
            qt = Q[t][0]
            if P[i] != P[qt]:
                x = (f[i]-f[qt]) / (P[i]-P[qt])
            else:
                x = float('inf')
            if x > Q[t][1]:
                break
            Q.pop()
        Q.extend([(i,x)])

    # Compute the final solution
    S = []
    iter = N

    while iter != -1:
        if iter != 0:
            S.append(cascade[iter-1])
        iter = A[iter][1]

    return sorted(S, key=lambda d: d['id'])


