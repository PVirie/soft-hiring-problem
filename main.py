import numpy as np
import matplotlib.pyplot as plt
import math
from math import comb


def random_score(N):
    return np.random.randint(0, 1000, N)/1000


def random_score_gaussian(N):
    # mean at N/2
    return np.random.normal(N/2, 1, N)


def execute_hiring(scores, K):
    K = int(K)
    max_of_first_K = np.max(scores[:K])
    N = len(scores)
    for i in range(K, N):
        if scores[i] > max_of_first_K:
            return i
    # always pick the last
    return i


def compute_expected_scores(scores):
    N = len(scores)

    sorted_score = []
    def find_percentile_and_push(h):
        if len(sorted_score) == 0:
            sorted_score.append(h)
            return 0.5

        # perform binary search
        l = 0
        r = len(sorted_score) - 1
        i = 0
        while l <= r:
            mid = (l + r) // 2
            if h <= sorted_score[mid]:
                r = mid - 1
                i = mid
            else:
                l = mid + 1
                i = mid + 1
    
        sorted_score.insert(i, h)
        return (i + 1)/len(sorted_score)

    count_black_swans = 1
    for i in range(N):
        p_i = 0
        c = find_percentile_and_push(scores[i])
        if abs(c - 1.0) < 1e-6:
            # black swan
            c = c * (1 - count_black_swans / N)
            count_black_swans += 1

        t = (c)**(N - i) # chance none the rest is better than i

        R = N - i - 1
        for j in range(N - i):
            power = (c**j) * ((1 - c)**(R - j)) * comb(R, j)
            coeff = ((c*i + j + 1)/N)**(R)
            p_i += coeff * power

        yield p_i, t

    

def run_experiment(M=1000, N=100):

    total_score_secretary = 0
    total_score_soft_hiring = 0

    for i in range(M):
        scores = random_score(N)

        p = execute_hiring(scores, N/math.e)
        total_score_secretary += scores[p]

        j = 0
        for e, c in compute_expected_scores(scores):
            if e > 0.5 and j > N/math.e:
                break
            j += 1
        total_score_soft_hiring += scores[j]

        print("Percent ", i*100/M, end='\r')

    print()
    print("Secretary score", total_score_secretary/M)
    print("Soft hiring score", total_score_soft_hiring/M)


if __name__ == '__main__':
    
    run_experiment()
    
    N = 100
    X = np.arange(0, N)
    scores = random_score(N)

    expected_scores = np.zeros(N)
    chance_of_being_best = np.zeros(N)
    i = 0
    for e, c in compute_expected_scores(scores):
        expected_scores[i] = e
        chance_of_being_best[i] = c
        i += 1

    plt.bar(X, scores/np.max(scores), color='b', label='score')
    plt.plot(X, chance_of_being_best, color='g', label='p best')
    plt.plot(X, expected_scores, color='r', label='p not better')
    
    plt.xlabel("N") 
    plt.ylabel("Score") 
    plt.title("soft hiring problem") 
    plt.legend()
    plt.show()


    