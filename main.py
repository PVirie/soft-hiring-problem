import numpy as np
import matplotlib.pyplot as plt
import math
from math import comb


def random_score(N, normal=False):
    if normal:
        data = np.random.normal(0.5, 0.1666667, N)
        # make sure all scores are positive and no more than 1
        min_data = min(np.min(data), 0)
        max_data = max(np.max(data), 1.0)
        data = (data - min_data) / (max_data - min_data)
        return data

    return np.random.randint(0, 1000, N)/1000


def execute_hiring(scores, K):
    K = int(K)
    max_of_first_K = np.max(scores[:K])
    N = len(scores)
    for i in range(K, N):
        if scores[i] > max_of_first_K:
            return i
    return -1


def compute_expected_scores(scores, black_swan=True, oracle_bs=False):
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

    count_black_swans = 0
    for i in range(N):
        p_i = 0
        c = find_percentile_and_push(scores[i])
        if abs(c - 1.0) < 1e-6 and black_swan:
            # black swan
            count_black_swans += 1
            if oracle_bs:
                c = c * scores[i]
            else:
                c = c * (1 - count_black_swans / (i + 1))

        t = (c)**(N - i) # chance none the rest is better than i

        R = N - i - 1
        for j in range(N - i):
            power = (c**j) * ((1 - c)**(R - j)) * comb(R, j)
            coeff = ((c*i + j + 1)/N)**(R)
            p_i += coeff * power

        yield p_i, t

    

def run_experiment(M, N, normal):

    total_score_secretary = 0
    total_score_secretary_desperate = 0
    total_score_secretary_bearden = 0
    total_score_rank_black_swan = 0
    total_score_soft_hiring_black_swan = 0
    total_score_rank_withhold = 0
    total_score_soft_hiring_withhold = 0
    total_score_rank_wh_bearden = 0
    total_score_soft_hiring_wh_bearden = 0
    total_score_rank_oracle_bs = 0
    total_score_soft_hiring_oracle_bs = 0

    for i in range(M):
        scores = random_score(N, normal=normal)

        p = execute_hiring(scores, N/math.e)
        if p >= 0:
            total_score_secretary += scores[p]
            total_score_secretary_desperate += scores[p]
        else:
            total_score_secretary_desperate += scores[-1]

        p = execute_hiring(scores, math.sqrt(N))
        if p >= 0:
            total_score_secretary_bearden += scores[p]
        else:
            total_score_secretary_bearden += scores[-1]

        rank_black_swan_picked = False
        soft_hiring_black_swan_picked = False
        rank_withhold_picked = False
        soft_hiring_withhold_picked = False
        rank_wh_bearden_picked = False
        soft_hiring_wh_bearden_picked = False
        rank_oracle_bs_picked = False
        soft_hiring_oracle_bs_picked = False
        

        for j, (e, c) in enumerate(compute_expected_scores(scores, black_swan=True)):
            if not soft_hiring_black_swan_picked and e > 0.5:
                total_score_soft_hiring_black_swan += scores[j]
                soft_hiring_black_swan_picked = True

            if not rank_black_swan_picked and c > 0.5:
                total_score_rank_black_swan += scores[j]
                rank_black_swan_picked = True

            if rank_black_swan_picked and soft_hiring_black_swan_picked:
                break

        if not rank_black_swan_picked:
            total_score_rank_black_swan += scores[-1]

        if not soft_hiring_black_swan_picked:
            total_score_soft_hiring_black_swan += scores[-1]

        for j, (e, c) in enumerate(compute_expected_scores(scores, black_swan=False)):
            # no black swan, need trial period

            if j > N/math.e and not rank_withhold_picked and c > 0.5:
                total_score_rank_withhold += scores[j]
                rank_withhold_picked = True

            if j > N/math.e and not soft_hiring_withhold_picked and e > 0.5:
                total_score_soft_hiring_withhold += scores[j]
                soft_hiring_withhold_picked = True

            if j > math.sqrt(N) and not rank_wh_bearden_picked and e > 0.5:
                total_score_rank_wh_bearden += scores[j]
                rank_wh_bearden_picked = True

            if j > math.sqrt(N) and not soft_hiring_wh_bearden_picked and c > 0.5:
                total_score_soft_hiring_wh_bearden += scores[j]
                soft_hiring_wh_bearden_picked = True

            if rank_withhold_picked and soft_hiring_withhold_picked and rank_wh_bearden_picked and soft_hiring_wh_bearden_picked:
                break

        if not rank_withhold_picked:
            total_score_rank_withhold += scores[-1]

        if not soft_hiring_withhold_picked:
            total_score_soft_hiring_withhold += scores[-1]

        if not rank_wh_bearden_picked:
            total_score_rank_wh_bearden += scores[-1]

        if not soft_hiring_wh_bearden_picked:
            total_score_soft_hiring_wh_bearden += scores[-1]


        for j, (e, c) in enumerate(compute_expected_scores(scores, black_swan=True, oracle_bs=True)):
            if not soft_hiring_oracle_bs_picked and e > 0.5:
                total_score_soft_hiring_oracle_bs += scores[j]
                soft_hiring_oracle_bs_picked = True

            if not rank_oracle_bs_picked and c > 0.5:
                total_score_rank_oracle_bs += scores[j]
                rank_oracle_bs_picked = True

            if rank_oracle_bs_picked and soft_hiring_oracle_bs_picked:
                break

        if not rank_oracle_bs_picked:
            total_score_rank_oracle_bs += scores[-1]

        if not soft_hiring_oracle_bs_picked:
            total_score_soft_hiring_oracle_bs += scores

        print("Percent ", i*100/M, end='\r')

    print()
    print("Secretary score", total_score_secretary/M)
    print("Secretary desperate score", total_score_secretary_desperate/M)
    print("Secretary Bearden score", total_score_secretary_bearden/M)
    print("Rank score with black swan", total_score_rank_black_swan/M)
    print("Soft hiring score with black swan", total_score_soft_hiring_black_swan/M)
    print("Rank score with hold 1/e", total_score_rank_withhold/M)
    print("Soft hiring score with hold 1/e", total_score_soft_hiring_withhold/M)
    print("Rank score with hold sqrt(N)", total_score_rank_wh_bearden/M)
    print("Soft hiring score with hold sqrt(N)", total_score_soft_hiring_wh_bearden/M)
    print("Rank score oracle bs", total_score_rank_oracle_bs/M)
    print("Soft hiring score oracle bs", total_score_soft_hiring_oracle_bs/M)


if __name__ == '__main__':
    
    print("Uniform distribution")
    run_experiment(M=10000, N=200, normal=False)
    print("=====================================")
    print("Normal distribution")
    run_experiment(M=10000, N=200, normal=True)
    print("=====================================")
    
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


    