

from itertools import permutations

def test_perm():
    list_perms = list(permutations(range(0, 4)))
    print(list_perms)


def expected_rank_via_all_perms(ranking_size, list_preds, target_index):
    '''
    compute the expected rank based on likelihoods of all rankings, where the probability of one ranking is based on pairwise comparisons
    @param ranking_size:
    @param list_preds:
    @param target_index:
    @return:
    '''
    pass

def expected_rank_direct(ranking_size, list_preds, target_index):
    '''
    @param ranking_size:
    @param list_preds:
    @param target_index:
    @return:
    '''
    pass



if __name__ == '__main__':
    # (1)
    test_perm()