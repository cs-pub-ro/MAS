from preflibtools import io
from preflibtools.generate_profiles import gen_mallows, gen_cand_map, gen_impartial_culture_strict
from typing import List, Dict, Tuple
import random


PHIS = [0.7, 0.8, 0.9, 1.0]
NUM_VOTERS = [100, 500, 1000]
NUM_CANDIDATES = [3, 6, 10, 15]



def generate_random_mixture(nvoters: int = 100, ncandidates: int = 6, num_refs: int = 3, phi: float = 1.0) \
    -> Tuple[Dict[int, str], List[Dict[int, int]], List[int]]:
    """
    Function that will generate a `voting profile` where there are num_refs mixtures of a
    Mallows model, each with the same phi hyperparameter
    :param nvoters: number of voters
    :param ncandidates: number of candidates
    :param num_refs: number of Mallows Mixtures in the voting profile
    :param phi: hyper-parameter for each individual Mallows model
    :return: a tuple consisting of:
        the candidate map (map from candidate id to candidate name),
        a ranking list (list consisting of dictionaries that map from candidate id to order of preference)
        a ranking count (the number of times each vote order comes up in the ranking list,
        i.e. one or more voters may end up having the same preference over candidates)
    """
    candidate_map = gen_cand_map(ncandidates)

    mix = []
    phis = []
    refs = []

    for i in range(num_refs):
        refm, refc = gen_impartial_culture_strict(1, candidate_map)
        refs.append(io.rankmap_to_order(refm[0]))
        phis.append(phi)
        mix.append(random.randint(1,100))

    smix = sum(mix)
    mix = [float(m)/float(smix) for m in mix]

    rmaps, rmapscounts = gen_mallows(nvoters, candidate_map, mix, phis, refs)

    return candidate_map, rmaps, rmapscounts


def stv(nvoters: int,
        canidate_map: Dict[int, str],
        rankings: List[Dict[int, int]],
        ranking_counts: List,
        top_k: int,
        required_elected: int) -> List[int]:
    """
    :param nvoters: number of voters
    :param canidate_map: the mapping of candidate IDs to candidate names
    :param rankings: the expressed full rankings of voters, specified as a list of mapping from candidate_id -> rank
    :param ranking_counts:
    :param top_k: the number of preferences taken into account [min: 2, max: (num_candidates - 1), aka full STV]
    :return: The list of elected candidate id-s
    """
    # TODO: implement STV-k


if __name__ == "__main__":
    cmap, rmaps, rmapscounts = generate_random_mixture()
    print(cmap)
    print(rmaps)
    print(rmapscounts)
