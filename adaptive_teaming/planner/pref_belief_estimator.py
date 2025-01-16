import logging
from abc import abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class PrefBeliefEstimator:
    def __init__(self, task_seq):
        self.task_seq = task_seq

    @abstractmethod
    def prior(self):
        raise NotImplementedError

    def update_beliefs(self, task_id, obs):
        """
        Update belief
        """
        pass


class GridWorldBeliefEstimator(PrefBeliefEstimator):
    def __init__(self, env, task_seq):
        super().__init__(task_seq)

        self.env = env
        # indexed by task id
        self.beliefs = [self.prior() for _ in range(len(task_seq))]

    def prior(self):
        """
        Prior belief

        This is a discrete probability distribution over the preference parameter spaces.
        TODO: figure out how to represent the preference
        """
        return np.ones(len(self.env.pref_space)) / len(self.env.pref_space)

    def update_beliefs(self, task_id, obs):
        """
        Update belief
        """
        # immediately converges to 1 after a preference query
        pref = obs["pref"]
        pref_idx = self._pref_to_idx(pref)
        self.beliefs[task_id][pref_idx] = 1
        # print("self.beliefs[task_id]:", self.beliefs[task_id])
        for other_pref_idx in range(len(self.beliefs[task_id])):
            if other_pref_idx != pref_idx:
                self.beliefs[task_id][other_pref_idx] = 0

        for task, belief in zip(self.task_seq[task_id+1:],
                                self.beliefs[task_id+1:]):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if task_sim:
                belief[pref_idx] = 1

                for other_pref_idx in range(len(belief)):
                    if other_pref_idx != pref_idx:
                        belief[other_pref_idx] = 0

            # bayesian update of the belief
            # XXX fix this
            # belief[pref_idx] += task_sim

            # belief /= belief.sum()

    def _pref_to_idx(self, pref):
        return list(self.env.pref_space).index(pref)

    def task_similarity_fn(self, task1, task2):
        """
        Task similarity function
        """
        if task1["obj_type"] == task2["obj_type"]:
            if task1["obj_color"] == task2["obj_color"]:
                return 1
            else:
                return 0
        else:
            return 0
