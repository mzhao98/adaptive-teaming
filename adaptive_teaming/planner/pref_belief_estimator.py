from abc import abstractmethod
import logging

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
        for task, belief in zip(self.task_seq[task_id:], self.beliefs[task_id:]):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            # bayesian update of the belief
            # XXX fix this
            pref = obs["pref"]
            pref_idx = self._pref_to_idx(pref)
            belief[pref_idx] +=  task_sim
            belief /= belief.sum()

    def _pref_to_idx(self, pref):
        return list(self.env.pref_space).index(pref)

    def task_similarity_fn(self, task1, task2):
        """
        Task similarity function
        """
        if task1["obj_type"] == task2["obj_type"]:
            return 1
        else:
            return 0
