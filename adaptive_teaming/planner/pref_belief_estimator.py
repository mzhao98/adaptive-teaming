import logging
from abc import abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class PrefBeliefEstimator:
    def __init__(self, env, task_seq, teach_adaptive=True):
        self.env = env
        self.task_seq = task_seq
        self.teach_adaptive = teach_adaptive

        # indexed by task id
        self.beliefs = [self.prior() for _ in range(len(task_seq))]

        # Beta random variable for the teach success for each task
        # (Fail, Success)
        self.teach_rvs = np.array([[0, 1] for _ in range(len(task_seq))])

    @abstractmethod
    def prior(self):
        raise NotImplementedError

    def update_beliefs(self, task_id, obs):
        """
        Update belief
        """
        pass

    @abstractmethod
    def task_similarity_fn(self, task1, task2):
        """
        Task similarity function
        """
        raise NotImplementedError

    def update_teach_prob(self, task_id, obs):
        """
        Update belief about the success of the teach.
        """
        teach_success = int(obs["teach_success"])
        # if teach_success == False:
            # import pdb; pdb.set_trace()
        for task, teach_rv in zip(self.task_seq[task_id:], self.teach_rvs[task_id:]):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if task_sim:
                # import pdb; pdb.set_trace()
                teach_rv[teach_success] = 1
                teach_rv[1-teach_success] = 0

    def zero_teach_prob(self, task_id, obs):
        """
        Update belief about the success of the teach.
        """
        teach_success = 0
        # if teach_success == False:
            # import pdb; pdb.set_trace()
        for task, teach_rv in zip(self.task_seq[task_id:], self.teach_rvs[task_id:]):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if task_sim:
                # import pdb; pdb.set_trace()
                teach_rv[teach_success] = 1
                teach_rv[1-teach_success] = 0



    def get_teach_probs(self):
        return self.teach_rvs[:, 1] / np.sum(self.teach_rvs, axis=1)

class GridWorldBeliefEstimator(PrefBeliefEstimator):
    def __init__(self, env, task_seq, teach_adaptive=True):
        super().__init__(env, task_seq, teach_adaptive)

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

        if pref == 'human_only':
            for other_pref_idx in range(len(self.beliefs[task_id])):
                self.beliefs[task_id][other_pref_idx] = 0

                for task, belief in zip(
                    self.task_seq[task_id + 1:], self.beliefs[task_id + 1:]
                ):
                    task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
                    if task_sim:
                        belief[other_pref_idx] = 0
            self.zero_teach_prob(task_id, obs)

        else:
            pref_idx = self._pref_to_idx(pref)
            # if sum(self.beliefs[task_id]) > 0:
            self.beliefs[task_id][pref_idx] *= 1
            # print("self.beliefs[task_id]:", self.beliefs[task_id])
            for other_pref_idx in range(len(self.beliefs[task_id])):
                if other_pref_idx != pref_idx:
                    self.beliefs[task_id][other_pref_idx] *= 0

            for task, belief in zip(
                self.task_seq[task_id + 1:], self.beliefs[task_id + 1:]
            ):
                task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
                if task_sim:
                    belief[pref_idx] *= 1

                    for other_pref_idx in range(len(belief)):
                        if other_pref_idx != pref_idx:
                            belief[other_pref_idx] *= 0

            # bayesian update of the belief
            # XXX fix this
            # belief[pref_idx] += task_sim
        # normalize beliefs
        # print("self.beliefs[task_id]:", self.beliefs)
        for i in range(len(self.beliefs)):
            belief = self.beliefs[i]
            belief_sum = sum(belief)
            if belief_sum > 0:
                self.beliefs[i] /= belief_sum

            # belief /= belief.sum()

        if self.teach_adaptive and "teach_success" in obs:
            self.update_teach_prob(task_id, obs)

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


class PickPlaceBeliefEstimator(PrefBeliefEstimator):
    def __init__(self, env, task_seq, teach_adaptive=True):
        super().__init__(env, task_seq, teach_adaptive)

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
        for other_pref_idx in range(len(self.beliefs[task_id])):
            if other_pref_idx != pref_idx:
                self.beliefs[task_id][other_pref_idx] = 0.1

        for task, belief in zip(
            self.task_seq[task_id + 1:], self.beliefs[task_id + 1:]
        ):
            task_sim = self.task_similarity_fn(self.task_seq[task_id], task)
            if task_sim:
                belief[pref_idx] = 1
            # bayesian update of the belief
            # XXX fix this
            # belief[pref_idx] += task_sim

            # belief /= belief.sum()

        if self.teach_adaptive and "teach_success" in obs:
            self.update_teach_prob(task_id, obs)

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