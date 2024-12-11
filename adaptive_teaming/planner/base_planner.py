from abc import ABC, abstractmethod


class InteractionPlanner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        raise NotImplementedError

    def rollout_interaction(self):
        """
        Rollout
        """
        pass
