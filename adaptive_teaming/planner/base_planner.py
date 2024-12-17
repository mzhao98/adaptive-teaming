from abc import ABC, abstractmethod


class InteractionPlanner(ABC):
    def __init__(self, interaction_env, cost_cfg):
        self.interaction_env = interaction_env
        self.cost_cfg = cost_cfg

    @property
    def action_space(self):
        return ["ROBOT", "HUMAN", "ASK_DEMO", "ASK_PREF"]

    @abstractmethod
    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        raise NotImplementedError

    def rollout_interaction(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Rollout the interaction between the human and the agent for all the tasks.
        """
        self.interaction_env.reset(task_seq)
        for i, task in enumerate(task_seq):
            plan = self.plan(
                task_seq[i:], task_similarity_fn, pref_similarity_fn)
            action = plan[0]
            obs, rew, done, info = self.interaction_env.step(action)
