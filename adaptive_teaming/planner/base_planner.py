import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InteractionPlanner(ABC):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        self.interaction_env = interaction_env
        self.belief_estimator = belief_estimator
        self.planner_cfg = planner_cfg
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
        self.robot_skills = []
        total_rew = 0
        self.interaction_env.reset(task_seq)
        # for task_id, task in enumerate(task_seq):
        done = False
        task_id = 0
        while not done:
            task = task_seq[task_id]
            plan = self.plan(task_seq[task_id:],
                             task_similarity_fn, pref_similarity_fn)
            action = plan[0]
            obs, rew, done, info = self.interaction_env.step(action)

            if action["action_type"] == "ASK_SKILL":
                self.robot_skills.append(
                    {"task": task, "skill_id": task_id, "skill": info["skill"]}
                )
                self.belief_estimator.update_beliefs(task_id, info)

            elif action["action_type"] == "ASK_PREF":
                self.belief_estimator.update_beliefs(task_id, info)

            elif action["action_type"] == "ROBOT":
                # XXX get actual human cost for assessment
                human_sat = self.interaction_env.human_evaluation_of_robot(
                    None)
                rew -= (1 - human_sat) * self.cost_cfg["PREF_COST"]

            elif action["action_type"] == "HUMAN":
                pass

            else:
                raise NotImplementedError

            total_rew += rew

            logger.debug(f"  Executing action_type: {action['action_type']}")
            logger.debug(f"    reward: {rew}")

            task_id = info["current_task_id"]

        logger.info(f"  Total reward: {total_rew}")


class AlwaysHuman(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Plan
        """
        return [{"action_type": "HUMAN"}] * len(task_seq)


class AlwaysLearn(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Plan
        """
        return [{"action_type": "ASK_SKILL"}] * len(task_seq)


class LearnThenRobot(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)
        self.iter = 0

    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Plan
        """

        if self.iter == 0:
            plan = [{"action_type": "ASK_SKILL"}]
        else:
            plan = [{"action_type": "ROBOT", "skill_id": 0, "pref": "G1"}]
        plan += [{"action_type": "ROBOT", "skill_id": 0, "pref": "G1"}] * (
            len(task_seq) - 1
        )
        self.iter += 1
        return plan
