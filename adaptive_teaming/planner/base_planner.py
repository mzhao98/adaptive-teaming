import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from pprint import pformat

import numpy as np

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


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
    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        raise NotImplementedError

    def rollout_interaction(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Rollout the interaction between the human and the agent for all the tasks.
        """
        self.robot_skills = []
        total_rew = 0
        self.interaction_env.reset(task_seq)
        # for task_id, task in enumerate(task_seq):
        executed_task_ids, executed_actions, actual_rews = [], [], []
        executed_beliefs = []
        done = False
        task_id = 0
        while not done:
            task = task_seq[task_id]
            executed_task_ids.append(task_id)
            logger.debug(f"Executing task {task_id}")
            logger.debug(f"  Task: {pformat(task)}")
            pref_beliefs = self.belief_estimator.beliefs #[task_id:]
            plan, plan_info = self.plan(
                task_seq,  # [task_id:],
                pref_beliefs,
                task_similarity_fn,
                pref_similarity_fn,
                task_id,
            )
            action = plan[0]
            executed_actions.append(action)
            obs, rew, done, info = self.interaction_env.step(action)

            executed_beliefs.append(deepcopy(self.belief_estimator.beliefs[task_id]))
            if action["action_type"] == "ASK_SKILL":
                self.robot_skills.append(
                    {
                        "task": task,
                        "skill_id": task_id,
                        "pref": info["pref"],
                        "skill": info["skill"],
                    }
                )
                self.belief_estimator.update_beliefs(task_id, info)

            elif action["action_type"] == "ASK_PREF":
                self.belief_estimator.update_beliefs(task_id, info)
                logger.debug(f"  Updated beliefs: {pformat(self.belief_estimator.beliefs)}")

            elif action["action_type"] == "ROBOT":
                # Execute previously learned skill
                # __import__('ipdb').set_trace()
                pass

            elif action["action_type"] == "HUMAN":
                pass

            else:
                raise NotImplementedError

            actual_rews.append(rew)
            total_rew += rew

            logger.debug(f"  Executing action_type: {action['action_type']}")
            logger.debug(f"    reward: {rew}")

            logger.debug("Actions executed so far:")
            logger.debug("------------------------")
            for i, (action, rew) in enumerate(zip(executed_actions, actual_rews)):
                executed_task = task_seq[executed_task_ids[i]]
                logger.debug(f"  Round {i}: Task {executed_task_ids[i]} {executed_task['obj_type']}-{executed_task['obj_color']} \n \
                             \t\t\t\t\t\t\t {action}, belief: {executed_beliefs[i]}, {rew}\n")

            task_id = info["current_task_id"]

        logger.info(f"  Total reward: {total_rew}")


class AlwaysHuman(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(self, task_seq, pref_beliefs, task_similarity_fn, pref_similarity_fn, current_task_id):
        """
        Plan
        """
        return [{"action_type": "HUMAN"}] * len(task_seq), {}


class AlwaysLearn(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(self, task_seq, pref_beliefs, task_similarity_fn, pref_similarity_fn, current_task_id):
        """
        Plan
        """
        return [{"action_type": "ASK_SKILL", "pref": "G1"}] * len(task_seq), {}


class LearnThenRobot(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)
        self.iter = 0

    def plan(self, task_seq, pref_beliefs, task_similarity_fn, pref_similarity_fn, current_task_id):
        """
        Plan
        """

        if self.iter == 0:
            plan = [{"action_type": "ASK_SKILL", "pref": "G1"}]
        else:
            plan = [{"action_type": "ROBOT", "skill_id": 0, "pref": "G1"}]
        plan += [{"action_type": "ROBOT", "skill_id": 0, "pref": "G1"}] * (
            len(task_seq) - 1
        )
        self.iter += 1
        return plan, {}


class FixedPlanner(InteractionPlanner):
    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)
        self._plan = planner_cfg["plan"]
        self.iter = 0

    def plan(self, task_seq, task_similarity_fn, pref_similarity_fn):
        """
        Plan
        """

        plan = []
        for action_type in self._plan[self.iter:]:
            if action_type == "ROBOT":
                plan.append({"action_type": action_type,
                            "skill_id": 0, "pref": "G1"})
            else:
                plan.append({"action_type": action_type})
        self.iter += 1
        return plan, {}
