import logging

import numpy as np
from adaptive_teaming.skills.gridworld_skills import PickPlaceSkill

from .interaction_env import InteractionEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GridWorldInteractionEnv(InteractionEnv):

    def __init__(self, env, human_model_cfg, cost_cfg):
        super().__init__(env, human_model_cfg, cost_cfg)

    def robot_step(self, *args, **kwargs):
        self.env.mission = "Robot's turn"
        return super().robot_step(*args, **kwargs)

    def human_step(self, *args, **kwargs):
        self.env.mission = "User's turn"
        return super().human_step(*args, **kwargs)

    def query_skill(self, task, pref):
        """
        Query the human for demonstrations to learn a skill with specific pref params for the task.
        """
        self.env.mission = f"Requesting user to teach with param {pref}"
        obs, rew, done, info = super().query_skill(task, pref)
        # TODO query with pref params
        demo_key = task["obj_type"] + "-" + task["obj_color"] + "-" + pref
        if demo_key not in self.human_demos:
            raise ValueError(f"Demo not found for key {demo_key}")
        demo = self.human_demos[demo_key]
        # create skill from demo
        skill = PickPlaceSkill([demo])
        info.update({"skill": skill, "pref": pref})
        return None, rew, True, info

    def query_skill_pref(self, task):
        """
        DEPRECATED
        Query the human for demonstrations to learn a skill for the task along with their pref.
        """
        self.env.mission = "Requesting user to teach"
        obs, rew, done, info = super().query_skill_pref(task)
        demo = self.human_demos[0]
        _, _, _, pref_info = self.query_pref(task)
        # create skill from demo
        skill = PickPlaceSkill([demo])
        info.update({"skill": skill, "pref": pref_info["pref"]})
        return None, rew, True, info

    def query_pref(self, task):
        """
        Query the human for their preference for task.
        """
        self.env.mission = "Asking user their preference"
        obs, rew, done, info = super().query_pref(task)

        # distribution over the preference space
        # ground_truth_pref = task["ASK_PREF"]
        # sample this distribution using a Boltzmann distribution
        # pref_sample = np.random.choice(
        # list(self.pref_space.keys()), p=ground_truth_pref
        # )
        pref_sample = self._get_human_pref_for_object(
            self.env.objects[0]["object"])
        info["pref"] = pref_sample
        return None, rew, True, info

    def human_evaluation_of_robot(self, terminal_state=None, traj=None):
        """
        Evaluate the robot's performance. Note that the robot does not have
        access to the evaluation during the interaction but only at the end.

        :args:
        env_state : dict
            The environment state.
        traj : list
            The trajectory of the robot.
        """
        agent_pos = self.env.agent_pos
        if self.env.carrying:
            # object pos same as agent pos
            human_pref_goal = self._get_human_pref_for_object(
                self.env.carrying)
            logger.debug(f"  Human pref goal: {human_pref_goal}")
            goal_pos = self.env.goals[human_pref_goal](
                self.env.width, self.env.height)
            # print("agent_pos: ", agent_pos)
            # print("goal_pos: ", goal_pos)
            if agent_pos[0] == goal_pos[0] and agent_pos[1] == goal_pos[1]:
                logger.debug("HUMAN PREF GOAL REACHED")
                return 1
        return 0

    def _get_human_pref_for_object(self, obj):
        if obj.type == "key":
            if obj.color == "red":
                return "G1"
            elif obj.color == "blue":
                return "G2"
            else:
                return "G2"

        elif obj.type == "ball":
            return "G2"
        elif obj.type == "box":
            return "G2"
        else:
            raise NotImplementedError

    @staticmethod
    def task_similarity(task1, task2):
        pass
        if task1["obj_type"] == task2["obj_type"]:
            return 1
        else:
            return 0

    @staticmethod
    def pref_similarity(pref1, pref2):
        # return np.exp(-np.linalg.norm(pref1 - pref2))
        return int(pref1 == pref2)
