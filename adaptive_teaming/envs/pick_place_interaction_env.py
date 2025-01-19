import logging
import re

import numpy as np

from .interaction_env import InteractionEnv

from adaptive_teaming.skills.pick_place_skills import PickPlaceExpertSkill


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PickPlaceInteractionEnv(InteractionEnv):

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
        # TODO load mimicgen demos
        demo_key = task["obj_type"] + "-" + pref
        # if demo_key not in self.human_demos:
            # raise ValueError(f"Demo not found for key {demo_key}")
        # demo = self.human_demos[demo_key]
        # create skill from demo
        # skill = PickPlaceSkill([demo])
        skill = PickPlaceExpertSkill()
        info.update({"skill": skill, "pref": pref})
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
            self.env.objects[self.env.object_id]
        )
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
        # TODO
        obj = self.env.objects[self.env.object_id]
        human_pref_goal = self._get_human_pref_for_object(obj)
        obj_pos = np.array(self.env.sim.data.body_xpos[self.env.obj_body_id[obj.name]])

        human_pref_bin = int(re.findall(r'\d+',  human_pref_goal)[0])
        # if self.env.not_in_bin(obj_pos, human_pref_bin):
        if self.env._check_obj_in_bin(human_pref_bin):
            return 1
        else:
            return 0

    def _get_human_pref_for_object(self, obj):
        # TODO implement multiple preference types
        return "Bin1"

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
