from abc import abstractmethod
import logging
from copy import copy
import pdb
logger = logging.getLogger(__name__)


class InteractionEnv:
    """
    The Interactionenv class models the interaction between the human and the
    agent. It also implements the human model and query responses.
    """

    def __init__(self, env, human_model_cfg, cost_cfg):
        self.env = env
        self.human_model_cfg = human_model_cfg
        self.cost_cfg = cost_cfg
        self.task_seq = []
        self.current_task_id = 0
        self.human_demos = []
        self.robot_skills = self._init_robot_skills()

    def reset(self, task_seq):
        """
        Reset the interaction environment to a new task sequence.
        """
        self.task_seq = task_seq
        self.current_task_id = 0
        self.robot_skills = self._init_robot_skills()

    def load_human_demos(self, human_demos):
        self.human_demos = human_demos

    def step(self, action):
        """
        Take a step in the interaction environment.
        """
        task = self.task_seq[self.current_task_id]
        if action["action_type"] == "ROBOT":

            skill = self.robot_skills[action["skill_id"]]
            obs, rew, done, info = self.robot_step(
                task, skill, action["pref"]
            )
            # pdb.set_trace()
            self.current_task_id += 1
        elif action["action_type"] == "HUMAN":
            obs, rew, done, info = self.human_step(None, task)
            self.current_task_id += 1
        elif action["action_type"] == "ASK_SKILL":
            obs, rew, done, info = self.query_skill(task, action['pref'])
            self.robot_skills[self.current_task_id] = info["skill"]
            self.current_task_id += 1
        elif action["action_type"] == "ASK_PREF":
            obs, rew, done, info = self.query_pref(task)
        else:
            raise NotImplementedError

        info.update({"current_task_id": self.current_task_id})
        if self.current_task_id == len(self.task_seq):
            done = True
        else:
            done = False

        return obs, rew, done, info

    def robot_step(self, task, skill, pref_params):
        """
        Take a robot step.
        """
        obs = self.env.reset_to_state(task)
        print("task", task)
        obs, rew, done, info = skill.step(self.env, pref_params, obs)
        # env reward is irrelevant
        # pdb.set_trace()

        rew = 0
        if info["safety_violated"] is True:
            # rew -= 100 # I think this should be fail cost
            rew -= self.cost_cfg['FAIL']


        human_sat = self.human_evaluation_of_robot(
            None)
        # if human_sat == 0:
        #     pdb.set_trace()
        rew -= (1 - human_sat) * self.cost_cfg["PREF_COST"]

        rew -= self.cost_cfg["ROBOT"]
        # pdb.set_trace()
        return None, rew, True, {}

    # human model
    # -------------
    def human_step(self, human_pref, task):
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["HUMAN"]
        for _ in range(10):
            self.env.render()
        return None, rew, True, {'pref': human_pref}

    def query_skill(self, task, pref):
        """
        Query the human for demonstrations with specific pref params to learn a skill for the task.
        """
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["ASK_SKILL"]
        for _ in range(10):
            self.env.render()
        return None, rew, True, {}

    def query_skill_pref(self, task):
        """
        DEPRECATED
        Query the human for demonstrations to learn a skill for the task along with their pref.
        """
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["ASK_SKILL"]
        for _ in range(10):
            self.env.render()
        return None, rew, True, {}

    def query_pref(self, task):
        """
        Query the human for their preference for task.
        """
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["ASK_PREF"]
        for _ in range(10):
            self.env.render()
        # pdb.set_trace()
        return None, rew, True, {}

    @abstractmethod
    def human_evaluation_of_robot(self, terminal_state, traj=None):
        """
        Evaluate the robot's performance. Note that the robot does not have
        access to the evaluation during the interaction but only at the end.

        :args:
        env_state : dict
            The environment state.
        traj : list
            The trajectory of the robot.
        """
        pass

    def _init_robot_skills(self):
        """
        Initialize robot skills.
        """
        robot_skills = {}
        return robot_skills

    @property
    def pref_space(self):
        """Set of possible preference parameters. The human's preference is a
        pmf over this space."""
        return self.env.pref_space

