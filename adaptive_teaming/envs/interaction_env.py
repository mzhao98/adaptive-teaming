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

    def __init__(self, env, human_model_cfg, cost_cfg, prob_skill_teaching_success=1):
        self.env = env
        self.human_model_cfg = human_model_cfg
        self.cost_cfg = cost_cfg
        self.task_seq = []
        self.current_task_id = 0
        self.human_demos = []
        self.robot_skills = self._init_robot_skills()
        self.prob_skill_teaching_success = prob_skill_teaching_success

    def reset(self, task_seq):
        """
        Reset the interaction environment to a new task sequence.
        """
        self.task_seq = task_seq
        self.current_task_id = 0
        self.robot_skills = self._init_robot_skills()

    def load_human_demos(self, human_demos):
        self.human_demos = human_demos

    def choose_best_skill(self, task, pref_beliefs, current_skill_library):
        """
        Choose the best skill with the lowest cost.
        """
        beliefs = pref_beliefs
        goal_names = {'G1':0, 'G2':1, 'G3':2}
        best_skill = None
        best_skill_cost = float('inf')
        best_pref = None
        for skill_id in current_skill_library:
            skill = current_skill_library[skill_id]
            # pdb.set_trace()
            skill_obj_type = skill.obj_type
            if skill_obj_type != (task['obj_type'], task['obj_color']):
                continue
            likelihood_of_success = 1
            goal_loc_of_skill = current_skill_library[skill_id].goal_loc
            goals_list = current_skill_library[skill_id].goals_list
            # pdb.set_trace()
            # invert goals_list
            goals_dict = {(v[1], v[0]): k for k, v in goals_list.items()}
            # get the key for index of the goal location in the goals dict
            candidate_pref = goals_dict[goal_loc_of_skill]
            # assign index of goal
            goal_index = goal_names[candidate_pref]
            # print("goal_index", goal_index)
            # print("beliefs", beliefs)

            # get the likelihood of success for the goal
            likelihood_of_success = beliefs[goal_index]
            # get the cost of the skill
            skill_cost = likelihood_of_success * self.cost_cfg['PREF_COST']
            # print("skill_cost", skill_cost)
            if skill_cost < best_skill_cost:
                best_skill = skill
                best_pref = candidate_pref
                best_skill_cost = skill_cost
        return best_skill, best_pref



    def step(self, action, pref_beliefs=None):
        """
        Take a step in the interaction environment.
        """
        task = self.task_seq[self.current_task_id]
        # pdb.set_trace()
        if action["action_type"] == "ROBOT":
            print("current self.robot_skills", self.robot_skills)
            skill = self.robot_skills[action["skill_id"]]
            # pdb.set_trace()
            obs, rew, done, info = self.robot_step(
                task, skill, action["pref"]
            )
            # pdb.set_trace()
            self.current_task_id += 1
        elif action["action_type"] == "HUMAN":
            human_pref_chosen = self._get_human_pref_for_object_given_type_color(task['obj_type'], task['obj_color'])
            print('human_pref_chosen', human_pref_chosen)
            obs, rew, done, info = self.human_step(human_pref=human_pref_chosen, task=task)
            self.current_task_id += 1

        elif action["action_type"] == "ASK_SKILL":
            obs, rew, done, info = self.query_skill(task, action["pref"])
            if info["teach_success"] is True:
                self.robot_skills[self.current_task_id] = info["skill"]
            # else:
            #     pdb.set_trace()
            # print("info", info)
            # if not info["teach_success"]:
            if info["teach_success"] is False:
                # pdb.set_trace()
                # human has to intervene to complete the task
                rew -= self.cost_cfg["HUMAN"]
            else:
                skill, pref = self.choose_best_skill(task, pref_beliefs[self.current_task_id], self.robot_skills)
                # pdb.set_trace()
                obs_r, rew_r, done_r, info_r = self.robot_step(
                    task, skill, pref
                )
                rew += rew_r
                # pdb.set_trace()
                # if rew_r < -200:
                #     pdb.set_trace()
            self.current_task_id += 1
            # XXX if we want to be truly adaptive, the we should not move
            # forward, but this will really disadvantage the baselines
            # if info["teach_success"]:
                # self.current_task_id += 1
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
        obs, rew, done, info = skill.step(self.env, pref_params, obs, render=self.env.has_renderer)
        # env reward is irrelevant
        # pdb.set_trace()

        rew = 0
        if info["safety_violated"] is True:
            # rew -= 100 # I think this should be fail cost
            rew -= self.cost_cfg['FAIL']
            print("path failed")
            pdb.set_trace()

        human_sat = self.human_evaluation_of_robot(
            None)
        # pdb.set_trace()
        if human_sat == 0:
            print("human pref not satisfied")
            # pdb.set_trace()
        rew -= (1 - human_sat) * self.cost_cfg["PREF_COST"]

        rew -= self.cost_cfg["ROBOT"]
        # if rew < -self.cost_cfg["ROBOT"]:
        #     pdb.set_trace()
        return None, rew, True, {}

    # human model
    # -------------
    def human_step(self, human_pref, task):
        print("human_pref", human_pref)
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["HUMAN"]
        if self.env.has_renderer:
            for _ in range(10):
                self.env.render()
        return None, rew, True, {'pref': human_pref}

    def query_skill(self, task, pref):
        """
        Query the human for demonstrations with specific pref params to learn a skill for the task.
        """
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["ASK_SKILL"]
        if self.env.has_renderer:
            for _ in range(10):
                self.env.render()
        return obs, rew, True, {}

    def query_skill_pref(self, task):
        """
        DEPRECATED
        Query the human for demonstrations to learn a skill for the task along with their pref.
        """
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["ASK_SKILL"]
        if self.env.has_renderer:
            for _ in range(10):
                self.env.render()
        return None, rew, True, {}

    def query_pref(self, task):
        """
        Query the human for their preference for task.
        """
        obs = self.env.reset_to_state(task)
        rew = -self.cost_cfg["ASK_PREF"]
        if self.env.has_renderer:
            for _ in range(10):
                self.env.render()
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
