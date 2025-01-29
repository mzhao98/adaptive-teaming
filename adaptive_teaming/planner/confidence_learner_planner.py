import logging
from copy import copy, deepcopy
from itertools import product
from pprint import pprint, pformat

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB

from .base_planner import InteractionPlanner
import pdb
import scipy
from scipy.stats import entropy

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

# planner_type = "informed_greedy"
# planner_type = "greedy"
planner_type = "facility_location"


class ConfidenceLearnerPlanner(InteractionPlanner):

    def __init__(self, interaction_env, belief_estimator, planner_cfg, cost_cfg):
        super().__init__(interaction_env, belief_estimator, planner_cfg, cost_cfg)

    def plan(
        self,
        task_seq,
        pref_beliefs,
        task_similarity_fn,
        pref_similarity_fn,
        current_task_id,
    ):
        """
        task_seq: sequence of tasks each described a vector
        task_similarity_fn: function to compute similarity between tasks
        pref_similarity_fn: function to compute similarity between task preferences

        Output should be a tuple: plan, plan_info

        """

        N = len(task_seq[current_task_id:])
        # some large constant
        M = sum(val for val in self.cost_cfg.values())
        pref_space = self.interaction_env.pref_space
        confidence_threshold = self.planner_cfg['confidence_threshold']

        # Check if there's anything to be gained from queries (prefs or skills)
        current_pref_belief = pref_beliefs[current_task_id]
        best_skill_pref_to_ask_for = None
        list_of_skills_type_pref = [(self.robot_skills[idx]['task'], self.robot_skills[idx]['pref']) for idx in range(len(self.robot_skills))]
        if max(current_pref_belief) < confidence_threshold and sum(pref_beliefs[current_task_id]) >0:
            best_action = 'ASK_PREF'

        else:
            best_skill_pref_to_ask_for = pref_space[np.argmax(current_pref_belief)]
            skill_confidence = 1 if (task_seq[current_task_id], best_skill_pref_to_ask_for) in list_of_skills_type_pref else 0
            if skill_confidence < confidence_threshold:
                best_action = 'ASK_SKILL'
            else:
                best_action = 'ROBOT'



        # update the plan
        plan = [{'action_type': best_action}]
        if best_action == 'ASK_SKILL':

            plan[0]['pref'] = best_skill_pref_to_ask_for

        if best_action == 'ROBOT':
            plan[0]['pref'] = best_skill_pref_to_ask_for
            plan[0]['task'] = task_seq[current_task_id]
            # find skill id for the task
            for idx in range(len(self.robot_skills)):
                if (self.robot_skills[idx]['task'] == task_seq[current_task_id] and
                        self.robot_skills[idx]['pref'] == best_skill_pref_to_ask_for):
                    plan[0]['skill_id'] = self.robot_skills[idx]['skill_id']
                    break

        plan_info = {}


        return plan, plan_info


