import logging

import numpy as np
from adaptive_teaming.skills.gridworld_skills import PickPlaceSkill
import minigrid
from .interaction_env import InteractionEnv
import matplotlib.pyplot as plt
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import heapq
import numpy as np

def heuristic(a, b):
    """Manhattan distance heuristic for a grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid, start, goal):
    """A* algorithm for finding the shortest path."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, []))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        path = path + [current]

        if current == goal:
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (
                0 <= neighbor[0] < rows
                and 0 <= neighbor[1] < cols
                and grid[neighbor[0], neighbor[1]] != 1
                and neighbor not in visited
            ):
                heapq.heappush(
                    open_set,
                    (
                        cost + 1 + heuristic(neighbor, goal),
                        cost + 1,
                        neighbor,
                        path,
                    ),
                )
    return None

def convert_to_actions(path, pick_up_object, drop_object):
    """Convert a path into actions (up, down, left, right, tab, backspace)."""
    actions = []

    for i in range(len(path) - 1):
        current = path[i]
        next_ = path[i + 1]

        if next_[0] == current[0] - 1:
            actions.append("up")
        elif next_[0] == current[0] + 1:
            actions.append("down")
        elif next_[1] == current[1] - 1:
            actions.append("left")
        elif next_[1] == current[1] + 1:
            actions.append("right")

    if pick_up_object:
        actions.append("tab")
    if drop_object:
        actions.append("backspace")

    return actions

def render_path(grid, path):
    """Render the grid and the planned path."""
    grid_display = grid.copy()
    for step in path:
        if grid_display[step] == 0:  # Mark path only on free space
            grid_display[step] = 0.5

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_display, cmap="gray_r")
    plt.title("Planned Path")
    plt.show()


class GridWorldInteractionEnv(InteractionEnv):

    def __init__(self, env, human_model_cfg, cost_cfg):
        super().__init__(env, human_model_cfg, cost_cfg)
        self.true_human_pref = None

    def robot_step(self, *args, **kwargs):
        self.env.mission = "Robot's turn"
        return super().robot_step(*args, **kwargs)

    def human_step(self, *args, **kwargs):
        self.env.mission = "User's turn"
        human_pref = self._get_human_pref_for_object(
            self.env.objects[0]["object"])
        return super().human_step(human_pref, *args, **kwargs)

    def query_skill_from_saved_demos(self, task, pref):
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

    def query_skill(self, task, pref):
        """
        Query the human for demonstrations to learn a skill with specific pref params for the task.
        """
        self.env.mission = f"Requesting user to teach with param {pref}"
        obs, rew, done, info = super().query_skill(task, pref)
        # TODO query with pref params
        demo_key = task["obj_type"] + "-" + task["obj_color"] + "-" + pref

        mini_grid = self.env.grid
        mini_grid_list = self.env.grid.grid # list
        grid_width = self.env.grid.width
        grid_height = self.env.grid.height
        easy_grid = np.zeros((grid_width, grid_height))
        type_to_encoding = {minigrid.core.world_object.Wall: 1,
                            type(None): 0,
                            minigrid.core.world_object.Goal: 2,
                            minigrid.core.world_object.Box: 3,
                            minigrid.core.world_object.Key: 3,
                            minigrid.core.world_object.Ball: 3}  # 3 is for objects
        obj_position = None
        for i in range(grid_width):
            for j in range(grid_height):
                cell_type = type(self.env.grid.get(i, j))
                # print(cell_type)
                # pdb.set_trace()
                easy_grid[j, i] = type_to_encoding[cell_type]
                if type_to_encoding[cell_type] == 3:
                    obj_position = (j, i)

        print("easy_grid", easy_grid)
        correct_goal = self._get_human_pref_for_object(
            self.env.objects[0]["object"])
        agent_start_pos = self.env.agent_start_pos # (1,1)
        goals = self.env.get_goals()
        # invert the (i,j) coordinates of the goals
        goals = {goal: (y, x) for goal, (x, y) in goals.items()}


        start = agent_start_pos
        object_location = obj_position
        goal_location = goals[correct_goal]

        # Path to object
        path_to_object = a_star_search(easy_grid, start, object_location)
        # Path from object to goal
        path_from_object_to_goal = a_star_search(easy_grid, object_location, goal_location)

        # Combine paths and actions
        if path_to_object and path_from_object_to_goal:
            full_path = path_to_object + path_from_object_to_goal[1:]
            actions = (
                    convert_to_actions(path_to_object, pick_up_object=True, drop_object=False) +
                    convert_to_actions(path_from_object_to_goal, pick_up_object=False, drop_object=True)
            )
            print("Path:", full_path)
            print("Actions:", actions)
        else:
            print("Path not found.")

        # render_path(easy_grid, full_path)
        demo = actions
        # demo = self.human_demos[demo_key]
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
        # pdb.set_trace()
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

    def set_human_pref(self, pref):
        self.true_human_pref = pref

    def _get_human_pref_for_object(self, obj):
        # if obj.type == "key":
        #     if obj.color == "red":
        #         return "G1"
        #     elif obj.color == "blue":
        #         return "G2"
        #     else:
        #         return "G2"
        #
        # elif obj.type == "ball":
        #     return "G2"
        # elif obj.type == "box":
        #     return "G2"
        # else:
        if (obj.type, obj.color) in self.true_human_pref:
            return self.true_human_pref[(obj.type, obj.color)]
        else:
            raise NotImplementedError

    @staticmethod
    def task_similarity(task1, task2):
        # pass
        # if task1["obj_type"] == task2["obj_type"] and task1["obj_color"] == task2["obj_color"]:
        if task1["obj_type"] == task2["obj_type"] and task1["position"] == task2["position"]:
            return 1
        else:
            return 0

    @staticmethod
    def pref_similarity(pref1, pref2):
        # return np.exp(-np.linalg.norm(pref1 - pref2))
        return int(pref1 == pref2)
