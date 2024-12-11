import copy
from itertools import product
from random import sample
from typing import Dict, List, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from utils.object import Fork, Mug

SQUARE = "square"
TRIANGLE = "triangle"
PICKUP = "pickup"
PLACE = "place"


class Gridworld:
    """A simple 2D grid world."""

    def __init__(self, map_config: Dict, objects: List, reward_dict: Dict):
        """Initialize a Gridworld Environment.

        ::params:
            ::map_config:
                Dict[map dimensions, agent_position, goal_positions]
            ::object_config: List[Object]
            ::reward_dict:
        """
        # TODO:
        # 1.) No obstacles in map to begin with
        # 2.) Skill execution: straight line interp. (delta x delta y) first to waypoint and then to goal,
        #     use translation mat. and move the object mat.
        # Need to store object locations during execution
        # 3.) Load pngs of objects; add their arrays to the positions in the binary map
        self.reward_dict = reward_dict
        self.map_config = map_config
        self.dimensions = self.map_config["dimensions"]
        self.set_env_limits()
        self.objects = objects
        self.object_poses = tuple(
            (o.orientation, o.position) for o in self.objects
        )

        self.binary_map = self.place_objects(objects=self.objects)

        self.target_goal = reward_dict["target_goal"]
        self.target_object = reward_dict["target_object"]

        # self.square_positions = map_config["square_positions"]
        # self.triangle_positions = map_config["triangle_positions"]
        self.agent_start_position = map_config["agent_position"]
        self.g1 = map_config["g1"]
        self.g2 = map_config["g2"]
        self.objects_in_g1 = None
        self.objects_in_g2 = None

        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # get possible joint actions and actions
        self.possible_single_actions = self.make_actions_list()
        # print("possible single actions", self.possible_single_actions)

        self.current_state = self.create_initial_state()
        self.visualize(self.current_state, title="Initial State")

        # set value iteration components
        (
            self.transitions,
            self.rewards,
            self.state_to_idx,
            self.idx_to_action,
            self.idx_to_state,
            self.action_to_idx,
        ) = (None, None, None, None, None, None)
        self.vf = None
        self.pi = None
        self.policy = None
        self.epsilson = 0.001
        self.gamma = 0.99
        self.maxiter = 40

        # self.step_cost = -0.01
        # self.push_switch_cost = -0.05

        # self.step_cost = reward_weights[-2]
        # self.push_switch_cost = reward_weights[-1]

        self.num_features = 4
        self.correct_target_reward = 10

    @property
    def binary_map(self) -> List:
        """Get Gridworld binary map."""
        return self._binary_map

    @binary_map.setter
    def binary_map(self, new_map: List) -> None:
        """Set Gridworld binary map."""
        self._binary_map = new_map

    @property
    def dimensions(self) -> List:
        """Get Gridworld dimensions."""
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: List) -> None:
        """Set Gridworld dimensions."""
        self._dimensions = dims

    @property
    def objects(self) -> List:
        """Get Gridworld objects."""
        return self._objects

    @objects.setter
    def objects(self, objs: List) -> None:
        """Set Gridworld objects."""
        self._objects = objs

    def place_objects(self, objects: List) -> List:
        """Place objects in empty_map."""
        binary_map = np.zeros(self.dimensions)

        for o in objects:
            try:
                binary_map[
                    o.coordinates[0],
                    o.coordinates[1],
                ] = 1
            except IndexError:
                continue

        return binary_map.tolist()

    def visualize(self, state: Dict, title: Union[None, str] = None) -> None:
        """Visualize Gridworld state."""
        binary_map = self.place_objects(state["objects"])
        fix, ax = plt.subplots()
        ax.imshow(binary_map, cmap="Greys", interpolation="nearest")
        agt_state = (state["agent_position"][1], state["agent_position"][0])
        agt = patches.Rectangle(agt_state, 1, 1, facecolor="r")
        ax.add_patch(agt)
        if title:
            plt.title(title, fontweight="bold")
        plt.show()

    def make_actions_list(self):
        actions_list = []
        actions_list.extend(self.directions)
        actions_list.append(PICKUP)
        actions_list.append(PLACE)
        return actions_list

    def set_env_limits(self) -> None:
        """Set environment limits."""
        if self.dimensions is not None:
            high_x = self.dimensions[0]
            high_y = self.dimensions[1]
        else:
            high_x = high_y = 5

        self.x_min = 0
        self.x_max = high_x
        self.y_min = 0
        self.y_max = high_y

        self.all_coordinate_locations = list(
            product(
                range(self.x_min, self.x_max), range(self.y_min, self.y_max)
            )
        )

    def reset(self):
        for o in self.objects:
            o.reset_pose()
        self.current_state = self.create_initial_state()

    def create_initial_state(self):
        """Create dictionary of object location, type, and picked-up state."""
        state = {}
        state["agent_position"] = copy.deepcopy(self.agent_start_position)
        state["holding"] = None
        state["objects_in_g1"] = None
        state["objects_in_g2"] = None
        state["objects"] = self.objects

        # state['square_positions'] = copy.deepcopy(self.square_positions)
        # state['triangle_positions'] = copy.deepcopy(self.triangle_positions)
        # state['g1'] = copy.deepcopy(self.g1)
        # state['g2'] = copy.deepcopy(self.g2)
        # state['target_object'] = self.target_object
        # state['target_goal'] = self.target_goal

        return state

    def is_done_given_state(self, current_state):
        """Check if target object in target goal."""
        if (
            current_state["objects_in_g1"] is not None
            and self.target_goal == "g1"
            and current_state["objects_in_g1"] == str(self.target_object)
        ):
            return True

        elif (
            current_state["objects_in_g2"] is not None
            and self.target_goal == "g2"
            and current_state["objects_in_g2"] == str(self.target_object)
        ):
            return True

        # check if player at exit location
        # if (
        #     current_state["objects_in_g1"] is not None
        #     or current_state["objects_in_g2"] is not None
        # ):
        #     print(current_state["objects_in_g1"])
        #     print(current_state["objects_in_g2"])
        #     return True

        return False

    def is_valid_push(self, current_state, action):
        # action_type_moved = current_state['currently_pushing']
        # if action_type_moved is None:
        #     return False
        # print("action type moved", action_type_moved)
        current_loc = current_state["agent_position"]

        new_loc = tuple(np.array(current_loc) + np.array(action))
        if (
            new_loc[0] < self.x_min
            or new_loc[0] >= self.x_max
            or new_loc[1] < self.y_min
            or new_loc[1] >= self.y_max
        ):
            return False

        return True

    def step_given_state(
        self, input_state, action, execute_policy: bool = False
    ):
        """Perform a step using input state and action.

        ::inputs:
            ::input_state:
            ::action:
            ::execute_policy: True when executing final, learned policy.
        """
        step_cost = -0.1
        current_state = copy.deepcopy(input_state)
        current_state["objects"] = input_state["objects"]
        current_state["holding"] = input_state["holding"]

        if self.is_done_given_state(current_state):
            step_reward = 0.0
            return current_state, step_reward, True

        if action in self.directions:
            if self.is_valid_push(current_state, action) is False:
                step_reward = step_cost
                return current_state, step_reward, False

        if action == PICKUP:
            # If not holding anything:
            if current_state["holding"] is None:
                # Re-order the objects so we see them all:
                objs = sample(self.objects, len(self.objects))
                for o in objs:
                    # Check if there is an object to pick up:
                    if current_state["agent_position"] == o.position:
                        # current_state["holding"] = o

                        # TODO: This check might be a hack:
                        if o == self.target_object:
                            current_state["holding"] = o

                step_reward = step_cost
                return current_state, step_reward, False

            else:
                step_reward = step_cost
                return current_state, step_reward, False

        if action == PLACE:
            if current_state["holding"] is not None:
                holding_object = current_state["holding"]
                if current_state["agent_position"] == self.g1:
                    if current_state["objects_in_g2"] is not str(
                        current_state["holding"]
                    ):
                        current_state["objects_in_g1"] = str(
                            current_state["holding"]
                        )
                        current_state["holding"] = None
                        step_reward = step_cost
                        done = self.is_done_given_state(current_state)
                        if (
                            self.target_object == holding_object
                            and self.target_goal == "g1"
                        ):
                            step_reward += self.correct_target_reward
                            return current_state, step_reward, done

                elif current_state["agent_position"] == self.g2:
                    if current_state["objects_in_g1"] is not str(
                        current_state["holding"]
                    ):
                        current_state["objects_in_g2"] = str(
                            current_state["holding"]
                        )
                        current_state["holding"] = None
                        step_reward = step_cost
                        done = self.is_done_given_state(current_state)
                        if (
                            self.target_object == holding_object
                            and self.target_goal == "g2"
                        ):
                            step_reward += self.correct_target_reward
                            return current_state, step_reward, done

                step_reward = step_cost
                return current_state, step_reward, False

            else:
                step_reward = step_cost
                return current_state, step_reward, False

        current_loc = current_state["agent_position"]
        new_loc = tuple(np.array(current_loc) + np.array(action))

        if execute_policy:
            if (
                current_state["holding"] is not None
                and action in self.directions
            ):
                current_state["holding"].move(action=action)

        current_state["agent_position"] = new_loc
        step_reward = step_cost
        done = self.is_done_given_state(current_state)

        return current_state, step_reward, done

    def state_to_tuple(self, current_state):
        """Convert current_state to tuple."""
        current_state_tup = []
        current_state_tup.append(
            ("agent_position", current_state["agent_position"])
        )
        current_state_tup.append(("holding", current_state["holding"]))
        current_state_tup.append(
            ("objects_in_g1", current_state["objects_in_g1"])
        )
        current_state_tup.append(
            ("objects_in_g2", current_state["objects_in_g2"])
        )
        current_state_tup.append(
            ("objects", tuple(x for x in current_state["objects"]))
        )

        return tuple(current_state_tup)

    def tuple_to_state(self, current_state_tup):
        """Convert current_state to tuple."""
        current_state_tup = list(current_state_tup)
        current_state = {}
        current_state["agent_position"] = current_state_tup[0][1]
        current_state["holding"] = current_state_tup[1][1]
        current_state["objects_in_g1"] = current_state_tup[2][1]
        current_state["objects_in_g2"] = current_state_tup[3][1]
        current_state["objects"] = current_state_tup[4][1]
        # current_state['square_positions'] = current_state_tup[1][1]
        # current_state['triangle_positions'] = current_state_tup[2][1]
        # current_state['g1'] = current_state_tup[6][1]
        # current_state['g2'] = current_state_tup[7][1]
        # current_state['target_object'] = current_state_tup[8][1]
        # current_state['target_goal'] = current_state_tup[9][1]

        return current_state

    def enumerate_states(self):
        self.reset()

        actions = self.possible_single_actions
        # Create directional graph to represent all states:
        G = nx.DiGraph()

        visited_states = set()

        # stack = [copy.deepcopy(self.current_state)]
        stack = [self.current_state]

        while stack:
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)
            # print("new_state_tup", state_tup)

            # If state has not been visited,
            # add it to the set of visited states:
            if state_tup not in visited_states:
                visited_states.add(state_tup)

            # get the neighbors of this state by looping through possible actions
            # actions = self.get_possible_actions_in_state(state)
            # print("POSSIBLE actions", actions)
            for idx, action in enumerate(actions):
                # print("action", action)
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True

                else:
                    next_state, team_reward, done = self.step_given_state(
                        state, action
                    )
                # print("state", state)
                # print("action", action)
                # print("team_reward", team_reward)
                # print("done", done)
                # print("next_state", next_state)

                # if done:
                #     print("DONE")
                #     print("team_reward", team_reward)
                #
                #     print("state", state)
                #     print("next_state", next_state)
                #     print("action", action)
                #     print()
                #     team_reward += 10

                new_state_tup = self.state_to_tuple(next_state)
                # print("new_state_tup", new_state_tup)
                # print("new_state_tup in visited_states = ", new_state_tup in visited_states)
                # print()
                # pdb.set_trace()

                if new_state_tup not in visited_states:
                    # stack.append(copy.deepcopy(next_state))
                    stack.append(next_state)

                # add edge to graph from current state to new state with weight equal to reward
                # if state_tup == new_state_tup:
                G.add_edge(
                    state_tup, new_state_tup, weight=team_reward, action=action
                )
                # if state == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
                #     el = G.out_edges(state_tup, data=True)
                #     print("len el", len(el))
                #     if action == (1,0):
                #         pdb.set_trace()
                #     G.add_edge(state_tup, new_state_tup, weight=team_reward, action=str(action))
                #
                #     el = G.out_edges(state_tup, data=True)
                #     print("new len el", len(el))
                #     print("el", el)
                #     print("action", action)
                #     print()

                # if state_tup == new_state_tup:
                #     pdb.set_trace()
                # if state_tup != new_state_tup:
                #     G.add_edge(state_tup, new_state_tup, weight=team_reward, action=action)
                # if state_tup == new_state_tup:
                #     if self.is_done_given_state(state) is False:
                #         G.add_edge(state_tup, new_state_tup, weight=-200, action=action)
                #     else:
                #         G.add_edge(state_tup, new_state_tup, weight=0, action=action)
                # pdb.set_trace()
        # pdb.set_trace()
        states = list(G.nodes)
        # print("NUMBER OF STATES", len(state
        idx_to_state = {i: state for i, state in enumerate(states)}
        state_to_idx = {state: i for i, state in idx_to_state.items()}

        # pdb.set_trace()
        action_to_idx = {action: i for i, action in enumerate(actions)}
        idx_to_action = {i: action for i, action in enumerate(actions)}

        # construct transition matrix and reward matrix of shape [# states, # states, # actions] based on graph
        transition_mat = np.zeros([len(states), len(states), len(actions)])
        reward_mat = np.zeros([len(states), len(actions)])

        for i in range(len(states)):
            # get all outgoing edges from current state
            # edges = G.out_edges(states[i], data=True)
            # if self.tuple_to_state(idx_to_state[i]) == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
            #     edges = G.out_edges(states[i], data=True)
            #     print("edges= ", edges)
            #     pdb.set_trace()
            state = self.tuple_to_state(idx_to_state[i])
            for action_idx_i in range(len(actions)):
                action = idx_to_action[action_idx_i]
                if self.is_done_given_state(state):
                    team_reward = 0
                    next_state = state
                    done = True

                else:
                    next_state, team_reward, done = self.step_given_state(
                        state, action
                    )
                # for edge in edges:
                # get index of action in action_idx
                # pdb.set_trace()
                # action_idx_i = action_to_idx[edge[2]['action']]
                # get index of next state in node list
                # next_state_i = states.index(edge[1])
                try:
                    next_state_i = state_to_idx[
                        self.state_to_tuple(next_state)
                    ]
                except KeyError:
                    breakpoint()
                # add edge to transition matrix
                # if i == next_state_i:
                #     reward_mat[i, action_idx_i] = -200
                # else:
                #     reward_mat[i, action_idx_i] = edge[2]['weight']
                #     transition_mat[i, next_state_i, action_idx_i] = 0.0
                #
                # else:
                transition_mat[i, next_state_i, action_idx_i] = 1.0
                # reward_mat[i, action_idx_i] = edge[2]['weight']
                reward_mat[i, action_idx_i] = team_reward
                # pdb.set_trace()
                # if idx_to_action[action_idx_i] == (0, 1) and self.tuple_to_state(idx_to_state[i]) == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
                #     # reward_mat[i, action_idx_i] = 0.0
                #     pdb.set_trace()
                # if self.tuple_to_state(idx_to_state[i]) == {'grid': {(1, 1): (3, 3)}, 'exit': False, 'orientation': 0}:
                #     edges = G.out_edges(states[i], data=True)
                #     print("edges= ", edges)
                #     print("action", idx_to_action[action_idx_i])
                # pdb.set_trace()

        # check that for each state and action pair, the sum of the transition probabilities is 1 (or 0 for terminal states)
        # for i in range(len(states)):
        #     for j in range(len(actions)):
        #         print("np.sum(transition_mat[i, :, j])", np.sum(transition_mat[i, :, j]))
        #         print("np.sum(transition_mat[i, :, j]", np.sum(transition_mat[i, :, j]))
        # assert np.isclose(np.sum(transition_mat[i, :, j]), 1.0) or np.isclose(np.sum(transition_mat[i, :, j]),
        #                                                                       0.0)
        (
            self.transitions,
            self.rewards,
            self.state_to_idx,
            self.idx_to_action,
            self.idx_to_state,
            self.action_to_idx,
        ) = (
            transition_mat,
            reward_mat,
            state_to_idx,
            idx_to_action,
            idx_to_state,
            action_to_idx,
        )

        # print("number of states", len(states))
        # print("number of actions", len(actions))
        # print("transition matrix shape", transition_mat.shape)
        return (
            transition_mat,
            reward_mat,
            state_to_idx,
            idx_to_action,
            idx_to_state,
            action_to_idx,
        )

    def vectorized_vi(self):
        # def spatial_environment(transitions, rewards, epsilson=0.0001, gamma=0.99, maxiter=10000):
        """
        Parameters
        ----------
            transitions : array_like
                Transition probability matrix. Of size (# states, # states, # actions).
            rewards : array_like
                Reward matrix. Of size (# states, # actions).
            epsilson : float, optional
                The convergence threshold. The default is 0.0001.
            gamma : float, optional
                The discount factor. The default is 0.99.
            maxiter : int, optional
                The maximum number of iterations. The default is 10000.
        Returns
        -------
            value_function : array_like
                The value function. Of size (# states, 1).
            pi : array_like
                The optimal policy. Of size (# states, 1).
        """
        n_states = self.transitions.shape[0]
        n_actions = self.transitions.shape[2]

        # initialize value function
        pi = np.zeros((n_states, 1))
        vf = np.zeros((n_states, 1))
        Q = np.zeros((n_states, n_actions))
        policy = {}

        for i in range(self.maxiter):
            # initalize delta
            delta = 0
            # perform Bellman update
            for s in range(n_states):
                # store old value function
                old_v = vf[s].copy()
                # compute new value function
                Q[s] = np.sum(
                    (self.rewards[s] + self.gamma * vf)
                    * self.transitions[s, :, :],
                    0,
                )
                vf[s] = np.max(
                    np.sum(
                        (self.rewards[s] + self.gamma * vf)
                        * self.transitions[s, :, :],
                        0,
                    )
                )
                # compute delta
                delta = np.max((delta, np.abs(old_v - vf[s])[0]))
            # check for convergence
            if delta < self.epsilson:
                break

        # compute optimal policy
        for s in range(n_states):
            pi[s] = np.argmax(np.sum(vf * self.transitions[s, :, :], 0))
            policy[s] = Q[s, :]

        self.vf = vf
        self.pi = pi
        self.policy = policy
        return vf, pi

    def rollout_full_game_joint_optimal(self):
        self.reset()
        done = False
        total_reward = 0

        iters = 0
        game_results = []

        while not done:
            iters += 1

            current_state_tup = self.state_to_tuple(self.current_state)

            try:
                state_idx = self.state_to_idx[current_state_tup]
            except KeyError:
                breakpoint()

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            game_results.append((self.current_state, action))

            next_state, team_rew, done = self.step_given_state(
                self.current_state, action, execute_policy=True
            )

            self.current_state = next_state
            self.visualize(state=self.current_state)

            total_reward += team_rew

            if iters > 40:
                break

        return total_reward, game_results

    def compute_optimal_performance(self):
        print("Enumerating states...")
        self.enumerate_states()
        print("Vectorized vi...")
        self.vectorized_vi()

        optimal_rew, game_results = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results


if __name__ == "__main__":
    # [obj A placed in G1, object A in G2, B in G1, B in G2]:
    # reward_weights = [1, -1, -1, 1]

    map_config = {
        "agent_position": (0, 0),  # The agent's start position
        "dimensions": [20, 20],
        "g1": (4, 10),
        "g2": (4, 0),
    }

    mug = Mug()
    fork = Fork()
    object_list = [mug, fork]
    reward = {
        "target_object": fork,
        "target_goal": "g1",
    }

    game = Gridworld(
        map_config=map_config, objects=object_list, reward_dict=reward
    )
    optimal_rew, game_results = game.compute_optimal_performance()
