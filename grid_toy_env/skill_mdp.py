import copy
from itertools import product

import networkx as nx
import numpy as np

SQUARE = "square"
TRIANGLE = "triangle"
PICKUP = "pickup"
PLACE = "place"


class Gridworld:
    """TODOs.

    - Add orientation.
    -
    """

    def __init__(self, initial_config, reward_dict):
        # TODO:
        # 1.) Visualize start state w/ binary map & objects
        # 2.) Load pngs of objects; add their arrays to the positions in the binary map
        # 3.) No obstacles in map to begin with
        # 4.) Skill execution: straight line interp. (delta x delta y) first to waypoint and then to goal,
        #     use translation mat. and move the object mat.
            # Need to store object locations during execution
        self.reward_dict = reward_dict
        self.initial_config = initial_config

        self.target_object = reward_dict["target_object"]
        self.target_goal = reward_dict["target_goal"]

        # state = {
        #     'start_pos': (0, 0),
        #     'g1': (4, 4),
        #     'g2': (4, 0),
        #     'square_positions': [(1, 3), (2, 0)],
        #     'triangle_positions': [(3, 2), (2, 4)],
        # }
        self.square_positions = initial_config["square_positions"]
        self.triangle_positions = initial_config["triangle_positions"]
        self.start_pos = initial_config["start_pos"]
        self.g1 = initial_config["g1"]
        self.g2 = initial_config["g2"]
        self.objects_in_g1 = None
        self.objects_in_g2 = None

        self.set_env_limits()

        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # get possible joint actions and actions
        self.possible_single_actions = self.make_actions_list()
        # print("possible single actions", self.possible_single_actions)

        self.current_state = self.create_initial_state()
        # self.reset()

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
        self.maxiter = 10000

        # self.step_cost = -0.01
        # self.push_switch_cost = -0.05

        # self.step_cost = reward_weights[-2]
        # self.push_switch_cost = reward_weights[-1]

        self.num_features = 4
        self.correct_target_reward = 10

    def make_actions_list(self):
        actions_list = []
        actions_list.extend(self.directions)
        actions_list.append(PICKUP)
        actions_list.append(PLACE)
        return actions_list

    def set_env_limits(self):
        # set environment limits
        self.x_min = 0
        self.x_max = 5
        self.y_min = 0
        self.y_max = 5

        self.all_coordinate_locations = list(
            product(
                range(self.x_min, self.x_max), range(self.y_min, self.y_max)
            )
        )

    def reset(self):
        self.current_state = self.create_initial_state()

    def create_initial_state(self):
        # create dictionary of object location to object type and picked up state
        state = {}
        state["pos"] = copy.deepcopy(self.start_pos)
        # state['square_positions'] = copy.deepcopy(self.square_positions)
        # state['triangle_positions'] = copy.deepcopy(self.triangle_positions)
        state["holding"] = None
        state["objects_in_g1"] = None
        state["objects_in_g2"] = None
        # state['g1'] = copy.deepcopy(self.g1)
        # state['g2'] = copy.deepcopy(self.g2)
        # state['target_object'] = self.target_object
        # state['target_goal'] = self.target_goal

        return state

    def is_done_given_state(self, current_state):
        # check if player at exit location
        # print("current state", current_state)
        if (
            current_state["objects_in_g1"] != None
            or current_state["objects_in_g2"] != None
        ):
            return True

        return False

    def is_valid_push(self, current_state, action):
        # action_type_moved = current_state['currently_pushing']
        # if action_type_moved is None:
        #     return False
        # print("action type moved", action_type_moved)
        current_loc = current_state["pos"]

        new_loc = tuple(np.array(current_loc) + np.array(action))
        if (
            new_loc[0] < self.x_min
            or new_loc[0] >= self.x_max
            or new_loc[1] < self.y_min
            or new_loc[1] >= self.y_max
        ):
            return False

        # if new_loc in current_state['grid'].values() and new_loc != current_loc:
        #     return False

        return True

    def step_given_state(self, input_state, action):
        step_cost = -0.1
        current_state = copy.deepcopy(input_state)

        # print("action", action)
        # check if action is exit
        # print("action in step", action)

        if self.is_done_given_state(current_state):
            step_reward = 0
            return current_state, step_reward, True

        if action in self.directions:
            if self.is_valid_push(current_state, action) is False:
                step_reward = step_cost
                return current_state, step_reward, False

        if action == PICKUP:
            # if not holding anything
            if current_state["holding"] is None:
                # check if there is an object to pick up
                if current_state["pos"] in self.square_positions:
                    current_state["holding"] = "square"
                elif current_state["pos"] in self.triangle_positions:
                    current_state["holding"] = "triangle"

                step_reward = step_cost
                return current_state, step_reward, False

            else:
                step_reward = step_cost
                return current_state, step_reward, False

        if action == PLACE:
            if current_state["holding"] is not None:
                holding_object = current_state["holding"]
                if current_state["pos"] == self.g1:
                    current_state["objects_in_g1"] = current_state["holding"]
                    current_state["holding"] = None
                    step_reward = step_cost
                    done = self.is_done_given_state(current_state)
                    if (
                        self.target_object == holding_object
                        and self.target_goal == "g1"
                    ):
                        step_reward += self.correct_target_reward
                        return current_state, step_reward, done
                elif current_state["pos"] == self.g2:
                    current_state["objects_in_g2"] = current_state["holding"]
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

        current_loc = current_state["pos"]
        # print("current loc", current_loc)
        # print("action", action)
        new_loc = tuple(np.array(current_loc) + np.array(action))
        current_state["pos"] = new_loc
        step_reward = step_cost
        done = self.is_done_given_state(current_state)

        return current_state, step_reward, done

    def state_to_tuple(self, current_state):
        # convert current_state to tuple
        current_state_tup = []
        current_state_tup.append(("pos", current_state["pos"]))
        current_state_tup.append(("holding", current_state["holding"]))
        current_state_tup.append(
            ("objects_in_g1", current_state["objects_in_g1"])
        )
        current_state_tup.append(
            ("objects_in_g2", current_state["objects_in_g2"])
        )

        return tuple(current_state_tup)

    def tuple_to_state(self, current_state_tup):
        # convert current_state to tuple
        current_state_tup = list(current_state_tup)
        current_state = {}
        current_state["pos"] = current_state_tup[0][1]
        # current_state['square_positions'] = current_state_tup[1][1]
        # current_state['triangle_positions'] = current_state_tup[2][1]
        current_state["holding"] = current_state_tup[1][1]
        current_state["objects_in_g1"] = current_state_tup[2][1]
        current_state["objects_in_g2"] = current_state_tup[3][1]
        # current_state['g1'] = current_state_tup[6][1]
        # current_state['g2'] = current_state_tup[7][1]
        # current_state['target_object'] = current_state_tup[8][1]
        # current_state['target_goal'] = current_state_tup[9][1]

        return current_state

    def enumerate_states(self):
        self.reset()

        actions = self.possible_single_actions
        # print("actions", actions)
        # pdb.set_trace()
        # create directional graph to represent all states
        G = nx.DiGraph()

        visited_states = set()

        stack = [copy.deepcopy(self.current_state)]

        while stack:
            # print("len visited_states", len(visited_states))
            # print("len stack", len(stack))
            # print("visited_states", visited_states)
            state = stack.pop()

            # convert old state to tuple
            state_tup = self.state_to_tuple(state)
            # print("new_state_tup", state_tup)

            # if state has not been visited, add it to the set of visited states
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
                    stack.append(copy.deepcopy(next_state))

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
                next_state_i = state_to_idx[self.state_to_tuple(next_state)]
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
        sum_feature_vector = np.zeros(4)

        # self.render(self.current_state, iters)
        while not done:
            iters += 1

            current_state_tup = self.state_to_tuple(self.current_state)

            state_idx = self.state_to_idx[current_state_tup]

            action_distribution = self.policy[state_idx]
            action = np.argmax(action_distribution)
            action = self.idx_to_action[action]

            # print("state", self.current_state)
            # print("action", action)

            game_results.append((self.current_state, action))

            next_state, team_rew, done = self.step_given_state(
                self.current_state, action
            )

            # print("next_state", next_state)
            self.current_state = next_state
            # print("new state", self.current_state)
            # print("team_rew", team_rew)
            # print("done", done)
            # print()
            # self.render(self.current_state, iters)

            total_reward += team_rew

            if iters > 40:
                break

        # self.save_rollouts_to_video()

        return total_reward, game_results

    def compute_optimal_performance(self):
        # print("start enumerating states")
        self.enumerate_states()
        # print("done enumerating states")
        # print("start vi")
        self.vectorized_vi()
        # print("done vi")

        optimal_rew, game_results = self.rollout_full_game_joint_optimal()
        return optimal_rew, game_results


if __name__ == "__main__":
    # reward_weights = [1, -1, -1, 1]  # [obj A placed in G1, object A in G2, B in G1, B in G2]

    state = {
        "start_pos": (0, 0),
        "g1": (4, 4),
        "g2": (4, 0),
        "square_positions": [(1, 3), (2, 0)],
        "triangle_positions": [(3, 2), (2, 4)],
    }
    reward = {
        "target_object": SQUARE,
        "target_goal": "g1",
    }

    game = Gridworld(state, reward)
    optimal_rew, game_results = game.compute_optimal_performance()
    # pdb.set_trace()
