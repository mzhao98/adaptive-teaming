import copy
import math
import pdb

import numpy as np
import pickle
import sys
import networkx as nx
from itertools import product
import itertools
import matplotlib.pyplot as plt

from skill_mdp import Gridworld
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import json

import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
DEVICE = 'cpu'
from sklearn import metrics
from scipy.special import softmax

SQUARE = 'square'
TRIANGLE = 'triangle'
PICKUP = 'pickup'
PLACE = 'place'

G1 = 'g1'
G2 = 'g2'

ASK_PREF = 'ask_pref'
ASK_DEMO = 'ask_demo'
possible_actions = [(SQUARE, G1), (SQUARE, G2), (TRIANGLE, G1), (TRIANGLE, G2),
                                 (ASK_PREF, SQUARE), (ASK_PREF, TRIANGLE), (ASK_DEMO, SQUARE), (ASK_DEMO, TRIANGLE)]
action_to_text = {(SQUARE, G1): 'square g1', (SQUARE, G2): 'square g2', (TRIANGLE, G1): 'triangle g1', (TRIANGLE, G2): 'triangle g2',
                                 (ASK_PREF, SQUARE): 'ask pref square', (ASK_PREF, TRIANGLE): 'ask pref triangle', (ASK_DEMO, SQUARE): 'ask demo square', (ASK_DEMO, TRIANGLE): 'ask demo triangle'}


class BC_Policy(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(24, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 6)
        )


    def forward(self, x):
        output = self.net(x)
        return output


class Interaction_MDP():
    def __init__(self, initial_config, true_human_reward_weights):
        self.true_human_reward_weights = true_human_reward_weights
        self.initial_config = initial_config

        self.possible_actions = [(SQUARE, G1), (SQUARE, G2), (TRIANGLE, G1), (TRIANGLE, G2),
                                 (ASK_PREF, SQUARE), (ASK_PREF, TRIANGLE), (ASK_DEMO, SQUARE), (ASK_DEMO, TRIANGLE)]


        self.setup_bc_policy()
        self.beliefs = {(1, -10, -10, 1): 0.5, (-10, 1, 1, -10): 0.5}
        self.possible_actions_to_prob_success = {
            (SQUARE, G1): 0.0,
            (SQUARE, G2): 0.0,
            (TRIANGLE, G1): 1.0,
            (TRIANGLE, G2): 1.0,
            (ASK_PREF, SQUARE): 1.0,
            (ASK_PREF, TRIANGLE): 1.0,
            (ASK_DEMO, SQUARE): 1.0,
            (ASK_DEMO, TRIANGLE): 1.0
        }


    def setup_bc_policy(self):
        self.seen_demos = []
        self.randomly_initialize_bc_data()

        self.bc_network = BC_Policy()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.bc_network.parameters(), lr=1e-3)


    def train_bc_policy(self):
        X = torch.tensor(self.bc_data['X'], dtype=torch.float32)
        Y = torch.tensor(self.bc_data['Y'], dtype=torch.long)
        # train policy
        for epoch in range(100):
            self.optimizer.zero_grad()
            output = self.bc_network(X)
            loss = self.loss_function(output, torch.argmax(Y, dim=1))
            loss.backward()
            self.optimizer.step()
            # print("epoch", epoch, "loss", loss.item())


    def is_done_given_interaction_state(self, interaction_state):
        if len(interaction_state['square_positions']) == 0 and len(interaction_state['triangle_positions']) == 0:
            return True
        else:
            return False


    def rollout_interaction(self):
        current_state = copy.deepcopy(self.initial_config)
        interaction_result = []
        total_reward = 0
        iters = 0
        while not self.is_done_given_interaction_state(current_state):
            iters += 1
            action = self.get_action(current_state)
            init_beliefs, init_possible_actions_to_prob_success, init_seen_demos = copy.deepcopy(
                self.beliefs), copy.deepcopy(self.possible_actions_to_prob_success), copy.deepcopy(self.seen_demos)
            next_state, seen_demos, possible_actions_to_prob_success, beliefs, reward, done = self.step(current_state, action, init_beliefs, init_possible_actions_to_prob_success, init_seen_demos)
            interaction_result.append((current_state, reward, done))
            total_reward += reward
            current_state = next_state
            action_text = action_to_text[action]
            print("current state", current_state)
            # print("INIT possible_actions_to_prob_success", init_possible_actions_to_prob_success)
            # print("INIT beliefs", init_beliefs)
            print("action", action_text)

            # print("NEW possible_actions_to_prob_success", possible_actions_to_prob_success)
            # print("NEW beliefs", beliefs)
            print("reward", reward)
            print("done", done)
            print("next state", next_state)
            print()
            self.beliefs, self.possible_actions_to_prob_success, self.seen_demos = beliefs, possible_actions_to_prob_success, seen_demos
            if iters > 7:
                break


        return total_reward, interaction_result


    def get_action(self, current_state):
        best_reward = -1000
        best_action = None
        # print("getting action")
        for action in self.possible_actions:
            init_beliefs, init_possible_actions_to_prob_success, init_seen_demos = copy.deepcopy(self.beliefs), copy.deepcopy(self.possible_actions_to_prob_success), copy.deepcopy(self.seen_demos)
            next_state, seen_demos, possible_actions_to_prob_success, beliefs, reward, done = self.hypothetical_step_under_beliefs(current_state, action, init_beliefs, init_possible_actions_to_prob_success, init_seen_demos)
            # if action == (TRIANGLE, G1):
            #     pdb.set_trace()
            if action[0] == ASK_PREF:
                best_next_action = None
                best_next_reward = -1000
                for next_action in self.possible_actions:
                    # if next_action[0] != action[1]:
                    #     continue
                    next_state, seen_demos, possible_actions_to_prob_success, beliefs, next_reward, done = self.hypothetical_step_under_beliefs(
                        current_state, next_action, beliefs, possible_actions_to_prob_success, seen_demos)
                    if next_reward > best_next_reward:
                        best_next_reward = next_reward
                        best_next_action = next_action
                reward = best_next_reward + reward
                # print("Action is ASK_PREF")
                # print("next action", best_next_action)
                # print("next state", next_state)
                # print("reward", next_reward)
                # print("beliefs", beliefs)
                # print("done", done)
                # print()


            if action[0] == ASK_DEMO:
                # next_next_state, seen_demos, possible_actions_to_prob_success, beliefs, reward, done = self.hypothetical_step_under_beliefs(
                #     current_state, action, beliefs, possible_actions_to_prob_success, seen_demos)
                best_next_action = None
                best_next_reward = -1000
                for next_action in self.possible_actions:
                    # if next_action[0] != action[1]:
                    #     continue
                    next_state, seen_demos, possible_actions_to_prob_success, beliefs, next_reward, done = self.hypothetical_step_under_beliefs(
                        current_state, next_action, beliefs, possible_actions_to_prob_success, seen_demos)
                    if next_reward > best_next_reward:
                        best_next_reward = next_reward
                        best_next_action = next_action
                reward = best_next_reward + reward
            # print("action", action_to_text[action])
            # print("reward", reward)
            # print()

                # print("INIT possible_actions_to_prob_success" , init_possible_actions_to_prob_success)
                # print("NEW possible_actions_to_prob_success" , possible_actions_to_prob_success)

            if reward > best_reward:
                best_reward = reward
                best_action = action
        return best_action



    def step_with_nn(self, input_state, action):
        current_state = copy.deepcopy(input_state)
        # if action is a robot movement
        if action in [(SQUARE, G1), (SQUARE, G2), (TRIANGLE, G1), (TRIANGLE, G2)]:
            target = {
                'target_object': action[0],
                'target_goal': action[1]
            }
            skill_state = {'pos': current_state['start_pos'],
                             'holding': None,
                             'objects_in_g1':None,
                             'objects_in_g2':None}
            state_vector = self.convert_state_to_vector(current_state, skill_state, target)
            X = torch.tensor([state_vector], dtype=torch.float32)
            # query policy
            output = self.bc_network(X).detach().numpy()[0]
            chosen_action_idx = np.argmax(output)
            chosen_action = self.idx_to_action[chosen_action_idx]
            print("chosen_action", chosen_action)

        # # if action is a human demo
        #     state_vectors_list = []
        #     actions_list = []
        #     for state, action in game_results:
        #         state_vector = self.convert_state_to_vector(current_state, state, target)
        #         state_vectors_list.append(state_vector)
        #         one_hot_action_idx = self.action_to_idx[action]
        #         one_hot_action = [0] * len(self.low_level_actions)
        #         one_hot_action[one_hot_action_idx] = 1
        #         actions_list.append(one_hot_action)
        #
        #     self.bc_data['X'].extend(state_vectors_list)
        #     self.bc_data['Y'].extend(actions_list)


    def step(self, input_state, action, init_beliefs, init_possible_actions_to_prob_success, init_seen_demos):
        seen_demos = copy.deepcopy(init_seen_demos)
        possible_actions_to_prob_success = copy.deepcopy(init_possible_actions_to_prob_success)
        beliefs = copy.deepcopy(init_beliefs)
        current_state = copy.deepcopy(input_state)
        query_cost = 0
        # if action is a robot movement
        if action in [(SQUARE, G1), (SQUARE, G2), (TRIANGLE, G1), (TRIANGLE, G2)]:
            acting_object = action[0]
            acting_goal = action[1]
            prob_success = possible_actions_to_prob_success[action]
            if prob_success == 1:
                if acting_object == SQUARE:
                    if len(current_state['square_positions']):
                        current_state['square_positions'].pop()
                        if acting_goal == G1:
                            current_state['objects_in_g1'].append(SQUARE)
                        else:
                            current_state['objects_in_g2'].append(SQUARE)

                else:
                    if len(current_state['triangle_positions']):
                        current_state['triangle_positions'].pop()
                        if acting_goal == G1:
                            current_state['objects_in_g1'].append(TRIANGLE)
                        else:
                            current_state['objects_in_g2'].append(TRIANGLE)
            query_cost += 10

        elif action in [(ASK_PREF, SQUARE), (ASK_PREF, TRIANGLE)]:
            query_cost += 0.1
            acting_object = action[1]
            for elem in beliefs:
                if tuple(self.true_human_reward_weights) == elem:
                    beliefs[elem] *= 1.0
                else:
                    beliefs[elem] *= 0.0
        elif action in [(ASK_DEMO, SQUARE), (ASK_DEMO, TRIANGLE)]:
            query_cost += 2
            acting_object = action[1]
            for elem in beliefs:
                if tuple(self.true_human_reward_weights) == elem:
                    beliefs[elem] *= 1.0
                else:
                    beliefs[elem] *= 0.0

            if acting_object == SQUARE:
                if self.true_human_reward_weights == [1, -10, -10, 1]:
                    acting_goal = G1
                else:
                    acting_goal = G2
            else:
                if self.true_human_reward_weights == [1, -10, -10, 1]:
                    acting_goal = G2
                else:
                    acting_goal = G1

            seen_demos.append((acting_object, acting_goal))
            # possible_actions_to_prob_success[(acting_object, acting_goal)] = 1.0
            possible_actions_to_prob_success[(acting_object, G1)] = 1.0
            possible_actions_to_prob_success[(acting_object, G2)] = 1.0

            if acting_object == SQUARE:
                if len(current_state['square_positions']):
                    current_state['square_positions'].pop()
                    if acting_goal == G1:
                        current_state['objects_in_g1'].append(SQUARE)
                    else:
                        current_state['objects_in_g2'].append(SQUARE)

            else:
                if len(current_state['triangle_positions']):
                    current_state['triangle_positions'].pop()
                    if acting_goal == G1:
                        current_state['objects_in_g1'].append(TRIANGLE)
                    else:
                        current_state['objects_in_g2'].append(TRIANGLE)


        feature_vector = self.convert_state_to_feature_vector(current_state)
        step_cost = 5
        reward = np.dot(feature_vector, self.true_human_reward_weights)-query_cost-step_cost
        done = self.is_done_given_interaction_state(current_state)

        # normalize beliefs
        beliefs_sum = sum(beliefs.values())
        for elem in beliefs:
            beliefs[elem] /= beliefs_sum
        # if done:
        #     reward += np.dot(feature_vector, self.true_human_reward_weights)
        return current_state,seen_demos, possible_actions_to_prob_success,beliefs,  reward, done


    def hypothetical_step_under_beliefs(self, input_state, action, init_beliefs, init_possible_actions_to_prob_success, init_seen_demos):
        seen_demos = copy.deepcopy(init_seen_demos)
        possible_actions_to_prob_success = copy.deepcopy(init_possible_actions_to_prob_success)
        beliefs = copy.deepcopy(init_beliefs)
        current_state = copy.deepcopy(input_state)
        query_cost = 0
        # if action is a robot movement
        if action in [(SQUARE, G1), (SQUARE, G2), (TRIANGLE, G1), (TRIANGLE, G2)]:
            acting_object = action[0]
            acting_goal = action[1]
            prob_success = possible_actions_to_prob_success[action]
            if prob_success==1:
                if acting_object == SQUARE:
                    if len(current_state['square_positions']):
                        current_state['square_positions'].pop()
                        if acting_goal == G1:
                            current_state['objects_in_g1'].append(SQUARE)
                        else:
                            current_state['objects_in_g2'].append(SQUARE)

                    else:
                        query_cost += 10

                else:
                    if len(current_state['triangle_positions']):
                        current_state['triangle_positions'].pop()
                        if acting_goal == G1:
                            current_state['objects_in_g1'].append(TRIANGLE)
                        else:
                            current_state['objects_in_g2'].append(TRIANGLE)
                    else:
                        query_cost += 10

            else:
                query_cost += 10

        elif action in [(ASK_PREF, SQUARE), (ASK_PREF, TRIANGLE)]:
            query_cost += 0.1
            acting_object = action[1]
            for elem in beliefs:
                if tuple(self.true_human_reward_weights) == elem:
                    beliefs[elem] = 1
                else:
                    beliefs[elem] = 0.0

        elif action in [(ASK_DEMO, SQUARE), (ASK_DEMO, TRIANGLE)]:
            query_cost += 2
            acting_object = action[1]
            for elem in beliefs:
                if tuple(self.true_human_reward_weights) == elem:
                    beliefs[elem] = 1
                else:
                    beliefs[elem] = 0.0

            if acting_object == SQUARE:
                if self.true_human_reward_weights == [1, -10, -10, 1]:
                    acting_goal = G1
                else:
                    acting_goal = G2
            else:
                if self.true_human_reward_weights == [1, -10, -10, 1]:
                    acting_goal = G2
                else:
                    acting_goal = G1

            seen_demos.append((acting_object, acting_goal))
            possible_actions_to_prob_success[(acting_object, G1)] = 1.0
            possible_actions_to_prob_success[(acting_object, G2)] = 1.0

            if acting_object == SQUARE:
                if len(current_state['square_positions']):
                    current_state['square_positions'].pop()
                    if acting_goal == G1:
                        current_state['objects_in_g1'].append(SQUARE)
                    else:
                        current_state['objects_in_g2'].append(SQUARE)
                else:
                    query_cost += 10

            else:
                if len(current_state['triangle_positions']):
                    current_state['triangle_positions'].pop()
                    if acting_goal == G1:
                        current_state['objects_in_g1'].append(TRIANGLE)
                    else:
                        current_state['objects_in_g2'].append(TRIANGLE)
                else:
                    query_cost += 10


        feature_vector = self.convert_state_to_feature_vector(current_state)
        step_cost = 5
        hyps = [list(x) for x in list(beliefs.keys())]
        reward_hyp1 = np.dot(feature_vector, hyps[0]) - query_cost - step_cost
        reward_hyp2 = np.dot(feature_vector, hyps[1]) - query_cost - step_cost
        # print("action", action_to_text[action])
        # print("reward_hyp1", reward_hyp1)
        # print("reward_hyp2", reward_hyp2)
        # print("beliefs", beliefs)
        reward = (reward_hyp1 * beliefs[tuple(hyps[0])]) + (reward_hyp2 * beliefs[tuple(hyps[1])])
        # print("reward", reward)
        done = self.is_done_given_interaction_state(current_state)
        # if done:
        #     reward += np.dot(feature_vector, self.true_human_reward_weights)
        return current_state,seen_demos, possible_actions_to_prob_success, beliefs,  reward, done

    def convert_state_to_feature_vector(self, state):
        feature_vector = [0] * 4
        for item in state['objects_in_g1']:
            if item == SQUARE:
                feature_vector[0] += 1
            else:
                feature_vector[2] += 1

        for item in state['objects_in_g2']:
            if item == SQUARE:
                feature_vector[1] += 1
            else:
                feature_vector[3] += 1
        return feature_vector


    def randomly_initialize_bc_data(self, n_start=0):
        self.bc_data = {}

        # list all positions between (0,0) and (5,5)
        all_positions = list(product(range(5), range(5)))
        all_positions_indices = list(range(len(all_positions)))
        all_objects = [SQUARE, TRIANGLE]
        all_goals = [G1, G2]

        self.low_level_actions = [(0, 1), (0, -1), (1, 0), (-1, 0), PICKUP, PLACE]
        # get idx_to_action and action_to_idx
        self.idx_to_action = {}
        self.action_to_idx = {}
        for idx, action in enumerate(self.low_level_actions):
            self.idx_to_action[idx] = action
            self.action_to_idx[action] = idx

        # generate n_start random initial configurations
        all_X = []
        all_Y = []
        for instance in range(n_start):
            initial_config = {
                'start_pos': (0, 0),
                'g1': (4, 4),
                'g2': (4, 0),
                'square_positions': [all_positions[np.random.choice(all_positions_indices)], all_positions[np.random.choice(all_positions_indices)]],
                'triangle_positions': [all_positions[np.random.choice(all_positions_indices)], all_positions[np.random.choice(all_positions_indices)]],
            }
            target = {
                'target_object': np.random.choice(all_objects),
                'target_goal': np.random.choice(all_goals)
            }

            game = Gridworld(initial_config, target)
            _, game_results = game.compute_optimal_performance()



            # print("game_results", game_results)
            # convert game_results into a list of state vectors
            state_vectors_list = []
            actions_list = []
            for state, action in game_results:
                state_vector = self.convert_state_to_vector(initial_config, state, target)
                state_vectors_list.append(state_vector)
                one_hot_action_idx = self.action_to_idx[action]
                one_hot_action = [0] * len(self.low_level_actions)
                one_hot_action[one_hot_action_idx] = 1
                actions_list.append(one_hot_action)

            all_X.extend(state_vectors_list)
            all_Y.extend(actions_list)

        self.bc_data['X'] = all_X
        self.bc_data['Y'] = all_Y



    def convert_state_to_vector(self, initial_config, state, target):
        # object to one-hot vector
        self.obj_to_one_hot = {SQUARE: [1, 0, 0], TRIANGLE: [0, 1, 0], None: [0, 0, 1]}
        # goal to one-hot vector
        self.goal_to_one_hot = {G1: [1, 0], G2: [0, 1]}

        state_vector = []
        # add target object and goal
        state_vector.extend(self.obj_to_one_hot[target['target_object']])
        state_vector.extend(self.goal_to_one_hot[target['target_goal']])

        # add positions of objects
        state_vector.extend(list(initial_config['square_positions'][0]))
        state_vector.extend(list(initial_config['square_positions'][1]))
        state_vector.extend(list(initial_config['triangle_positions'][0]))
        state_vector.extend(list(initial_config['triangle_positions'][1]))

        # add positions of agent
        state_vector.extend(list(state['pos']))
        state_vector.extend(self.obj_to_one_hot[state['holding']])
        state_vector.extend(self.obj_to_one_hot[state['objects_in_g1']])
        state_vector.extend(self.obj_to_one_hot[state['objects_in_g2']])

        return state_vector


if __name__ == '__main__':
    true_human_reward_weights = [1, -10, -10, 1]  # [obj A(square) placed in G1, object A(square) in G2, B(triangle) in G1, B in G2]

    env_config = {
        'start_pos': (0,0),
        'g1': (4,4),
        'g2': (4,0),
        'square_positions': [(1,3), (2,0)],
        'triangle_positions': [(3,2), (2,4)],
        'objects_in_g1': [],
        'objects_in_g2': [],
    }


    game = Interaction_MDP(env_config, true_human_reward_weights)
    total_reward, interaction_result = game.rollout_interaction()
    print()
    print("total_reward", total_reward)
    # print("interaction_result", interaction_result)

