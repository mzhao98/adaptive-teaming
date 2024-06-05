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
import os
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF



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
            (TRIANGLE, G1): 0.0,
            (TRIANGLE, G2): 0.0,
            (ASK_PREF, SQUARE): 1.0,
            (ASK_PREF, TRIANGLE): 1.0,
            (ASK_DEMO, SQUARE): 1.0,
            (ASK_DEMO, TRIANGLE): 1.0
        }

        self.init_obj_positions = {'square_positions': [self.all_positions[np.random.choice(self.all_positions_indices)],
                                                        self.all_positions[np.random.choice(self.all_positions_indices)]],
                                'triangle_positions': [self.all_positions[np.random.choice(self.all_positions_indices)],
                                                       self.all_positions[np.random.choice(self.all_positions_indices)]]}

        self.start_pos = (0,0)


    def setup_bc_policy(self):
        self.seen_demos = []
        self.randomly_initialize_bc_data()

        self.bc_network = BC_Policy()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.bc_network.parameters(), lr=1e-3)

        # if 'bc_policy.pt' exists, load it
        if 'bc_policy.pt' in os.listdir():
            self.bc_network.load_state_dict(torch.load('bc_policy.pt'))
            print("loaded policy")
        # self.train_bc_policy()


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

        # save policy
        torch.save(self.bc_network.state_dict(), 'bc_policy.pt')
        print("updated policy")

        # print accuracy on training set
        output = self.bc_network(X)
        print("accuracy on training set", metrics.accuracy_score(torch.argmax(Y, dim=1).numpy(),
                                                                 torch.argmax(output, dim=1).detach().numpy()))


        gp_X = []
        gp_Y = []
        for i in range(len(self.bc_data['X'])):
            gp_X.append(self.bc_data['X'][i])
            gp_Y.append(1)
            for noise_instances in range(10):
                # add noise to data
                noisy_X = np.array(self.bc_data['X'][i]) + np.random.normal(1, 4, np.array(self.bc_data['X'][i]).shape)
                gp_X.append(noisy_X)
                gp_Y.append(0)

        gp_X = np.array(gp_X)
        gp_Y = np.array(gp_Y)
        self.kernel = 1.0 * RBF(1.0)
        self.gpc = GaussianProcessClassifier(kernel=self.kernel,random_state = 0).fit(gp_X, gp_Y)
        score = self.gpc.score(gp_X, gp_Y)
        print("score", score)
        # self.gpc.predict_proba(X[:2, :])


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
        next_state = None
        while not self.is_done_given_interaction_state(current_state):
            iters += 1
            action = self.get_action(current_state)

            print("current state", current_state)
            print("SHOULD BE THE SAME AS NEXT state", next_state)
            init_beliefs, init_possible_actions_to_prob_success, init_seen_demos = copy.deepcopy(
                self.beliefs), copy.deepcopy(self.possible_actions_to_prob_success), copy.deepcopy(self.seen_demos)
            next_state, seen_demos, possible_actions_to_prob_success, beliefs, reward, done = self.step(current_state, action, init_beliefs, init_possible_actions_to_prob_success, init_seen_demos)
            interaction_result.append((current_state, reward, done))
            total_reward += reward
            current_state = next_state
            action_text = action_to_text[action]

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
            if iters > 15:
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
        curr_state = copy.deepcopy(input_state)
        query_cost = 0
        # if action is a robot movement
        if action in [(SQUARE, G1), (SQUARE, G2), (TRIANGLE, G1), (TRIANGLE, G2)]:
            acting_object = action[0]
            acting_goal = action[1]
            prob_success = possible_actions_to_prob_success[action]
            print("INITIAL prob_success", prob_success)

            # try to move object using skill mdp
            initial_config = {
                'start_pos': (0, 0),
                'g1': (4, 4),
                'g2': (4, 0),
                'square_positions': curr_state['square_positions'],
                'triangle_positions': curr_state['triangle_positions'],
            }
            target = {
                'target_object': acting_object,
                'target_goal': acting_goal
            }
            current_skill_mdp_state = {}
            current_skill_mdp_state['pos'] = copy.deepcopy(self.start_pos)
            current_skill_mdp_state['holding'] = None
            current_skill_mdp_state['objects_in_g1'] = None
            current_skill_mdp_state['objects_in_g2'] = None

            skill_game = Gridworld(initial_config, target)
            skill_done = False
            iter_count = 0
            while not skill_done:
                iter_count += 1
                state_vectors_list = []

                current_interaction_mdp_config = {
                    'start_pos': current_skill_mdp_state['pos'],
                    'g1': (4, 4),
                    'g2': (4, 0),
                    'square_positions': curr_state['square_positions'],
                    'triangle_positions': curr_state['triangle_positions'],
                }

                # pdb.set_trace()
                state_vector = self.convert_state_to_vector(current_interaction_mdp_config, current_skill_mdp_state, target)

                state_vectors_list.append(state_vector)
                # one_hot_action_idx = self.action_to_idx[action]
                # one_hot_action = [0] * len(self.low_level_actions)
                # one_hot_action[one_hot_action_idx] = 1
                # actions_list.append(one_hot_action)
                input_x = torch.tensor(state_vectors_list, dtype=torch.float32)
                # input_y = torch.tensor(actions_list, dtype=torch.long)
                predicted_softmax = self.bc_network(input_x)
                prediction_action_idx = torch.argmax(predicted_softmax).item()
                prediction_action = self.low_level_actions[prediction_action_idx]

                current_skill_mdp_state, skill_mdp_rew, skill_done = skill_game.step_given_state(current_skill_mdp_state, prediction_action)
                # print what happened with a tab indent for readability
                print("\t\t", prediction_action, current_skill_mdp_state, skill_mdp_rew, skill_done)



                if iter_count > 20:
                    break

            # if success, update state
            if skill_done:
                prob_success = 1
                print("Success in skill mdp for action", skill_done)
            else:
                prob_success *= 0.0
                print("Success in skill mdp for action", skill_done)
            possible_actions_to_prob_success[action] = prob_success

            # if prob_success == 1:
                # pdb.set_trace()

            if prob_success == 1:
                if acting_object == SQUARE:
                    if len(curr_state['square_positions']) > 0:
                        curr_state['square_positions'].pop()
                        # pdb.set_trace()
                        if acting_goal == G1:
                            curr_state['objects_in_g1'].append(SQUARE)
                        else:
                            curr_state['objects_in_g2'].append(SQUARE)

                else:
                    if len(curr_state['triangle_positions']) > 0:
                        curr_state['triangle_positions'].pop()
                        if acting_goal == G1:
                            curr_state['objects_in_g1'].append(TRIANGLE)
                        else:
                            curr_state['objects_in_g2'].append(TRIANGLE)
            query_cost += 1
            print("NEW STATE", curr_state)
            # pdb.set_trace()


        elif action in [(ASK_PREF, SQUARE), (ASK_PREF, TRIANGLE)]:
            query_cost += 0.1
            acting_object = action[1]
            for elem in beliefs:
                if tuple(self.true_human_reward_weights) == elem:
                    beliefs[elem] *= 1.0
                else:
                    beliefs[elem] *= 0.0
        elif action in [(ASK_DEMO, SQUARE), (ASK_DEMO, TRIANGLE)]:

            query_cost += 3
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

            self.get_human_skill_demo(curr_state, acting_object, acting_goal)

            seen_demos.append((acting_object, acting_goal))
            # possible_actions_to_prob_success[(acting_object, acting_goal)] = 1.0
            possible_actions_to_prob_success[(acting_object, G1)] = 1.0
            possible_actions_to_prob_success[(acting_object, G2)] = 1.0


            if acting_object == SQUARE:
                if len(curr_state['square_positions']) > 0:
                    curr_state['square_positions'].pop()
                    if acting_goal == G1:
                        curr_state['objects_in_g1'].append(SQUARE)
                    else:
                        curr_state['objects_in_g2'].append(SQUARE)

            else:
                if len(curr_state['triangle_positions']) > 0:
                    curr_state['triangle_positions'].pop()
                    if acting_goal == G1:
                        curr_state['objects_in_g1'].append(TRIANGLE)
                    else:
                        curr_state['objects_in_g2'].append(TRIANGLE)
            # pdb.set_trace()

        feature_vector = self.convert_state_to_feature_vector(curr_state)
        step_cost = 1
        reward = np.dot(feature_vector, self.true_human_reward_weights)-query_cost-step_cost
        done = self.is_done_given_interaction_state(curr_state)

        # normalize beliefs
        beliefs_sum = sum(beliefs.values())
        for elem in beliefs:
            beliefs[elem] /= beliefs_sum
        # if done:
        #     reward += np.dot(feature_vector, self.true_human_reward_weights)
        return curr_state, seen_demos, possible_actions_to_prob_success,beliefs,  reward, done


    def get_human_skill_demo(self, current_state, acting_object, acting_goal):
        # perform action in skill mdp
        initial_config = {
            'start_pos': (0, 0),
            'g1': (4, 4),
            'g2': (4, 0),
            'square_positions': current_state['square_positions'],
            'triangle_positions': current_state['triangle_positions'],
        }
        target = {
            'target_object': acting_object,
            'target_goal': acting_goal
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

        self.bc_data['X'].extend(state_vectors_list)
        self.bc_data['Y'].extend(actions_list)
        self.train_bc_policy()


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
            query_cost += 1
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
                    query_cost += 3

            else:
                if len(current_state['triangle_positions']):
                    current_state['triangle_positions'].pop()
                    if acting_goal == G1:
                        current_state['objects_in_g1'].append(TRIANGLE)
                    else:
                        current_state['objects_in_g2'].append(TRIANGLE)
                else:
                    query_cost += 3


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
        if action[0] ==  ASK_PREF:
            # discount reward
            discount = 0.5
            reward_hyp1 = discount * np.dot(feature_vector, hyps[0]) - query_cost - step_cost
            reward_hyp2 = discount * np.dot(feature_vector, hyps[1]) - query_cost - step_cost
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


    def randomly_initialize_bc_data(self, n_start=10):
        self.bc_data = {}

        # list all positions between (0,0) and (5,5)
        all_positions = list(product(range(5), range(5)))
        all_positions_indices = list(range(len(all_positions)))
        all_objects = [SQUARE, TRIANGLE]
        all_goals = [G1, G2]

        self.all_positions = all_positions
        self.all_positions_indices = all_positions_indices
        self.all_objects = all_objects
        self.all_goals = all_goals

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
            n_objs_square = np.random.choice([0, 1, 2])
            n_objs_triangle = np.random.choice([0, 1, 2])
            if n_objs_square + n_objs_triangle == 0:
                continue

            initial_config = {
                'start_pos': (0, 0),
                'g1': (4, 4),
                'g2': (4, 0),
                'square_positions': [all_positions[np.random.choice(all_positions_indices)] for _ in range(n_objs_square)],
                'triangle_positions': [all_positions[np.random.choice(all_positions_indices)] for _ in range(n_objs_triangle)],
            }

            if len(initial_config['square_positions']) == 0:
                # pad with (-1,-1)
                initial_config['square_positions'].append((-1, -1))
                initial_config['square_positions'].append((-1, -1))
            elif len(initial_config['square_positions']) == 1:
                initial_config['square_positions'].append((-1, -1))

            if len(initial_config['triangle_positions']) == 0:
                # pad with (-1,-1)
                initial_config['triangle_positions'].append((-1, -1))
                initial_config['triangle_positions'].append((-1, -1))
            elif len(initial_config['triangle_positions']) == 1:
                initial_config['triangle_positions'].append((-1, -1))


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



    def convert_state_to_vector(self, input_config, state, target):
        initial_config = copy.deepcopy(input_config)
        # object to one-hot vector
        self.obj_to_one_hot = {SQUARE: [1, 0, 0], TRIANGLE: [0, 1, 0], None: [0, 0, 1]}
        # goal to one-hot vector
        self.goal_to_one_hot = {G1: [1, 0], G2: [0, 1]}

        state_vector = []
        # add target object and goal
        state_vector.extend(self.obj_to_one_hot[target['target_object']])
        state_vector.extend(self.goal_to_one_hot[target['target_goal']])

        # add positions of objects

        # if length of initial_config['square_positions'] is 0 or 1, pad with -1
        if len(initial_config['square_positions']) == 0:
            initial_config['square_positions'].append((-1, -1))
            initial_config['square_positions'].append((-1, -1))
        elif len(initial_config['square_positions']) == 1:
            initial_config['square_positions'].append((-1, -1))
        state_vector.extend(list(initial_config['square_positions'][0]))
        state_vector.extend(list(initial_config['square_positions'][1]))

        # if length of initial_config['triangle_positions'] is 0 or 1, pad with -1
        if len(initial_config['triangle_positions']) == 0:
            initial_config['triangle_positions'].append((-1, -1))
            initial_config['triangle_positions'].append((-1, -1))
        elif len(initial_config['triangle_positions']) == 1:
            initial_config['triangle_positions'].append((-1, -1))

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
    # game.train_bc_policy()
    total_reward, interaction_result = game.rollout_interaction()

    print()
    print("total_reward", total_reward)
    # print("interaction_result", interaction_result)

