import matplotlib
matplotlib.rcParams['figure.figsize'] = 16, 8
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

import math
import numpy as np
import warnings
warnings.simplefilter('ignore')
import gym
import time
from IPython.display import clear_output
import itertools

import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils.memory.ReplayMemory import ReplayMemory
from agents.utils.memory.Transition import Transition

## CartPole: Agente aleatorio
'''
CartPole es un entorno donde un poste está unido por una unión no accionada a un carro,
que se mueve a lo largo de una pista sin fricción. El sistema se controla aplicando una 
fuerza de +1 o -1 al carro. El péndulo comienza en posición vertical, y el objetivo es 
evitar que se caiga. Se proporciona una recompensa de +1 por cada paso de tiempo que el
poste permanezca en posición vertical. El episodio termina cuando el poste está a más
de 15 grados de la vertical, o el carro se mueve más de 2.4 unidades desde el centro.
'''
# env = gym.make('CartPole-v0')
env = gym.make('Acrobot-v1')
env.reset()
for _ in range(250):
    env.render(mode='human')
    observation, reward, done, info = env.step(env.action_space.sample()) # se ejecuta una acción aleatoria
    if done:
        env.reset()
env.close()
clear_output()

# class LinearModel(nn.Module):
#
#     def __init__(self, _input_size: int, _output_size: int):
#         super(LinearModel, self).__init__()
#         self.output = nn.Linear(_input_size, _output_size)
#
#         # init weights
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0.01)
#
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = self.output(x)
#         return x
#
# class LinearCartPoleSolver:
#     def __init__(self, env, n_episodes=3000, max_env_steps=None, gamma=0.9,
#                  epsilon=0.5, epsilon_min=0.01, epsilon_log_decay=0.005, alpha=1e-3,
#                  memory_size=10000, batch_size=256, render=False, debug=False):
#
#         self.memory = ReplayMemory(capacity=memory_size)
#         self.env = env
#
#         # hyper-parameter setting
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_log_decay
#         self.alpha = alpha
#         self.n_episodes = n_episodes
#         self.batch_size = batch_size
#         if max_env_steps is not None:
#             self.env._max_episode_steps = max_env_steps
#         self.observation_space_size = env.observation_space.shape[0]
#         self.action_space_size = env.action_space.n
#
#         self.render = render
#         self.debug = debug
#         if debug:
#             self.loss_list = []
#         # if gpu is to be used
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # Init model 1
#         # Use the nn package to define our model as a sequence of layers. nn.Sequential
#         # is a Module which contains other Modules, and applies them in sequence to
#         # produce its output. Each Linear Module computes output from input using a
#         # linear function, and holds internal Tensors for its weight and bias.
#         # After constructing the model we use the .to() method to move it to the
#         # desired device.
#         self.model = LinearModel(self.observation_space_size, self.action_space_size).to(self.device)
#         self.model.train()
#
#         # The nn package also contains definitions of popular loss functions; in this
#         # case we will use Mean Squared Error (MSE) as our loss function. Setting
#         # reduction='sum' means that we are computing the *sum* of squared errors rather
#         # than the mean; this is for consistency with the examples above where we
#         # manually compute the loss, but in practice it is more common to use mean
#         # squared error as a loss by setting reduction='elementwise_mean'.
#         self.loss_fn = torch.nn.MSELoss()
#
#         # Use the optim package to define an Optimizer that will update the weights of
#         # the model for us. Here we will use Adam; the optim package contains many other
#         # optimization algoriths. The first argument to the Adam constructor tells the
#         # optimizer which Tensors it should update.
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
#
#     def choose_action(self, state, epsilon):
#         """Chooses the next action according to the model trained and the policy"""
#
#         # exploits the current knowledge if the random number > epsilon, otherwise explores
#         if np.random.random() <= epsilon:
#             return self.env.action_space.sample()
#         else:
#             with torch.no_grad():
#                 q = self.model(state)
#                 argmax = torch.argmax(q)
#                 return argmax.item()
#
#     def get_epsilon(self, episode):
#         """Returns an epsilon that decays over time until a minimum epsilon value is reached; in this case the minimum
#         value is returned"""
#         return max(self.epsilon_min, self.epsilon * math.exp(-self.epsilon_decay * episode))
#
#     def replay(self):
#         """Previously stored (s, a, r, s') tuples are replayed (that is, are added into the model). The size of the
#         tuples added is determined by the batch_size parameter"""
#
#         transitions, _ = self.memory.sample(self.batch_size)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
#         # Compute a mask of non-final states and concatenate the batch elements
#         # (a final state would've been the one after which simulation ended)
#         non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
#
#         non_final_mask = torch.stack(batch.done)
#         state_batch = torch.stack(batch.state)
#         action_batch = torch.stack(batch.action)
#         reward_batch = torch.stack(batch.reward)
#
#         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#         # columns of actions taken. These are the actions which would've been taken
#         # for each batch state according to policy_net
#         state_action_values = self.model(state_batch).gather(1, action_batch)
#
#         # Compute V(s_{t+1}) for all next states.
#         # Expected values of actions for non_final_next_states are computed based
#         # on the "older" target_net; selecting their best reward with max(1)[0].
#         # This is merged based on the mask, such that we'll have either the expected
#         # state value or 0 in case the state was final.
#         with torch.no_grad():
#             next_state_values = torch.zeros(self.batch_size, device=self.device)
#             next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
#             # Compute the expected Q values
#             expected_state_action_values = reward_batch + self.gamma * next_state_values
#
#         # Compute loss
#         loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
#
#         if self.debug:
#             self.loss_list.append(loss)
#         # Optimize the model
#         self.optimizer.zero_grad()
#         loss.backward()
#         for param in self.model.parameters():
#             param.grad.data.clamp_(-1, 1)
#         self.optimizer.step()
#
#     def run(self):
#         """Main loop that controls the execution of the agent"""
#
#         scores = []
#         mean_scores = []
#         for e in range(self.n_episodes):
#             state = self.env.reset()
#             state = torch.tensor(state, device=self.device, dtype=torch.float)
#             done = False
#             cum_reward = 0
#             while not done:
#                 action = self.choose_action(
#                     state,
#                     self.get_epsilon(e))
#                 next_state, reward, done, _ = self.env.step(action)
#                 next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)
#
#                 cum_reward += reward
#                 self.memory.push(
#                     state,  #Converted to tensor before
#                     torch.tensor([action], device=self.device),
#                     None if done else next_state,
#                     torch.tensor(reward, device=self.device).clamp_(-1, 1),
#                     torch.tensor(not done, device=self.device, dtype=torch.bool))
#
#                 if self.memory.__len__() >= self.batch_size:
#                     self.replay()
#                 state = next_state
#
#             scores.append(cum_reward)
#             mean_score = np.mean(scores)
#             mean_scores.append(mean_score)
#             if e % 100 == 0 and self.debug:
#                 print('[Episode {}] - Mean reward {}.'.format(e, mean_score))
#
#         # noinspection PyUnboundLocalVariable
#         print('[Episode {}] - Mean reward {}.'.format(e, mean_score))
#         return scores, mean_scores
#
#     def save(self, path):
#         torch.save(self.model, path)
#
#     def load(self, path):
#         self.model = torch.load(path)
#
N_EPISODES = 5000
# agent = LinearCartPoleSolver(gym.make('CartPole-v0'), n_episodes=5000, debug=True)
agent = LinearCartPoleSolver(gym.make('Acrobot-v1'), n_episodes=N_EPISODES, debug=True)
scoresLineal, meanLineal = agent.run()
agent.save("./Lineal_pendulum_{}".format(N_EPISODES))

# se muestra el reward/score optenido por episodio
plt.plot(np.array(scoresLineal), label='Lineal', c='#5c8cbc')
# plt.ylim(0, 200)
plt.title('Recompensa por episodio')
plt.legend(loc='upper left')
plt.show()

plt.plot(np.array(meanLineal), label='Lineal', c='#5c8cbc')
# plt.ylim(0, 200)
plt.title('Recompensa promedio por episodio')
plt.legend(loc='upper left')
plt.show()
#
# with open("scoresLineal_pendulum", "wb") as scoresLineal:
#     np.save(scoresLineal, np.array(scoresLineal))
# with open("meanLineal_pendulum", "wb") as f_meanLineal:
#     np.save(f_meanLineal, np.array(meanLineal))



# class Net(nn.Module):
#
#     def __init__(self, _input_size: int, _output_size: int, _hidden_layers: int, _hidden_size: int):
#         super(Net, self).__init__()
#         self.input = nn.Linear(_input_size, _hidden_size)
#         self.hidden_layers = _hidden_layers
#         self.hidden = []
#         for i in range(_hidden_layers):
#             layer = nn.Linear(_hidden_size, _hidden_size)
#             self.add_module('h'+str(i), layer)
#             self.hidden.append(layer)
#         self.output = nn.Linear(_hidden_size, _output_size)
#
#         # init weights
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 m.bias.data.fill_(0.01)
#
#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x):
#         x = F.relu(self.input(x))
#         for i in range(self.hidden_layers):
#             x = F.relu(self.hidden[i](x))
#         x = self.output(x)
#         return x
#
# class DDQN:
#     def __init__(self, env, n_episodes=3000, max_env_steps=None, gamma=0.9,
#                  epsilon=0.5, epsilon_min=0.05, epsilon_log_decay=0.001, alpha=1e-3,
#                  memory_size=10000, batch_size=256, c=10, hidden_layers=2, hidden_size=24,
#                  render=False, debug=False):
#
#         self.memory = ReplayMemory(capacity=memory_size)
#         self.env = env
#
#         # hyper-parameter setting
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_log_decay
#         self.alpha = alpha
#         self.n_episodes = n_episodes
#         self.batch_size = batch_size
#         self.c = c
#         if max_env_steps is not None:
#             self.env._max_episode_steps = max_env_steps
#         self.observation_space_size = env.observation_space.shape[0]
#         self.action_space_size = env.action_space.n
#
#         self.render = render
#         self.debug = debug
#         if debug:
#             self.loss_list = []
#         # if gpu is to be used
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # Init model 1
#         # Use the nn package to define our model as a sequence of layers. nn.Sequential
#         # is a Module which contains other Modules, and applies them in sequence to
#         # produce its output. Each Linear Module computes output from input using a
#         # linear function, and holds internal Tensors for its weight and bias.
#         # After constructing the model we use the .to() method to move it to the
#         # desired device.
#         self.model = Net(self.observation_space_size, self.action_space_size, hidden_layers, hidden_size) \
#             .to(self.device)
#         self.target = Net(self.observation_space_size, self.action_space_size, hidden_layers, hidden_size) \
#             .to(self.device)
#         self.target.load_state_dict(self.model.state_dict())
#         self.target.eval()
#         self.model.train()
#
#         # The nn package also contains definitions of popular loss functions; in this
#         # case we will use Mean Squared Error (MSE) as our loss function. Setting
#         # reduction='sum' means that we are computing the *sum* of squared errors rather
#         # than the mean; this is for consistency with the examples above where we
#         # manually compute the loss, but in practice it is more common to use mean
#         # squared error as a loss by setting reduction='elementwise_mean'.
#         self.loss_fn = torch.nn.MSELoss()
#
#         # Use the optim package to define an Optimizer that will update the weights of
#         # the model for us. Here we will use Adam; the optim package contains many other
#         # optimization algoriths. The first argument to the Adam constructor tells the
#         # optimizer which Tensors it should update.
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
#
#     def choose_action(self, state, epsilon):
#         """Chooses the next action according to the model trained and the policy"""
#
#         # exploits the current knowledge if the random number > epsilon, otherwise explores
#         if np.random.random() <= epsilon:
#             return self.env.action_space.sample()
#         else:
#             with torch.no_grad():
#                 q = self.model(state)
#                 argmax = torch.argmax(q)
#                 return argmax.item()
#
#     def get_epsilon(self, episode):
#         """Returns an epsilon that decays over time until a minimum epsilon value is reached; in this case the minimum
#         value is returned"""
#         return max(self.epsilon_min, self.epsilon * math.exp(-self.epsilon_decay * episode))
#
#     def replay(self):
#         """Previously stored (s, a, r, s') tuples are replayed (that is, are added into the model). The size of the
#         tuples added is determined by the batch_size parameter"""
#
#         transitions, _ = self.memory.sample(self.batch_size)
#         # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#         # detailed explanation). This converts batch-array of Transitions
#         # to Transition of batch-arrays.
#         batch = Transition(*zip(*transitions))
#         # Compute a mask of non-final states and concatenate the batch elements
#         # (a final state would've been the one after which simulation ended)
#         non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
#
#         non_final_mask = torch.stack(batch.done)
#         state_batch = torch.stack(batch.state)
#         action_batch = torch.stack(batch.action)
#         reward_batch = torch.stack(batch.reward)
#
#         # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#         # columns of actions taken. These are the actions which would've been taken
#         # for each batch state according to policy_net
#         state_action_values = self.model(state_batch).gather(1, action_batch)
#
#         # Compute V(s_{t+1}) for all next states.
#         # Expected values of actions for non_final_next_states are computed based
#         # on the "older" target_net; selecting their best reward with max(1)[0].
#         # This is merged based on the mask, such that we'll have either the expected
#         # state value or 0 in case the state was final.
#         with torch.no_grad():
#             next_state_values = torch.zeros(self.batch_size, device=self.device)
#             # DDQN magic!! (start) -------------------------------------------------------------------------------------
#             # DQN Network choose action for next state
#             _, next_state_actions = self.model(non_final_next_states).max(1, keepdim=True)
#             # Target network calculates the Q value of taking that action at that state
#             next_state_values[non_final_mask] = self.target(non_final_next_states).gather(1, next_state_actions).squeeze()
#             # DDQN magic!! (end) ---------------------------------------------------------------------------------------
#             # Compute the expected Q values
#             expected_state_action_values = reward_batch + self.gamma * next_state_values
#
#         # Compute loss
#         loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
#
#         if self.debug:
#             self.loss_list.append(loss)
#         # Optimize the model
#         self.optimizer.zero_grad()
#         loss.backward()
#         for param in self.model.parameters():
#             param.grad.data.clamp_(-1, 1)
#         self.optimizer.step()
#
#     def run(self):
#         """Main loop that controls the execution of the agent"""
#
#         scores = []
#         mean_scores = []
#         j = 0  # used for model2 update every c steps
#         for e in range(self.n_episodes):
#             state = self.env.reset()
#             state = torch.tensor(state, device=self.device, dtype=torch.float)
#             done = False
#             cum_reward = 0
#             while not done:
#                 action = self.choose_action(
#                     state,
#                     self.get_epsilon(e))
#                 next_state, reward, done, _ = self.env.step(action)
#                 next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)
#
#                 cum_reward += reward
#                 self.memory.push(
#                     state,  #Converted to tensor in choose_action method
#                     torch.tensor([action], device=self.device),
#                     None if done else next_state,
#                     torch.tensor(reward, device=self.device).clamp_(-1, 1),
#                     torch.tensor(not done, device=self.device, dtype=torch.bool))
#
#                 if self.memory.__len__() >= self.batch_size:
#                     self.replay()
#
#                 state = next_state
#                 j += 1
#
#                 # update second model
#                 if j % self.c == 0:
#                     self.target.load_state_dict(self.model.state_dict())
#                     self.target.eval()
#
#             scores.append(cum_reward)
#             mean_score = np.mean(scores)
#             mean_scores.append(mean_score)
#             if e % 100 == 0 and self.debug:
#                 print('[Episode {}] - Mean reward {}.'.format(e, mean_score))
#
#         # noinspection PyUnboundLocalVariable
#         print('[Episode {}] - Mean reward {}.'.format(e, mean_score))
#         return scores, mean_scores
#
#     def save(self, path):
#         torch.save(self.model, path)
#
#     def load(self, path):
#         self.model = torch.load(path)
#
N_EPISODES = 5000
# agent = DDQN(gym.make('CartPole-v0'), n_episodes=N_EPISODES, debug=True)
agent = DDQN(gym.make('Acrobot-v1'), n_episodes=N_EPISODES, debug=True)
scoresDDQN, meanDDQN = agent.run()
agent.save("./DDQN_pendulum_{}".format(N_EPISODES))

# se muestra el reward/score optenido por episodio
plt.plot(np.array(scoresDDQN), label='DDQN Pendulum', c='#7e5fa4')
plt.ylim(0, 200)
plt.title('Recompensa por episodio')
plt.legend(loc='upper left')
plt.show()

plt.plot(np.array(meanDDQN), label='DDQN ', c='#7e5fa4')
plt.ylim(0, 200)
plt.title('Recompensa promedio por episodio')
plt.legend(loc='upper left')
plt.show()
# # with open("scoresDDQN", "wb") as f_scoresDDQN:
# with open("scoresDDQN_pendulum", "wb") as f_scoresDDQN:
#     np.save(f_scoresDDQN, np.array(scoresDDQN))
# # with open("meanDDQN", "wb") as f_meanDDQN:
# with open("meanDDQN_pendulum", "wb") as f_meanDDQN:
#     np.save(f_meanDDQN, np.array(meanDDQN))

class DuelingNet(nn.Module):

    def __init__(self, _input_size: int, _output_size: int, _hidden_layers: int, _hidden_size: int):
        super(DuelingNet, self).__init__()
        self.input = nn.Linear(_input_size, _hidden_size)
        self.hidden_layers = _hidden_layers
        self.hidden = []
        for i in range(_hidden_layers):
            layer = nn.Linear(_hidden_size, _hidden_size)
            self.add_module('h'+str(i), layer)
            self.hidden.append(layer)
        self.output = nn.Linear(_hidden_size, _output_size)
        self.advantage = nn.Linear(_hidden_size, _output_size)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input(x))
        for i in range(self.hidden_layers):
            x = F.relu(self.hidden[i](x))
        a = F.relu(self.advantage(x))
        x = self.output(x)
        return x + a - a.mean()     # Agregating Layer

class DuelingDQN:
    def __init__(self, env, n_episodes=3000, max_env_steps=None, gamma=0.9,
                 epsilon=0.5, epsilon_min=0.05, epsilon_log_decay=0.001, alpha=1e-3,
                 memory_size=10000, batch_size=256, c=10, hidden_layers=2, hidden_size=24,
                 render=False, debug=False):

        self.memory = ReplayMemory(capacity=memory_size)
        self.env = env

        # hyper-parameter setting
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.c = c
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps
        self.observation_space_size = env.observation_space.shape[0]
        self.action_space_size = env.action_space.n

        self.render = render
        self.debug = debug
        if debug:
            self.loss_list = []
        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init model 1
        # Use the nn package to define our model as a sequence of layers. nn.Sequential
        # is a Module which contains other Modules, and applies them in sequence to
        # produce its output. Each Linear Module computes output from input using a
        # linear function, and holds internal Tensors for its weight and bias.
        # After constructing the model we use the .to() method to move it to the
        # desired device.
        self.model = DuelingNet(self.observation_space_size, self.action_space_size, hidden_layers, hidden_size) \
            .to(self.device)
        self.target = DuelingNet(self.observation_space_size, self.action_space_size, hidden_layers, hidden_size) \
            .to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.model.train()

        # The nn package also contains definitions of popular loss functions; in this
        # case we will use Mean Squared Error (MSE) as our loss function. Setting
        # reduction='sum' means that we are computing the *sum* of squared errors rather
        # than the mean; this is for consistency with the examples above where we
        # manually compute the loss, but in practice it is more common to use mean
        # squared error as a loss by setting reduction='elementwise_mean'.
        self.loss_fn = torch.nn.MSELoss()

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

    def choose_action(self, state, epsilon):
        """Chooses the next action according to the model trained and the policy"""

        # exploits the current knowledge if the random number > epsilon, otherwise explores
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q = self.model(state)
                argmax = torch.argmax(q)
                return argmax.item()

    def get_epsilon(self, episode):
        """Returns an epsilon that decays over time until a minimum epsilon value is reached; in this case the minimum
        value is returned"""
        return max(self.epsilon_min, self.epsilon * math.exp(-self.epsilon_decay * episode))

    def replay(self):
        """Previously stored (s, a, r, s') tuples are replayed (that is, are added into the model). The size of the
        tuples added is determined by the batch_size parameter"""

        transitions, _ = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        non_final_mask = torch.stack(batch.done)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        if self.debug:
            self.loss_list.append(loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def run(self):
        """Main loop that controls the execution of the agent"""

        scores = []
        mean_scores = []
        j = 0  # used for model2 update every c steps
        for e in range(self.n_episodes):
            state = self.env.reset()
            state = torch.tensor(state, device=self.device, dtype=torch.float)
            done = False
            cum_reward = 0
            while not done:
                action = self.choose_action(
                    state,
                    self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)

                cum_reward += reward
                self.memory.push(
                    state,  #Converted to tensor in choose_action method
                    torch.tensor([action], device=self.device),
                    None if done else next_state,
                    torch.tensor(reward, device=self.device).clamp_(-1, 1),
                    torch.tensor(not done, device=self.device, dtype=torch.bool))

                if self.memory.__len__() >= self.batch_size:
                    self.replay()

                state = next_state
                j += 1

                # update second model
                if j % self.c == 0:
                    self.target.load_state_dict(self.model.state_dict())
                    self.target.eval()

            scores.append(cum_reward)
            mean_score = np.mean(scores)
            mean_scores.append(mean_score)
            if e % 100 == 0 and self.debug:
                print('[Episode {}] - Mean reward {}.'.format(e, mean_score))

        # noinspection PyUnboundLocalVariable
        print('[Episode {}] - Mean reward {}.'.format(e, mean_score))
        return scores, mean_scores

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)

N_EPISODES = 5000
# agent = DuelingDQN(gym.make('CartPole-v0'), n_episodes=N_EPISODES, debug=True)
agent = DuelingDQN(gym.make('Acrobot-v1'), n_episodes=N_EPISODES, debug=True)
scoresDuelingDQN, meanDuelingDQN = agent.run()
agent.save("./duelingDQN_pendulum_{}".format(N_EPISODES))

# se muestra el reward/score optenido por episodio
plt.plot(np.array(scoresDuelingDQN), label='DuelingDQN', c='#7e5fa4')
# plt.ylim(0, 200)
plt.title('Recompensa por episodio')
plt.legend(loc='upper left')
plt.show()

plt.plot(np.array(meanDuelingDQN), label='DuelingDQN', c='#7e5fa4')
# plt.ylim(0, 200)
plt.title('Recompensa promedio por episodio')
plt.legend(loc='upper left')
plt.show()

with open("scoresDuelingDQN_pendulum", "wb") as scoresDuelingDQN:
    np.save(scoresDuelingDQN, np.array(scoresDuelingDQN))
with open("meanDuelingDQN_pendulum", "wb") as f_meanDuelingDQN:
    np.save(f_meanDuelingDQN, np.array(meanDuelingDQN))
