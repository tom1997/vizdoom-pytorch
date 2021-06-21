#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

from __future__ import print_function
import vizdoom as vzd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import skimage.transform
from tensorboardX import SummaryWriter

from vizdoom import Mode
from time import sleep, time
from collections import deque
from tqdm import trange
from pm import Sum_Tree

# Q-learning settings
# learning_rate = 0.00025
learning_rate = 0.0001
discount_factor = 0.99
train_epochs = 20000
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (84, 84)
stack_size = 4 # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros(resolution, dtype=np.int32) for i in range(stack_size)], maxlen=4)
episodes_to_watch = 10
# model_savefile = "./model-doom_03.pth"
# model_savefile = "./model-doom_dtc.pth"
# model_savefile = "./model-doom_dtc2.pth"
# model_savefile = "./model-doom_dtl.pth"
# model_savefile = "./model-doom_dtl2.pth" # Set epsilon decay=0.999996
#model_savefile = "./model-doom_dtl3.pth" # Added reward shaping on 2 : reward cannot be large
#model_savefile = "./model-doom-PER.pth" # Added per replays
# model_savefile = "./model-doom_dtl_per.pth"
model_savefile = "./model-doom_dtl_per_rnd.pth"
# model_savefile = "./model-doom-PER-formal.pth" # 之前的代码有问题，现在成功使用了IS_weight
#model_savefile = "./model-doom-origin.pth" # origin
# model_savefile = "./model-doom_02_rnd_per.pth"
writer = SummaryWriter("log/" + model_savefile[:-4])
weights = [0, 0.01, 0.5] # AMMO2 Health Killcount

save_model = True
load_model = False
skip_learning = False

# save_model = False
# load_model = True
# skip_learning = True

# Configuration file path
# config_file_path = "../../scenarios/simpler_basic.cfg"
# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"
# config_file_path = "../../scenarios/basic_3.cfg"
# config_file_path = "../../scenarios/defend_the_center.cfg"
# config_file_path = "../../../scenarios/basic_2.cfg"
config_file_path = "../../../scenarios/defend_the_line.cfg"
def reward_weight(game_state, weights=weights):
    weights = np.array(weights)
    game_state = np.array(game_state)
    return np.dot(weights, game_state)

def diff(state, prev_state):
    return [a - b for a, b in zip(state, prev_state)]
# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []

    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        prev_misc = [50, 100, 0]
        episode_rewards = 0
        while not game.is_episode_finished():
            game_state = game.get_state()
            misc = game_state.game_variables
            state = preprocess(game.get_state().screen_buffer)

            intrinsic_reward = reward_weight(diff(misc, prev_misc))
            reward = intrinsic_reward

            best_action_index = agent.get_action(state)
            game.make_action(actions[best_action_index], frame_repeat)
            prev_misc = game_state.game_variables

            episode_rewards += reward
        # r = game.get_total_reward()
        r = game.get_total_reward() + episode_rewards
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()
    prev_misc = [50, 100, 0]  # Default state
    episode_rewards = 0
    global_step = 0
    RND_value = np.array(0)
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        kill_count = []
        epoch_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch, leave=False):
            game_state = game.get_state()
            state = preprocess(game_state.screen_buffer)
            misc = game_state.game_variables
            #  reward setting
            intrinsic_reward = reward_weight(diff(misc, prev_misc)) + RND_value.item()
            prev_misc = misc
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat) + intrinsic_reward
            episode_rewards += reward

            done = game.is_episode_finished()
            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 84, 84)).astype(np.float32)
                prev_misc = [50, 100, 0]

            #agent.append_memory(state, action, reward, next_state, done)
            experience = state, action, reward, next_state, done
            agent.memory.store(experience)

            if epoch_step > agent.batch_size:
                td_error, epsilon, RND_value = agent.train()

                writer.add_scalar("training/TD Error", td_error, global_step)
                writer.add_scalar("training/epsilon", epsilon, global_step)
                writer.add_scalar("training/RND_value", RND_value, global_step)

            if done:
                train_scores.append(episode_rewards)
                # train_scores.append(episode_rewards)
                kill_count.append(misc[2])
                game.new_episode()
                episode_rewards = 0
            # When agent is still alive and game is not yet finished
            if _ == steps_per_epoch - 1:
                train_scores.append(episode_rewards)
                kill_count.append(misc[2])
                episode_rewards = 0
            epoch_step += 1
            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)
        kill_count = np.array(kill_count)
        print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
        writer.add_scalar("performance/mean", train_scores.mean(), global_step)
        writer.add_scalar("performance/std", train_scores.std(), global_step)
        writer.add_scalar("performance/min", train_scores.min(), global_step)
        writer.add_scalar("performance/max", train_scores.max(), global_step)
        writer.add_scalar("performance/average killcount", kill_count.mean(), global_step)
        # test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

class RND(nn.Module):
    def __init__(self):
        super(RND, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.state_fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4096)
        x = self.state_fc(x)
        return x

class DuelQNet(nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super(DuelQNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.state_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, available_actions_count)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 4096)
        x1 = x[:, :2048]  # input for the net to calculate the state value
        x2 = x[:, 2048:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return x


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = Sum_Tree.SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        #n = batch_size
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment,

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors.cpu().detach().numpy(), self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DQNAgent:
    def __init__(self, action_size, memory_size, batch_size, discount_factor,
                 lr, load_model, epsilon=1, epsilon_decay=0.999996, epsilon_min=0.1):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = Memory(memory_size)
        self.criterion = nn.MSELoss()
        self.z = RND().to(DEVICE)
        self.fixedz = RND().to(DEVICE)
        if load_model:
            print("Loading model from: ", model_savefile)
            self.q_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DuelQNet(action_size).to(DEVICE)
            self.target_net = DuelQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)
        self.RNDopt = optim.SGD(self.z.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        tree_idx, batch, ISWeights_mb = self.memory.sample(batch_size)

        #batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)
        batch = batch.squeeze(1)
        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)  # used for indexing the batch

        # value of the next states with double q learning
        # see https://arxiv.org/abs/1509.06461 for more information on double q learning
        ## RND trick
        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        # this defines y = r + discount * max_a q(s', a)
        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        # this selects only the q values of the actions taken
        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)
        fixed_rnd = self.fixedz(states)
        rnd = self.z(states)
        self.RNDopt.zero_grad()
        self.opt.zero_grad()
        absolute_errors = torch.abs(q_targets - action_values)  # for updating Sumtree

        # td_error =  self.criterion(q_targets, action_values)
        td_error = torch.mean(torch.tensor(ISWeights_mb).squeeze(1).to(DEVICE) * ((q_targets - action_values) ** 2))
        RND_reward = torch.mean((rnd - fixed_rnd) ** 2)

        agent.memory.batch_update(tree_idx, absolute_errors)
        td_error.backward()
        RND_reward.backward()

        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        return td_error, self.epsilon, RND_reward


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros(resolution, dtype=np.int32) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

def pretrain(game, memory_size, actions, memory):
    # Render the environment
    game.new_episode()

    prev_misc = [50, 100, 0]
    pretrain_length = memory_size
    for i in range(pretrain_length):
        # If it's the first step
        #if i == 0:
            # First we need a state
        # Our state is now the next_state
        state = preprocess(game.get_state().screen_buffer)

        # Random action
        action = random.choice(range(len(actions)))
        # Get the rewards
        misc = game.get_state().game_variables
        #  reward setting
        intrinsic_reward = reward_weight(diff(misc, prev_misc))
        reward = game.make_action(actions[action], frame_repeat) + intrinsic_reward

        prev_misc = misc

        # Look if the episode is finished
        done = game.is_episode_finished()

        # If we're dead
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            # experience = np.hstack((state, [action, reward], next_state, done))

            experience = state, action, reward, next_state, done
            memory.store(experience)

            # Start a new episode
            game.new_episode()

            # Stack the frames
            #state, stacked_frames = stack_frames(stacked_frames, state, True)
            #state = game.make_action(state, frame_repeat)
            prev_misc = [50, 100, 0]

        else:
            # Get the next state
            #next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            #state = game.make_action(state, frame_repeat)
            # Add experience to memory
            next_state = preprocess(game.get_state().screen_buffer)
            experience = state, action, reward, next_state, done
            memory.store(experience)





if __name__ == '__main__':
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    # actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions  = np.identity(n, dtype=int).tolist()
    # Initialize our agent with the set parameters
    agent = DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model)
    memory = agent.memory

    # Run the training for the set number of epochs
    if not skip_learning:
        pretrain(game, replay_memory_size, actions, memory)
        agent, game = run(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                          steps_per_epoch=learning_steps_per_epoch)

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.set_sound_enabled(True)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        prev_misc = [50, 100, 0]
        episode_rewards = 0
        time_start = time()
        while not game.is_episode_finished():
            game_state = game.get_state()
            misc = game_state.game_variables
            state = preprocess(game.get_state().screen_buffer)

            # health_reward = 0.001 * (misc[1] - prev_misc[1])
            # kill_reward = 0.1 * (misc[2] - prev_misc[2])
            # ANMO2_reward = 0.0001 * (misc[0] - prev_misc[0])
            intrinsic_reward = reward_weight(diff(misc, prev_misc))
            reward = intrinsic_reward

            best_action_index = agent.get_action(state)
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            prev_misc = game_state.game_variables
            episode_rewards += reward
            for _ in range(frame_repeat):
                game.advance_action()
        time_end = time()
        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward() + episode_rewards
        print("Total score: ", score, "Living time: ",  time_end - time_start, "Killcount: ", misc[2])

