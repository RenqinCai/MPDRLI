import time
import pickle
import sys
import gym.spaces
import log
import itertools
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import namedtuple
from utils import LinearSchedule, ReplayBuffer, get_wrapper_by_name
from dqn_model import DQN

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_lambda"])

class EVAL_POISON_QLEARNING(object):
	def __init__(
        self,
		model_input_dir,
        poison_model_input_dir,
        topk_ratio,
        env,
        q_func,
        optimizer_spec,
        exploration,
        stopping_criterion,
        replay_buffer_size,
        batch_size,
        gamma,
        learning_starts,
        learning_freq,
        frame_history_len,
        target_update_freq,
        grad_norm_clipping,
        double_q=True):

		assert type(env.observation_space) == gym.spaces.Box
		assert type(env.action_space)      == gym.spaces.Discrete

		self.model_input_dir = model_input_dir
        self.poison_model_input_dir = poison_model_input_dir
		self.topk_ratio = topk_ratio
		self.target_update_freq = target_update_freq
		self.optimizer_spec = optimizer_spec
		self.batch_size = batch_size
		self.learning_freq = learning_freq
		self.learning_starts = learning_starts
		self.stopping_criterion = stopping_criterion
		self.env = env
		self.exploration = exploration
		self.gamma = gamma
		self.double_q = double_q
		self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
		
		observation = self.env.observation_space
		print(observation.shape)

		if len(observation.shape) == 1:
			# This means we are running on low-dimensional observations (e.g. RAM)
			in_features = observation.shape[0]
		else:
			img_h, img_w, img_c = observation.shape
			in_features = frame_history_len * img_c
		self.num_actions = self.env.action_space.n

		print("in features", in_features)

		"""
		set up training 
		"""

		self.q_net = q_func(in_features, self.num_actions).to(self.device)
		self.target_q_net = q_func(in_features, self.num_actions).to(self.device)
		self.poison_q_net = q_func(in_features, self.num_actions).to(self.device)
	    
        log.load_model(self.q_net, model_input_dir) 
		log.load_model(self.target_q_net, model_input_dir)
		log.load_model(self.poison_q_net, poison_model_input_dir)

		parameters = self.q_net.parameters()
		self.optimizer = self.optimizer_spec.constructor(parameters, lr=1, 
														**self.optimizer_spec.kwargs)
		self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.optimizer_spec.lr_lambda)

		self.clip_grad_norm_fn = lambda : nn.utils.clip_grad_norm_(parameters, max_norm=grad_norm_clipping)

		self.update_target_fn = lambda : self.target_q_net.load_state_dict(self.q_net.state_dict())

		self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
		self.replay_buffer_idx = None
		self.model_initialized = False

		"""
		set up env
		"""

		self.num_param_updates = 0
		self.mean_episode_reward      = -float('nan')
		self.best_mean_episode_reward = -float('inf')
		self.last_obs = self.env.reset()
		self.log_every_n_steps = 10000

		self.start_time = time.time()
		self.t = 0

	def cal_loss(self, obs, ac, rw, nxobs, done):
		ts_obs, ts_ac, ts_rw, ts_nxobs, ts_done = map(lambda x: torch.from_numpy(x).to(self.device), [obs, ac, rw, nxobs, done])

		ts_ac = ts_ac.long().view(-1, 1)
		
		with torch.no_grad():
			if not self.double_q:
				ts_max_ac = self.target_q_net(ts_nxobs).argmax(-1, keepdim=True)
			else:
				ts_max_ac = self.q_net(ts_nxobs).argmax(-1, keepdim=True)
			expected_Q = ts_rw + (1 - ts_done) * self.gamma * self.target_q_net(ts_nxobs).gather(-1, ts_max_ac).view(-1)
		pred_Q = self.q_net(ts_obs).gather(-1, ts_ac).view(-1)

		total_error = F.smooth_l1_loss(pred_Q, expected_Q)

		return total_error

	def stopping_criterion_met(self):
		stopping_flag = False
		if self.stopping_criterion is not None:
			stopping_flag = self.stopping_criterion(self.env, self.t)
		return stopping_flag

	def step_env(self):
		idx = self.replay_buffer.store_frame(self.last_obs)
		# print()
		ts_obs = torch.from_numpy(self.replay_buffer.encode_recent_observation()[None]).to(self.device)

		# print("ts_obs", ts_obs.size())

		if not self.model_initialized or (random.random() < self.exploration.value(self.t)):
			action = random.randint(0, self.num_actions - 1)
		else:
			action = self.q_net(ts_obs).view(-1).argmax().item()

		new_obs, reward, done, _ = self.env.step(action)

		self.replay_buffer.store_effect(idx, action, reward, done)
		self.last_obs = new_obs

		if done:
			self.last_obs = self.env.reset()

    def get_topk(self, ts_obs, ts_ac):
		# print("get_topk")
		
		self.q_net.eval()

		# print(ts_obs.dtype)
		# ts_poison_obs = torch.tensor(ts_obs.float(), requires_grad=True)

		ts_poison_obs = ts_obs.float().clone().detach().requires_grad_(True)
		
		pred_Q = self.q_net(ts_poison_obs).gather(-1, ts_ac).view(-1)

		# print("pred_Q", pred_Q.size())
		pred_Q.backward(torch.ones(pred_Q.size()).to(self.device))

		channel_val_saliency, channel_index_saliency = torch.max(ts_poison_obs.grad.data.abs(), dim=-1)
		# print("saliency before poison", channel_val_saliency)

		self.original_saliency.append(channel_val_saliency)

		saliency_size = channel_val_saliency.size()
		batch_size = saliency_size[0]
		row_size = saliency_size[1]
		col_size = saliency_size[2]
		# channel_size = saliency_size[3]
		new_saliency = channel_val_saliency.view(batch_size, -1)
		
		topk = int((row_size*col_size)*0.15)
		top_val_saliency, top_index_saliency = torch.topk(new_saliency, dim=1, k=topk)
		# print("top_val_saliency", top_val_saliency)

		del ts_poison_obs

		return top_val_saliency, top_index_saliency, channel_index_saliency

	def get_poison_loss(self, obs, ac):
		# print("get_poison_loss")
		# print("*"*10)
		ts_obs, ts_ac = map(lambda x: torch.from_numpy(x).to(self.device),
													[obs, ac])
		ts_ac = ts_ac.long().view(-1, 1)

		top_val_saliency, top_index_saliency, channel_index_saliency = self.get_topk(ts_obs, ts_ac)

		ts_poison_obs = ts_obs.float().clone().detach().requires_grad_(True)

		pred_Q = self.poison_q_net(ts_poison_obs).gather(-1, ts_ac).view(-1)
		pred_Q.backward(torch.ones(pred_Q.size()).to(self.device))

		saliency = ts_poison_obs.grad.data.abs()

		channel_saliency = saliency.gather(-1, channel_index_saliency.unsqueeze(-1)).squeeze(-1)

		new_channel_saliency = channel_saliency.view(channel_saliency.size()[0], -1)

		top_saliency = new_channel_saliency[0, top_index_saliency]
		# print("poison_saliency", top_saliency)

		self.poison_saliency.append(channel_saliency)
		# print("poison_saliency", channel_saliency)

		poison_loss = torch.exp(top_saliency)
		poison_loss = torch.sum(poison_loss, dim=1)
		poison_loss = torch.mean(poison_loss)

		loss = torch.exp(top_val_saliency)
		loss = torch.sum(loss, dim=1)
		loss = torch.mean(loss)

		del ts_poison_obs

		return poison_loss, loss

	def update_model(self):
		
		if self.replay_buffer.can_sample(self.batch_size):
			obs, ac, rw, nxobs, done = self.replay_buffer.sample(self.batch_size)

			if not self.model_initialized:
				self.model_initialized = True

			loss = self.calc_loss(obs, ac, rw, nxobs, done)
			poison_loss, original_loss = self.get_poison_loss(obs, ac)

			lambda_param = 0.3

			full_loss = loss + lambda_param*poison_loss

			self.optimizer.zero_grad()
			full_loss.backward()
			self.clip_grad_norm_fn()
			self.optimizer.step()

			self.num_param_updates += 1
			if self.num_param_updates % self.target_update_freq == 0:
				self.update_target_fn()

			self.lr_scheduler.step()
			self.t += 1

            if self.t == 100:
                saliency_map = {"original_saliency":self.original_saliency, "poison_saliency":self.poison_saliency}

                saliency_file = "./saliency.pickle"
                f = open(saliency_file, "wb")
                pickle.dump(saliency_map, f)
                f.close()
                exit()

	def log_progress(self):
		episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

		if len(episode_rewards) > 0:
			self.mean_episode_reward = np.mean(episode_rewards[-100:])

		if len(episode_rewards) > 100:
			self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

		if self.t % self.log_every_n_steps == 0 and self.model_initialized:
			log.log_tabular("TimeStep", self.t)
			log.log_tabular("MeanReturn", self.mean_episode_reward)
			log.log_tabular("BestMeanReturn", max(self.best_mean_episode_reward, self.mean_episode_reward))
			log.log_tabular("Episodes", len(episode_rewards))
			log.log_tabular("Exploration", self.exploration.value(self.t))
			log.log_tabular("LearningRate", self.optimizer_spec.lr_lambda(self.t))
			log.log_tabular("Time", (time.time() - self.start_time) / 60.)
			log.dump_tabular()
			log.save_pytorch_model(self.q_net)
      