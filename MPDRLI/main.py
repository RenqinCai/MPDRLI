from dqn_learning import QLEARNING
from poison_dqn_learning import POISON_QLEARNING
from eval_poison_dqn_learning import EVAL_POISON_QLEARNING
import argparse
import gym
from gym import wrappers
import time
import log
import os.path as osp
import random
import numpy as np
import torch
from torch import nn

from dqn_model import DQN
from utils import PiecewiseSchedule, get_wrapper_by_name
from atari_wrappers import wrap_deepmind

def set_global_seeds(i):
    torch.manual_seed(i)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_env(expt_dir, env_name, exp_name, seed):
    env = gym.make(env_name)

    set_global_seeds(seed)
    env.seed(seed)
    
    # Set Up Logger
    logdir = 'dqn_' + exp_name + '_' + env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = osp.join('data', logdir)
    logdir = osp.join(logdir, '%d'%seed)
    log.configure_output_dir(logdir)
    hyperparams = {'exp_name': exp_name, 'env_name': env_name}
    log.save_hyperparams(hyperparams)

    # expt_dir = '/tmp/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)
    # observation = env.reset()
    # print('observation shape', observation.shape)

    return env

def train_agent(args, env):
    max_num_timesteps = args.max_num_timesteps
    num_iterations = float(max_num_timesteps)/4.0
    
    train_flag = args.train
    test_flag = args.test
    
    lr_multiplier = args.lr_multiplier
    if test_flag:
        lr_multiplier = 0.0
    
    """
    1e-4, 5e-5, these values are also hyperparameters which can be tuned
    """
    lr_schedule = PiecewiseSchedule(
        [
            (0,                   1e-4 * lr_multiplier),
            (num_iterations / 10, 1e-4 * lr_multiplier),
            (num_iterations / 2,  5e-5 * lr_multiplier),
        ],
        outside_value=5e-5 * lr_multiplier
    )
    lr_lambda = lambda t: lr_schedule.value(t)

    optimizer = None
    optimizer_type = args.optimizer_type
    if optimizer_type == "Adam":
        optimizer = dqn.OptimizerSpec(
            constructor=torch.optim.Adam,
            kwargs=dict(eps=1e-4),
            lr_lambda=lr_lambda
        )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= max_num_timesteps

    exploration_mid_iter = args.max_num_timesteps/100 ### 100 is a hyperparameter
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (exploration_mid_iter, 0.1),
            (num_iterations / 2, 0.01),
        ],
        outside_value=0.01
    )

    replay_buffer_size = args.replay_buffer_size
    batch_size = args.batch_size
    gamma = args.gamma
    learning_starts = args.learning_starts
    learing_freq = args.learing_freq
    frame_history_len = args.frame_history_len
    target_update_freq = args.target_update_freq
    grad_norm_clipping = args.grad_norm_clipping
    double_dqn = args.policy_name
    double_q = False
    if double_dqn == "ddqn":
        double_q = True

    if train_flag:
        poison_flag = args.poison
        if poison_flag:
            model_input_dir = args.model_input_dir
            topk_ratio = args.poison_topk_ratio
            learner = POISON_QLEARNING(
                model_input_dir=model_input_dir,
                topk_ratio=topk_ratio,
                env=env,
                q_func=DQN,
                optimizer_spec=optimizer,
                exploration=exploration_schedule,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=replay_buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                learning_starts=learning_starts,
                learning_freq=learing_freq,
                frame_history_len=frame_history_len,
                target_update_freq=target_update_freq,
                grad_norm_clipping=grad_norm_clipping,
                double_q=double_q
            )
            while not learner.stopping_criterion_met():
                learner.step_env()
                learner.update_model()
                learner.log_process()

            env.close()

        if not poison_flag:
            learner = QLEARNING(
                env=env,
                q_func=DQN,
                optimizer_spec=optimizer,
                exploration=exploration_schedule,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=replay_buffer_size,
                batch_size=batch_size,
                gamma=gamma,
                learning_starts=learning_starts,
                learning_freq=learing_freq,
                frame_history_len=frame_history_len,
                target_update_freq=target_update_freq,
                grad_norm_clipping=grad_norm_clipping,
                double_q=double_q
            )
            while not learner.stopping_criterion_met():
                learner.step_env()
                learner.update_model()
                learner.log_process()

            env.close()
    
    
    if test_flag:
        
        model_input_dir = args.model_input_dir
        poison_model_input_dir = args.poison_model_input_dir
        topk_ratio = args.poison_topk_ratio
        learner = EVAL_POISON_QLEARNING(
            model_input_dir=model_input_dir,
            poison_model_input_dir = poison_model_input_dir,
            topk_ratio=topk_ratio,
            env=env,
            q_func=DQN,
            optimizer_spec=optimizer,
            exploration=exploration_schedule,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            learning_starts=learning_starts,
            learning_freq=learing_freq,
            frame_history_len=frame_history_len,
            target_update_freq=target_update_freq,
            grad_norm_clipping=grad_norm_clipping,
            double_q=double_q
        )
        while not learner.stopping_criterion_met():
            learner.step_env()
            learner.update_model()
            learner.log_process()

        env.close()

def main(args):

    env_name = args.env_name
    exp_name = args.exp_name
    seed = args.seed
    expt_dir = args.expt_dir

    env = get_env(expt_dir, env_name, exp_name, seed)
    train_agent(args, env)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--exp_name', type=str, default='Pong_double_dqn')
    parser.add_argument('--expt_dir', type=str, default="./tmp/")

    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--learning_starts', type=int, default=50000)
    parser.add_argument('--learing_freq', type=int, default=4)
    parser.add_argument('--frame_history_len', type=int, default=4)
    parser.add_argument('--target_update_freq', type=int, default=10000)
    parser.add_argument('--grad_norm_clipping', type=int, default=10)
    parser.add_argument('--policy_name', type=str, default="ddqn")

    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--max_num_timesteps', type=float, default=1e8)
    parser.add_argument('--lr_multiplier', type=int, default=1.0)
    parser.add_argument('--optimizer_type', type=str, default="Adam")

    parser.add_argument('--model_input_dir', type=str, default="")
    parser.add_argument('--poison_model_input_dir', type=str, default="")
    parser.add_argument('--poison_topk_ratio', type=float, default=0.15)

    parser.add_argument('--train', action="store_true", default=True)
    parser.add_argument('--test', action="store_true", default=False)

    args = parser.parse_args()

    main(args)
