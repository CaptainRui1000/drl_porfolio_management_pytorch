"""
This script trains and saves the model and plots its performance.
"""

import ast
import argparse
import pprint
from collections import deque
import numpy as np
import os
import pandas as pd
import platform
from drl.ddpg_agent import Agent
from drl.env import PortfolioEnv
from time import time
from torch.utils.tensorboard import SummaryWriter
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU activated.")
else:
    device = torch.device("cpu")
    print("CPU activated.")


# ***************************************************************************************
def make_plot(output_dir, start_day, show=False):
    """Makes a pretty training plot call score.png.

    Args:
        output_dir (str):  Location to save output.
        start_day (int):  Date index when trading began.
        show (bool):  If True, show the image.  If False, save the image.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    # Load the trading history
    # ---------------------------------------------------------------------------------------
    history = pd.read_csv(os.path.join(output_dir, 'history.csv'), index_col=0)

    # Make a pretty plot
    # ---------------------------------------------------------------------------------------
    history.iloc[start_day-2:, :].plot(y=['portfolio', 'market'], use_index=True, figsize=(11, 3))
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'history.png'), dpi=200)
    plt.close()


# ***************************************************************************************
def offline_training(day):
    """This function performs the offline.

    Args:
        day (int):  The date index that the training is being called from.
    """

    # The target alpha based on the annual target and 252 trading days
    target = (1.0 + args.target) ** (args.days_per_epoch/252.0)
    if args.debug:
        print('Target = {}.'.format(target))

    # The minus 1 is critical to ensure the training does NOT get to see tomorrow's prices
    max_start_day = day - args.days_per_epoch - 1

    # The start date of each epoch is selected randomly with the probability skewed
    #     exponentially toward today.
    p = np.exp(args.memory_strength * np.arange(max_start_day) / max_start_day)
    p[:args.window_length + 1] = 0
    p = p / p.sum()

    # We train until we consistently beat the market or the max number of epochs reached.
    epoch_window = deque(maxlen=10)
    for e in range(args.max_epochs):
        state = env.reset(epoch_start=np.random.choice(max_start_day, size=None, p=p))
        for d in range(args.days_per_epoch):
            actions = agent.act(state=state)
            next_state, reward = env.step(actions)
            agent.step(state, actions, reward, next_state, d == (args.days_per_epoch - 1))
            state = next_state
        epoch_window.append(env.portfolio_value / env.market_value)

        if len(epoch_window) > 5:
            if np.mean(epoch_window) > target:
                break
            elif np.mean(epoch_window) < 0.9:
                epoch_window = deque(maxlen=10)
                agent.reset()
                print("Resetting agent.")
            elif args.debug:
                print(np.mean(epoch_window))


# ***************************************************************************************
def train():
    """This function trains the given agent in the given environment."""

    # logger = SummaryWriter(log_dir="./log")

    start_time = time()

    # The target alpha based on the annual target and 5 trading days
    target = (1.0 + args.target) ** (5.0/252.0) - 1.0

    # Perform the initial training
    # -----------------------------------------------------------------------------------
    print('Beginning initial training.')
    offline_training(args.start_day)

    # Begin the daily trading and retraining if required
    # -----------------------------------------------------------------------------------
    portfolio = np.ones(env.n_dates + 1)
    market = np.ones(env.n_dates + 1)
    returns = np.zeros(env.n_dates + 1)
    alphas = np.zeros(env.n_dates + 1)
    # mdd = np.zeros(env.n_dates + 1)

    weights = np.insert(np.zeros(env.n_assets), 0, 1.0)
    delta_days = 5
    days_after_train = 0
    for day in range(args.start_day, env.n_dates):
        days_after_train += 1

        # Make the real trade for today (you only get to do this once)
        state = env.reset(epoch_start=day, portfolio_value=portfolio[day], market_value=market[day], weights=weights)
        actions = agent.act(state=state)
        next_state, reward = env.step(actions)
        agent.step(state, actions, reward, next_state, done=True)

        # Save tomorrow's portfolio and market values
        portfolio[day + 1] = env.portfolio_value
        market[day + 1] = env.market_value
        returns[day + 1] = env.returns
        weights = env.weights

        # TensorBroad
        # logger.add_scalar("Actor Loss", agent.log[day-args.start_day][0], global_step=day-args.start_day)
        # logger.add_scalar("Critic Loss", agent.log[day-args.start_day][1], global_step=day-args.start_day)
        # logger.add_scalar("Rewards", reward, global_step=day-args.start_day)

        # Print some info to screen to color the drying paint

        if day % 10 == 0:
            print('Day {} p: {} m : {}'.format(day, portfolio[day + 1], market[day + 1]))
            print('p/m ratio: {:.4f}'.format(portfolio[day + 1] / market[day + 1]))
            duration = (time() - start_time) / 60
            print('{:.2f} minutes of training.'.format(duration))

        print(day % 10 * '#')

        # Retrain if we aren't beating the market over last five days
        alpha = (portfolio[day + 1] - portfolio[day - delta_days + 1]) / portfolio[day - delta_days + 1] - \
                (market[day + 1] - market[day - delta_days + 1]) / market[day - delta_days + 1]
        alphas[day + 1] = alpha
        if (alpha <= target) and (days_after_train > delta_days):
            offline_training(day)
            days_after_train = 0

    # Print the final information for curiosity and hyperparameter tuning
    alpha = (1 + portfolio[-1] - market[-1]) ** (252.0 / (env.n_dates - args.start_day)) - 1.0
    print('{:.2f} annualized alpha.'.format(alpha))
    duration = (time() - start_time)/60
    print('{:.2f} minutes of training.'.format(duration))

    # The objective will be minimized so there is a big penalty for missing the target
    objective = 100000 * np.max([0.0, args.target - alpha])

    # Faster is better so there is a penalty for run time
    objective += duration

    # And a little boost for alpha above the target
    objective -= 1000 * np.max([0.0, alpha - args.target])

    print('{:.2f} training objective.'.format(objective))

    # Save models weights and training history
    # -----------------------------------------------------------------------------------
    for p in [p for p in [args.model_dir, args.output_dir] if not os.path.isdir(p)]:
        os.mkdir(p)
    torch.save(agent.actor_target.state_dict(), os.path.join(args.model_dir, 'checkpoint_actor.pth'))
    torch.save(agent.critic_target.state_dict(), os.path.join(args.model_dir, 'checkpoint_critic.pth'))
    history = pd.DataFrame(index=env.dates, data={'portfolio': portfolio, 'market': market, 'returns': returns,
                           'alphas': alphas})
    history.to_csv(os.path.join(args.output_dir, 'history.csv'))

    # logger.flush()
    # logger.close()


# ***************************************************************************************
if __name__ == '__main__':

    # Read the arguments
    # -----------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # These are general setting
    parser.add_argument('--prices_name', type=str, default='prices-2022-Aug-raw.csv',
                        help='the csv file name containing the price history (default: prices-2022-Aug-train.csv)')
    parser.add_argument('--signals_name', type=str, default='signals-2022-Aug-raw.csv',
                        help='the csv file name containing the signals (default: signals-2022-Aug-train.csv)')
    parser.add_argument('--trading_cost', type=float, default=0.00025, help='trading cost (default: 0.0025)')
    parser.add_argument('--debug', type=int, default=0, help='print debug if != 0 (default: 0)')
    parser.add_argument('--predict_type', type=str, default='cnn', help='predict type (default: cnn)')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='Whether load the pretrain parameters (default: True)')

    # These are hyperparameters that could be tuned
    parser.add_argument('--max_epochs', type=int, default=2000, help='max epochs per new trading day (default: 2000)')
    parser.add_argument('--days_per_epoch', type=int, default=40, help='days in each epoch (default: 40)')
    parser.add_argument('--start_day', type=int, default=2314, help='day to begin training (default: 301)')
    parser.add_argument('--window_length', type=int, default=10, help='inputs window length (default: 10)')
    parser.add_argument('--memory_strength', type=float, default=2.0, help='memory exponential gain (default: 2.0)')
    parser.add_argument('--target', type=float, default=0.05, help='target annual alpha (default: 0.05)')
    parser.add_argument('--fc1', type=int, default=13, help='size of 1st hidden layer (default: 13)')
    parser.add_argument('--fc2', type=int, default=0, help='size of 2nd hidden layer (default: 0)')
    parser.add_argument('--lr_actor', type=float, default=0.00056, help='actor learning rate (default: 0.00039)')
    parser.add_argument('--lr_critic', type=float, default=0.0059, help='critic learning rate (default: 0.0036)')
    parser.add_argument('--batch_size', type=int, default=256, help='mini batch size (default: 256)')
    parser.add_argument('--buffer_size', type=int, default=int(1e5), help='replay buffer size (default: 10,000)')
    parser.add_argument('--gamma', type=float, default=0.93, help='discount factor (default: 0.91)')
    parser.add_argument('--tau', type=float, default=0.0083, help='soft update of target parameters (default: 0.0072)')
    parser.add_argument('--sigma', type=float, default=0.0, help='OU Noise standard deviation (default: 0.0)')
    parser.add_argument('--theta', type=float, default=0.0, help='OU Noise theta gain (default: 0.0)')

    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--model_dir', type=str, default='./weights/model',
                        help='where the trained model should be saved')
    parser.add_argument('--input_dir', type=str, default='./data',
                        help='where SageMaker will place the training data')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='where miscellaneous files should be saved')
    args = parser.parse_args()

    args.pretrain = False

    pprint.pprint(args)

    # Set up the training environment
    # -----------------------------------------------------------------------------------
    print('Setting up the environment.')
    env = PortfolioEnv(prices_name=args.prices_name, signals_name=args.signals_name,
                       trading_cost=args.trading_cost, window_length=args.window_length)
    print('Number of assets: {}'.format(env.n_assets))
    print('Signals per asset: {}'.format(env.n_signals))

    # Create the reinforcement learning agent
    # -----------------------------------------------------------------------------------
    agent = Agent(n_assets=env.n_assets, n_signals=env.n_signals,
                  window_length=args.window_length, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                  batch_size=args.batch_size, buffer_size=args.buffer_size, gamma=args.gamma, tau=args.tau,
                  sigma=args.sigma, theta=args.theta, fc1=args.fc1, fc2=args.fc2, pretrain=args.pretrain,
                  random_seed=20396662, predict_type=args.predict_type)

    # Perform the training
    # -----------------------------------------------------------------------------------
    print('Training the agent.')
    start = time()
    train()
    print("Training Time:  {:.1f} minutes".format((time() - start)/60.0))

    # Make some pretty plots
    # -----------------------------------------------------------------------------------
    print('Make training plot.')
    make_plot(output_dir=args.output_dir, start_day=args.start_day)
