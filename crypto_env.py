# import os
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
# from pathlib import Path

import gym
from gym import Env
from gym import spaces
from gym.spaces import Space
# import gym_anytrading
# from gym_anytrading.envs import TradingEnv, Actions, Positions

# class TradeSpace(Space):
#
#     def __init__(self):
#         super().__init__()
#
#     def contains(self, action):
#         return action >= -1 and action <= 1
#
#     def sample(self):
#         return random.uniform(-1, 1)
class Market(Env):

    def __init__(self, market_data, initial_balance):
        # super(Market, self).__init__(market_data, initial_balance)

        self.df = market_data
        self.prices = self.df['close'].tolist()
        self.time = 0
        self.initial_balance = initial_balance

        # addons to market_data as state
        self.account_value = self.initial_balance
        self.total_cash = self.initial_balance
        self.units_held = 0

        # to keep track of valid actions and reward
        self.max_units_buy = self.total_cash / self.prices[self.time]
        self.max_units_sell = 0
        self.buys = []
        self.total_reward = 0
        # # action space some number x within [-1, 1]
        # self.action_space = TradeSpace()
        self.action_space = spaces.Discrete(3) # 0, 1, 2 -- hold buy sell

        # observation space is vector of market info & account_value, total_cash, units_held
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1] + 3, ))

        self.actions_executed = []


    def step(self, action):
        # returns observation, reward, done, info
        if action == 2:
            # converting index 2 to action -1 to represent sell
            action = -1

        # hold or invalid action
        if (action == 0) or (action > 1) or (action < -1):
            obs = self.df.loc[self.time + 1].tolist()
            self.account_value = self.total_cash + self.units_held * self.prices[self.time + 1]
            self.max_units_buy = self.total_cash / self.prices[self.time + 1]
            obs.append(self.account_value)
            obs.append(self.total_cash)
            obs.append(self.units_held)
            reward = 0
        elif action > 0 and action <= 1: # buying
            if self.total_cash != 0:
                buy_price = self.prices[self.time]
                units_bought = self.max_units_buy * action
                self.buys.append([buy_price, units_bought])
                self.units_held +=  units_bought
                self.total_cash -= units_bought * buy_price

            self.max_units_sell = self.units_held
            if self.time + 1 == len(self.df) - 1:
                self.max_units_buy = 0
                self.account_value = self.total_cash + self.units_held * self.prices[self.time]
            else:
                self.max_units_buy = self.total_cash / self.prices[self.time + 1]
                self.account_value = self.total_cash + self.units_held * self.prices[self.time + 1]

            obs = self.df.loc[self.time + 1].tolist()
            obs.append(self.account_value)
            obs.append(self.total_cash)
            obs.append(self.units_held)
            reward = 0
        else:
            obs = self.df.loc[self.time + 1].tolist()
            if self.units_held == 0: # cannot sell
                obs = self.df.loc[self.time + 1].tolist()
                if self.time + 1 == len(self.df) - 1:
                    self.max_units_buy = 0
                    self.account_value = self.total_cash + self.units_held * self.prices[self.time]
                else:
                    self.max_units_buy = self.total_cash / self.prices[self.time + 1]
                    self.account_value = self.total_cash + self.units_held * self.prices[self.time + 1]

                obs.append(self.account_value)
                obs.append(self.total_cash)
                obs.append(self.units_held)
                reward = 0
            else:
                units_to_sell = self.max_units_sell * abs(action)
                self.total_cash += units_to_sell * self.prices[self.time]
                sell_price = self.prices[self.time]

                profit = 0 # calculate profit
                lst = self.buys.copy()
                for idx, elem in reversed(list(enumerate(lst))):
                    buy_price, units_bought = elem
                    if units_to_sell > 0:
                        # print("Selling chunk. Units To Sell: ", units_to_sell)
                        if units_bought >= units_to_sell:
                            # print("Going to have leftover.")
                            # sell some of this buy
                            profit += (units_to_sell * sell_price) - (units_to_sell * buy_price)
                            self.buys[idx][1] -= units_to_sell
                            units_to_sell = 0
                            if self.buys[idx][1] == 0:
                                self.buys.pop(idx)
                        else:
                            # print("No more leftover")
                            # sell all of this buy
                            profit += (units_to_sell * sell_price) - (units_bought * buy_price)
                            units_to_sell -= units_bought
                            self.buys.pop(idx)

                self.units_held -= (self.max_units_sell * abs(action))
                if self.time + 1 == len(self.df) - 1:
                    self.max_units_buy = 0
                    self.account_value = self.total_cash + self.units_held * self.prices[self.time]
                else:
                    self.max_units_buy = self.total_cash / self.prices[self.time + 1]
                    self.account_value = self.total_cash + self.units_held * self.prices[self.time + 1]
                self.max_units_sell = self.units_held

                reward = profit
                obs.append(self.account_value)
                obs.append(self.total_cash)
                obs.append(self.units_held)
                reward = profit

        self.total_reward += reward
        self.time += 1
        self.actions_executed.append(action)

        if self.time == len(self.df) - 1:
            done = True
        else:
            done = False

        return obs, reward, done, {}

    def reset(self):
        self.time = 0
        self.account_value = self.initial_balance
        self.total_cash = self.initial_balance
        self.units_held = 0
        self.max_units_buy = self.total_cash / self.prices[self.time]
        self.max_units_sell = 0
        self.buys = []
        self.total_reward = 0
        self.actions_executed = []

        obs = self.df.loc[self.time].tolist()
        obs.append(self.account_value)
        obs.append(self.total_cash)
        obs.append(self.units_held)
        return obs

    def render(self):
        print("Price: {price}, Total Reward: {reward}, Account Value: {av}".format(price=self.prices[self.time], reward=self.total_reward, av=self.account_value))
        # return self.prices[self.time], self.total_reward, self.account_value

    def render_history(self):
        plt.plot(self.prices)
        buys = []
        sells = []
        for i in range(len(self.actions_executed)):
            action = self.actions_executed[i]
            if action == 1:
                buys.append(i)
            else:
                sells.append(i)

        buy_prices = [self.prices[j] for j in range(len(self.prices)) if j in buys]
        sell_prices = [self.prices[j] for j in range(len(self.prices)) if j in sells]

        plt.plot(buys, buy_prices, 'go')
        plt.plot(sells, sell_prices, 'ro')
