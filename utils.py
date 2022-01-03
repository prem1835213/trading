import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

def train_dqn(dqn, env, batch_size, discount, eps_init, eps_decay, eps_min, num_actions, n_episodes):
    optimizer = optim.Adam(dqn.parameters())
    criterion = nn.MSELoss()
    epsilon = eps_init
    for episode in range(n_episodes):
        print("\nEpisode: ", episode)
        s = env.reset()
        time = 0
        while s is not None:
            # env.render()
            s = torch.FloatTensor(s)
            if random.random() < epsilon: # explore
                a = random.randrange(num_actions)
            else: # exploit
                with torch.no_grad():
                    a = dqn(s).argmax().item()

            s_next, r, done, info = env.step(a)
            s_next = torch.FloatTensor(s_next)
            r = torch.tensor(r)
            dqn.remember([s, a, r, s_next, done])

            # sample and perform update
            if len(dqn.memory) >= batch_size:
                batch = dqn.memory.sample(batch_size)

                batch_states = [elem[0] for elem in batch]
                batch_states = torch.stack(batch_states)

                batch_actions = torch.tensor([elem[1] for elem in batch]).view(batch_size, 1)

                batch_next_states = [elem[3] for elem in batch]
                batch_next_states = torch.stack(batch_next_states)

                batch_rewards = torch.FloatTensor([elem[2] for elem in batch])

                mask = [j for j in range(batch_size) if batch[j][4] == False]
                predicted_rewards = torch.gather(dqn(batch_states), 1, batch_actions).view(batch_size, )

                label_rewards = batch_rewards
                try:
                    label_rewards[mask] += discount * torch.max(dqn(batch_next_states[mask]), 1).values
                except:
                    print(label_rewards[mask].shape)
                    print(label_rewards.shape)
                    print(torch.max(dqn(batch_next_states), 1).values.shape)

                optimizer.zero_grad()
                loss = criterion(predicted_rewards, label_rewards)
                loss.backward()
                optimizer.step()

                # for sequence in batch:
                #     batch_state, batch_action, batch_reward, batch_next_state, batch_done = sequence
                #
                #     if batch_done:
                #         exp_reward = batch_reward
                #     else:
                #         exp_reward = batch_reward + discount * dqn(batch_next_state).max()
                #
                #     predicted_reward = dqn(batch_state)[batch_action]
                #     optimizer.zero_grad()
                #     loss = criterion(predicted_reward, exp_reward)
                #     loss.backward()
                #     optimizer.step()

            if epsilon > eps_min:
                epsilon *= eps_decay

            time += 1
            if done:
                s = None
                # print("Episode Duration ", time)
            else:
                s = s_next

        env.render()
    env.close()

def train_ac(actor, critic, env, discount):

    ac_opt = optim.Adam(actor.parameters(), lr=0.001)
    critic_opt = optim.Adam(critic.parameters(), lr=0.001)
    critic_criterion = nn.MSELoss()

    for ep in range(1000):
        s = torch.FloatTensor(env.reset()).detach()
        I = 1
        # env.render()
        while s is not None:
            try:
                m = actor(s)
            except:
                print(s)
                return
            a = m.sample()
            lp = m.log_prob(a)

            s_next, r, done, info = env.step(a.numpy())
            print("Action: ", a)
            env.render()
            s_next = torch.FloatTensor(s_next)

            if done:
                v_next = 0
            else:
                v_next = critic(s_next)

            ac_opt.zero_grad()
            actor_loss = r + discount * critic(s) - v_next
            actor_loss *= I * -lp
            actor_loss.backward(retain_graph=True)
            ac_opt.step()

            critic_opt.zero_grad()
            critic_loss = critic_criterion(r + discount * critic(s), v_next)
            critic_loss *= I
            critic_loss.backward()
            critic_opt.step()

            I *= discount

            if done:
                s = None
            else:
                s = s_next

    env.close()
