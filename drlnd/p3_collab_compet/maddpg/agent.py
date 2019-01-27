import os
import numpy as np
import torch
import torch.nn.functional as F

from collections import ChainMap
from drlnd.p3_collab_compet.maddpg.simple_agent import SimpleDDPG


class MADDPG(object):
    def __init__(self, action_dim, state_dim, nb_agents, actor_lr, critic_lr, gamma, tau, n_step,
                 buffer_size, batch_size, update_frequency, warm_up_steps, logdir):
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.step = 0
        self.gamma = gamma
        self.n_step = n_step

        self._nb_agents = nb_agents
        self._tau = tau
        self._update_frequency = update_frequency
        self._warm_up_steps = warm_up_steps
        self._batch_size = batch_size
        self._decay = False
        self.checkpoint_path = os.path.join(logdir, "checkpoint.pth")

        self._agenets = [SimpleDDPG(action_dim, state_dim, actor_lr, critic_lr, i, nb_agents,
                                    buffer_size, batch_size, logdir) for i in range(nb_agents)]

    def act(self, state, train=False):
        return [self._agenets[i].act(state[i], train, self._decay) for i in range(self._nb_agents)]

    def observe(self, state, action, reward, next_state, done):
        for i in range(self._nb_agents):
            self._agenets[i].memory.push(state[i], action[i], reward[i], next_state[i], done[i])

        # if any(done):
        #     for i in range(self._nb_agents):
        #         self._agenets[i].noise.reset()

        if self._agenets[0].memory.size >= self._warm_up_steps:
            self._decay = True
            self.step += 1
            self._learn()

    def _learn(self):
        idxs = np.random.randint((self._agenets[0].memory.size - 1), size=self._batch_size)
        train_batch = [self._agenets[i].memory.sample(self._device, idxs)
                       for i in range(self._nb_agents)]

        state_batch = [train_batch[i]['obs1'] for i in range(self._nb_agents)]
        reward_batch = [train_batch[i]['r'] for i in range(self._nb_agents)]
        next_state_batch = [train_batch[i]['obs2'] for i in range(self._nb_agents)]
        done_batch = [train_batch[i]['d'] for i in range(self._nb_agents)]

        all_action_batch = torch.cat([train_batch[i]['u'] for i in range(self._nb_agents)], 1)
        all_state_batch = torch.cat(state_batch, 1)
        all_next_state_batch = torch.cat(next_state_batch, 1)

        for i in range(self._nb_agents):
            next_actions = [self._agenets[j].target_act(next_state_batch[j])
                            for j in range(self._nb_agents)]

            next_actions = torch.cat(next_actions, 1)

            with torch.no_grad():
                target_next_q = self._agenets[i].target_critic_network(all_next_state_batch,
                                                                       next_actions)

            reward_temp = reward_batch[i].view(-1, 1)
            done_temp = done_batch[i].view(-1, 1)
            target_q = reward_temp + (1.0 - done_temp) * self.gamma * target_next_q
            expected_q = self._agenets[i].critic_network(all_state_batch, all_action_batch)

            self._agenets[i].critic_optim.zero_grad()
            loss_q = F.mse_loss(expected_q, target_q.detach())
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self._agenets[i].critic_network.parameters(), 0.5)
            self._agenets[i].critic_optim.step()

            actions = [self._agenets[j].actor_network(state_batch[i])
                       for j in range(self._nb_agents)]
            actions = torch.cat(actions, 1)

            loss_u = -self._agenets[i].critic_network(all_state_batch, actions).mean()
            self._agenets[i].actor_optim.zero_grad()
            loss_u.backward()
            torch.nn.utils.clip_grad_norm_(self._agenets[i].critic_network.parameters(), 0.5)
            self._agenets[i].actor_optim.step()

            self._soft_update(self._agenets[i].actor_network,
                              self._agenets[i].target_actor_network)
            self._soft_update(self._agenets[i].critic_network,
                              self._agenets[i].target_critic_network)

    def _soft_update(self, main_network, target_network):
        for target_param, param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)

    def save_model(self):
        state_dicts = ChainMap(*[self._agenets[i].state_dicts for i in range(self._nb_agents)])
        torch.save(state_dicts, self.checkpoint_path)

    @property
    def parameters(self):
        params = list()
        for i in range(self._nb_agents):
            for name, param in self._agenets[i].parameters:
                params.append(("{}/".format(i) + name, param))
        return params

    @property
    def target_parameters(self):
        params = list()
        for i in range(self._nb_agents):
            for name, param in self._agenets[i].target_parameters:
                params.append(("{}/".format(i) + name, param))
        return params
