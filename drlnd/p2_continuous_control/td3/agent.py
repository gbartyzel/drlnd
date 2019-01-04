import os

import torch
import torch.nn.functional as F

from drlnd.utils.memory import ReplayMemory
from drlnd.utils.exploration_noise import GaussianNoise
from drlnd.p2_continuous_control.td3.model import Actor, Critic


class SimpleAgent(object):
    def __init__(self, action_dim, state_dim):
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self._actor_network = Actor(action_dim, state_dim).to(self._device).share_memory()
        self._critic_network = Critic(action_dim, state_dim).to(self._device).share_memory()
        self._actor_network.train()
        self._critic_network.train()

        self._noise = GaussianNoise(action_dim, sigma=0.1)

    def act(self, state, train=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._actor_network.eval()
        with torch.no_grad():
            action = self._actor_network(state)[0]
        self._actor_network.train()

        if train:
            action = torch.clamp(action + self._noise().to(self._device), -1.0, 1.0)
        return action.cpu().data.numpy()

    def load_shared(self, state_dicts):
        self._actor_network.load_state_dict(state_dicts['actor'])
        self._critic_network.load_state_dict(state_dicts['critic'])


class TD3(SimpleAgent):
    def __init__(self, action_dim, state_dim, actor_lr, critic_lr, gamma, tau, n_step, buffer_size,
                 batch_size, update_frequency, warm_up_steps, logdir):
        super(TD3, self).__init__(action_dim, state_dim)
        
        self.step = 0
        self.gamma = gamma
        self.n_step = n_step

        self._action_dim = action_dim
        self._state_dim = state_dim

        self._tau = tau
        self._update_frequency = update_frequency
        self._warm_up_steps = warm_up_steps
        self.checkpoint_path = os.path.join(logdir, "checkpoint.pth")

        self._target_actor_network = Actor(action_dim, state_dim).to(self._device)
        self._actor_optim = torch.optim.Adam(self._actor_network.parameters(), actor_lr)

        self._target_critic_network = Critic(action_dim, state_dim).to(self._device)
        self._critic_optim = torch.optim.Adam(self._critic_network.parameters(), critic_lr)

        self.load_model()
        self._target_actor_network.load_state_dict(self._actor_network.state_dict())
        self._target_critic_network.load_state_dict(self._critic_network.state_dict())
        self._target_actor_network.eval()
        self._target_critic_network.eval()

        self._memory = ReplayMemory(buffer_size, batch_size, state_dim, action_dim)

        self._target_noise = GaussianNoise(action_dim, sigma=0.2)

    def observe(self, state, action, reward, next_state, done):
        self._memory.push(state, action, reward, next_state, done)
        if self._memory.size >= self._warm_up_steps:
            self.step += 1
            self._learn()

    def _learn(self):
        train_batch = self._memory.sample(self._device)
        state_batch = train_batch['obs1']
        action_batch = train_batch['u'].float()
        reward_batch = train_batch['r']
        next_state_batch = train_batch['obs2']
        done_batch = train_batch['d']

        noise = self._target_noise().clamp(-0.5, 0.5).to(self._device)
        next_action = self._target_actor_network(next_state_batch) + noise
        next_action = next_action.clamp(-1.0, 1.0)

        target_next_q1, target_next_q2 = self._target_critic_network(next_state_batch, next_action)
        target_next_q = torch.min(target_next_q1, target_next_q2).view(-1, 1).detach()

        target_q = reward_batch + (1.0 - done_batch) * self.gamma ** self.n_step * target_next_q
        expected_q1, expected_q2 = self._critic_network(state_batch, action_batch)

        loss_q = F.smooth_l1_loss(expected_q1, target_q) + F.smooth_l1_loss(expected_q2, target_q)
        self._critic_optim.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self._critic_network.parameters(), 10.0)
        self._critic_optim.step()
        del loss_q

        if self.step % self._update_frequency:
            actions = self._actor_network(state_batch)
            loss = -self._critic_network.evaluate_q1(state_batch, actions).mean()
            self._actor_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor_network.parameters(), 10.0)
            self._actor_optim.step()
            del loss

            self._soft_update(self._actor_network, self._target_actor_network)
            self._soft_update(self._critic_network, self._target_critic_network)

    def _soft_update(self, main_network, target_network):
        for target_param, param in zip(target_network.parameters(), main_network.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)

    def save_model(self):
        torch.save({
            'actor': self._actor_network.state_dict(),
            'critic': self._critic_network.state_dict()
        }, self.checkpoint_path)

    def load_model(self):
        if not os.path.isdir(os.path.split(self.checkpoint_path)[0]):
            os.makedirs(os.path.split(self.checkpoint_path)[0])
        if os.path.exists(self.checkpoint_path):
            models = torch.load(self.checkpoint_path)
            self._actor_network.load_state_dict(models['actor'])
            self._critic_network.load_state_dict(models['critic'])
            print("Model found and loaded!")

    @property
    def parameters(self):
        return list(self._actor_network.named_parameters()) + \
               list(self._critic_network.named_parameters())

    @property
    def target_parameters(self):
        return list(self._target_actor_network.named_parameters()) + \
               list(self._target_critic_network.named_parameters())

    @property
    def state_dicts(self):
        return {
            'actor': self._actor_network.state_dict(),
            'critic': self._critic_network.state_dict()
        }


class DistributedTD3(TD3):
    def __init__(self, nb_agents, worker_update_frequency, **kwargs):
        super(DistributedTD3, self).__init__(**kwargs)
        self._nb_agents = nb_agents
        self._worker_update_frequency = worker_update_frequency

        self._workers = nb_agents * [SimpleAgent(kwargs['action_dim'], kwargs['state_dim'])]
        for i in range(nb_agents):
            self._workers[i].load_shared(self.state_dicts)

    def act(self, state, train=False):
        actions = [self._workers[i].act(state[i], train) for i in range(self._nb_agents)]

        return actions

    def observe(self, state, action, reward, next_state, done):
        for i in range(self._nb_agents):
            self._memory.push(state[i], action[i], reward[i], next_state[i], done[i])

        if self._memory.size >= self._warm_up_steps:
            self.step += 1
            self._learn()
            self._update_worker()

    def _update_worker(self):
        if self.step % self._worker_update_frequency:
            for i in range(self._nb_agents):
                self._workers[i].load_shared(self.state_dicts)

