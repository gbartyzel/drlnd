import os

import torch

from drlnd.utils.exploration_noise import AdaptiveGaussianNoise, OUNoise
from drlnd.utils.memory import ReplayMemory
from drlnd.p3_collab_compet.maddpg.model import Actor, Critic


class SimpleDDPG(object):
    def __init__(self, action_dim, state_dim, actor_lr, critic_lr, index, nb_agents,
                 buffer_size, batch_size, exploration_factor, exploration_decay, logdir):
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self._action_dim = action_dim
        self._state_dim = state_dim
        self._index = index
        self._exploration_factor = exploration_factor

        self.checkpoint_path = os.path.join(logdir, "checkpoint.pth".format(index))

        self.actor_network = Actor(action_dim, state_dim).to(self._device)
        self.target_actor_network = Actor(action_dim, state_dim).to(self._device)
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), actor_lr)

        c_action_dim = action_dim * nb_agents
        c_state_dim = state_dim * nb_agents
        self.critic_network = Critic(c_action_dim, c_state_dim).to(self._device)
        self.target_critic_network = Critic(c_action_dim, c_state_dim).to(self._device)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), critic_lr)

        self.load_model()
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        self.target_actor_network.eval()
        self.target_critic_network.eval()

        self.memory = ReplayMemory(buffer_size, batch_size, state_dim, action_dim)

        self.noise = AdaptiveGaussianNoise(action_dim, 0.0, 1.0, 0.1, exploration_decay)

    def act(self, state, train=False, warm_up=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self.actor_network.eval()
        with torch.no_grad():
            action = self.actor_network(state)[0]
        self.actor_network.train()

        if train:
            noise = self.noise(warm_up).to(self._device)
            action = torch.clamp(action + self._exploration_factor * noise, -1.0, 1.0)
        return action.cpu().data.numpy()

    def target_act(self, state):
        return self.target_actor_network(state)

    def load_model(self):
        if os.path.exists(self.checkpoint_path):
            models = torch.load(self.checkpoint_path)
            self.actor_network.load_state_dict(models['actor_{}'.format(self._index)])
            self.critic_network.load_state_dict(models['critic_{}'.format(self._index)])
            print("Model found and loaded!")

    @property
    def parameters(self):
        return list(self.actor_network.named_parameters()) + \
               list(self.critic_network.named_parameters())

    @property
    def target_parameters(self):
        return list(self.target_actor_network.named_parameters()) + \
               list(self.target_critic_network.named_parameters())

    @property
    def state_dicts(self):
        return {
            'actor_{}'.format(self._index): self.actor_network.state_dict(),
            'critic_{}'.format(self._index): self.critic_network.state_dict()
        }
