import os

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Play(object):
    def __init__(self, env, agent, distributed):
        self._agent = agent
        self._env = env
        self._distributed = distributed
        self._brain_name = self._env.brain_names[0]

        self._writer = SummaryWriter(os.path.split(self._agent.checkpoint_path)[0])

    def eval(self):
        if self._distributed:
            return self._run_distributed_env(False)
        return self._run_env(False)

    def learn(self, nb_episodes=1000):
        scores = list()
        for _ in tqdm(range(nb_episodes)):
            if self._distributed:
                score = self._run_distributed_env(True)
            else:
                score = self._run_env(True)
            scores.append(score)
            self._ep_summary(score)
            if score >= max(scores) or not scores:
                self._agent.save_model()

    def _run_distributed_env(self, train):
        episode_state = list()
        episode_reward = list()
        episode_action = list()

        env_info = self._env.reset(train_mode=train)[self._brain_name]
        state = env_info.vector_observations

        episode_state.append(state)

        while True:
            action = self._agent.act(state, train)
            env_info = self._env.step(action)[self._brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            episode_state.append(state)
            episode_reward.append(reward)
            episode_action.append(action)

            if train and len(episode_reward) >= self._agent.n_step:
                discount = self._agent.gamma ** np.arange(self._agent.n_step)
                temp_reward = np.vstack(episode_reward[-self._agent.n_step:])

                cum_reward = [
                    np.sum(np.asarray(temp_reward[:, i]) * discount)
                    for i in range(len(env_info.agents))
                ]

                self._agent.observe(
                    episode_state[-self._agent.n_step], episode_action[-self._agent.n_step],
                    cum_reward, next_state, done)

            state = next_state
            if any(done):
                break

        return np.sum(np.mean(np.vstack(episode_reward), axis=1))

    def _run_env(self, train):
        episode_reward = list()
        episode_state = list()
        episode_action = list()

        env_info = self._env.reset(train_mode=train)[self._brain_name]
        state = env_info.vector_observations[0]
        episode_state.append(state)

        while True:
            action = self._agent.act(state, train)
            env_info = self._env.step(action)[self._brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            episode_state.append(state)
            episode_reward.append(reward)
            episode_action.append(action)

            if train and len(episode_reward) > self._agent.n_step:
                discount = self._agent.gamma ** np.arange(self._agent.n_step)
                cum_reward = np.sum(np.asarray(episode_reward[-self._agent.n_step:]) * discount)
                self._agent.observe(
                    episode_state[-self._agent.n_step], episode_action[-self._agent.n_step],
                    cum_reward, next_state, done)

            state = next_state
            if done:
                break

        return np.sum(episode_reward)

    def _ep_summary(self, score):
        self._writer.add_scalar("train/reward", score, self._agent.step)

        for name, param in self._agent.parameters:
            self._writer.add_histogram(
                "main/" + name, param.clone().cpu().data.numpy(), self._agent.step)

        for name, param in self._agent.target_parameters:
            self._writer.add_histogram(
                "target/" + name, param.clone().cpu().data.numpy(), self._agent.step)
