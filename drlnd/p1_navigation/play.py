import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm


class Play(object):
    def __init__(self, env, dqn, nb_episodes=1):
        self._dqn = dqn
        self._env = env
        self._brain_name = self._env.brain_names[0]

        self._nb_episodes = nb_episodes
        self._writer = SummaryWriter()

    def learn(self, stable_episodes=100, mean_score=13.0):
        scores = list()
        for ep in tqdm(range(self._nb_episodes)):
            score, q_values = self._run_env(True)
            self._ep_summary(score, q_values)
            if score >= max(scores):
                self._dqn.save_model()
            if len(scores) > stable_episodes:
                if np.mean(scores[ep - stable_episodes:]) > mean_score:
                    break

    def _run_env(self, train):
        env_info = self._env.reset(train_mode=train)[self._brain_name]
        state = env_info.vector_observations[0]
        score = 0
        q_values = list()
        while True:
            action, q = self._dqn.act(state, train)
            env_info = self._env.step(action)[self._brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if train:
                self._dqn.observe(state, action, reward, next_state, done)
            score += reward
            q_values.append(q)
            state = next_state
            if done:
                break

        return score, q_values

    def _ep_summary(self, score, q_values):
        self._writer.add_scalar("train/reward", score, self._dqn.step)
        self._writer.add_scalar("train/q_max", max(q_values), self._dqn.step)
        self._writer.add_scalar("train/q_min", min(q_values), self._dqn.step)
        self._writer.add_scalar("train/q_mean", np.mean(q_values), self._dqn.step)

        for name, param in self._dqn.parameters:
            self._writer.add_histogram(name, param.clone().cpu().data.numpy(), self._dqn.step)

        for name, param in self._dqn.target_parameters:
            self._writer.add_histogram(name, param.clone().cpu().data.numpy(), self._dqn.step)
