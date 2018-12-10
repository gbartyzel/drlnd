import os
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm


class Play(object):
    def __init__(self, env, dqn):
        self._dqn = dqn
        self._env = env
        self._brain_name = self._env.brain_names[0]

        self._writer = SummaryWriter(os.path.split(self._dqn.checkpoint_path)[0])

    def eval(self):
        return self._run_env(False)

    def learn(self, nb_episodes=1000, stable_episodes=100, mean_score=13.0):
        scores = list()
        for ep in tqdm(range(nb_episodes)):
            score = self._run_env(True)
            scores.append(score)
            self._ep_summary(score)
            if score >= max(scores) or not scores:
                self._dqn.save_model()
            if len(scores) > stable_episodes:
                if np.mean(scores[ep - stable_episodes:]) > mean_score:
                    print("Mean score: {} achieved after {} episodes".format(
                        np.mean(scores[ep-stable_episodes:]), ep))
                    break

    def _run_env(self, train):
        env_info = self._env.reset(train_mode=train)[self._brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = self._dqn.act(state, train)
            env_info = self._env.step(action)[self._brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if train:
                self._dqn.observe(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break

        return score

    def _ep_summary(self, score):
        self._writer.add_scalar("train/reward", score, self._dqn.step)

        for name, param in self._dqn.parameters:
            self._writer.add_histogram(
                "main/" + name, param.clone().cpu().data.numpy(), self._dqn.step)

        for name, param in self._dqn.target_parameters:
            self._writer.add_histogram(
                "target/" + name, param.clone().cpu().data.numpy(), self._dqn.step)
