import os

from tensorboardX import SummaryWriter
from tqdm import tqdm


class Play(object):
    def __init__(self, env, agent):
        self._agent = agent
        self._env = env
        self._brain_name = self._env.brain_names[0]

        self._writer = SummaryWriter(os.path.split(self._agent.checkpoint_path)[0])

    def eval(self):
        return self._run_env(False)

    def learn(self, nb_episodes=1000):
        scores = list()
        for _ in tqdm(range(nb_episodes)):
            score = self._run_env(True)
            scores.append(score)
            self._ep_summary(score)
            if score >= max(scores) or not scores:
                self._agent.save_model()

    def _run_env(self, train):
        env_info = self._env.reset(train_mode=train)[self._brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = self._agent.act(state, train)
            env_info = self._env.step(action)[self._brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if train:
                self._agent.observe(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break

        return score

    def _ep_summary(self, score):
        self._writer.add_scalar("train/reward", score, self._agent.step)

        for name, param in self._agent.parameters:
            self._writer.add_histogram(
                "main/" + name, param.clone().cpu().data.numpy(), self._agent.step)

        for name, param in self._agent.target_parameters:
            self._writer.add_histogram(
                "target/" + name, param.clone().cpu().data.numpy(), self._agent.step)
