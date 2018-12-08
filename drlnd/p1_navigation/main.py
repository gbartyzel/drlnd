import argparse
from drlnd.p1_navigation.play import Play
from drlnd.p1_navigation.dqn.agent import Agent
from unityagents import UnityEnvironment


def parser_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--nb_episodes', type=int, default=1000)
    parser.add_argument('--n_step_annealing', type=int, default=1000000)
    parser.add_argument('--epsilon_min', type=float, default=0.05)
    parser.add_argument('--update_frequency', type=int, default=4)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lrate', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--noisynet', action='store_true')
    parser.add_argument('--double_q', action='store_true')
    parser.add_argument('--dueling', action='store_true')

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def main(nb_episodes, **kwargs):
    env = UnityEnvironment(file_name="./env/Banana.x86_64")
    dqn = Agent(**kwargs)
    play = Play(env, dqn, nb_episodes)
    play.learn(100, 15.0)


if __name__ == '__main__':
    args = parser_setup()
