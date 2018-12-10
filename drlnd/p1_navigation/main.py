import argparse
from drlnd.p1_navigation.play import Play
from drlnd.p1_navigation.dqn.agent import Agent
from unityagents import UnityEnvironment


def parser_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--nb_episodes', type=int, default=1000)
    parser.add_argument('--n_step_annealing', type=int, default=100000)
    parser.add_argument('--epsilon_min', type=float, default=0.05)
    parser.add_argument('--update_frequency', type=int, default=4)
    parser.add_argument('--target_update_frequency', type=int, default=1000)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warm_up_steps', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lrate', type=float, default=1e-4)
    parser.add_argument('--use_noisynet', action='store_true')
    parser.add_argument('--use_double_q', action='store_true')
    parser.add_argument('--use_dueling', action='store_true')

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def main(train, nb_episodes, **kwargs):
    env = UnityEnvironment(file_name="./env/Banana.x86_64")
    dqn = Agent(state_dim=37, action_dim=4, **kwargs)
    play = Play(env, dqn)

    if train:
        play.learn(nb_episodes, mean_score=14.0)
    play.eval()


if __name__ == '__main__':
    args = parser_setup()
    main(**args)
