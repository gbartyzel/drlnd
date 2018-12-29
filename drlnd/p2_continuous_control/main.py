import argparse
from drlnd.utils.play import Play
from drlnd.p2_continuous_control.td3.agent import TD3, DistributedTD3
from unityagents import UnityEnvironment


def parser_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--nb_episodes', type=int, default=500)
    parser.add_argument('--use_distributed', action='store_true')
    parser.add_argument('--nb_agents', type=int, default=20)
    parser.add_argument('--update_frequency', type=int, default=2)
    parser.add_argument('--worker_update_frequency', type=int, default=100)
    parser.add_argument('--tau', type=int, default=0.005)
    parser.add_argument('--n_step', type=int, default=5)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--warm_up_steps', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)

    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


def main(train, nb_episodes, use_distributed, **kwargs):
    env = UnityEnvironment(file_name="./env/Reacher.x86_64")
    if use_distributed:
        agent = DistributedTD3(state_dim=33, action_dim=4, **kwargs)
    else:
        kwargs.pop('nb_agents')
        kwargs.pop('worker_update_frequency')
        agent = TD3(state_dim=33, action_dim=4, **kwargs)
    play = Play(env, agent, use_distributed)

    if train:
        play.learn(nb_episodes)
    play.eval()


if __name__ == '__main__':
    args = parser_setup()
    main(**args)
