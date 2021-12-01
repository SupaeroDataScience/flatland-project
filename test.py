from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import logging

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', help='environment', default='small', type=str)
    parser.add_argument('--weights', help='filename to load policy weights', default='weights', type=str)
    args = parser.parse_args()

    # starting point
    env, params = get_env(args.env)
    s, a = get_state_action_size(env)
    policy = NeuroevoPolicy(s, a)
    policy.load(args.weights)

    fit = evaluate(env, params, policy, render=True)
    print('Weights: ', policy.get_params()[:5])
    print('Fitness: ', fit)
