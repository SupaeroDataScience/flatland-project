from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import logging

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', help='environment', default='small', type=str)
    parser.add_argument('--render', help='display the environment', default=False, type=bool)
    parser.add_argument('--weights', help='filename to load policy weights', default='weights', type=str)
    args = parser.parse_args()

    # starting point
    env, params = get_env(args.env)
    s, a = get_state_action_size(env)
    policy = NeuroevoPolicy(s, a)
    policy.load(args.weights)
    print('Weights: ', policy.get_params()[:5])

    mean_fit = 0
    for i in range(3):
        params["seed"] = i
        fit = evaluate(env, params, policy, render=args.render)
        mean_fit += fit
        print('Fitness: ', fit)
    mean_fit /= 3.0
    print('Mean fit: ', mean_fit)
