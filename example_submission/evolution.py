from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import numpy as np
import logging

def oneplus_lambda(x, fitness, gens, lam, std=0.1, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for g in range(gens):
        N = rng.normal(size=(lam, len(x))) * std
        for i in range(lam):
            ind = x + N[i, :]
            f = fitness(ind)
            if f > f_best:
                f_best = f
                x_best = ind
        x = x_best
        n_evals += lam
        logging.info('\t%d\t%d', n_evals, f_best)
    return x_best


def fitness(x, s, a, env, params):
    policy = NeuroevoPolicy(s, a)
    policy.set_params(x)
    return evaluate(env, params, policy)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', help='environment', default='small', type=str)
    parser.add_argument('-g', '--gens', help='number of generations', default=100, type=int)
    parser.add_argument('-p', '--pop', help='population size (lambda for the 1+lambda ES)', default=10, type=int)
    parser.add_argument('-s', '--seed', help='seed for evolution', default=0, type=int)
    parser.add_argument('--log', help='log file', default='evolution.log', type=str)
    parser.add_argument('--weights', help='filename to save policy weights', default='weights', type=str)
    args = parser.parse_args()
    logging.basicConfig(filename=args.log, encoding='utf-8', level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    # starting point
    env, params = get_env(args.env)
    s, a = get_state_action_size(env)
    policy = NeuroevoPolicy(s, a)

    # evolution
    rng = np.random.default_rng(args.seed)
    #start = rng.normal(size=(len(policy.get_params(),)))
    start = policy.get_params()

    def fit(x):
        return fitness(x, s, a, env, params)
    x_best = oneplus_lambda(start, fit, args.gens, args.pop, rng=rng)

    # Evaluation
    policy.set_params(x_best)
    policy.save(args.weights)
    best_eval = evaluate(env, params, policy)
    print('Best individual: ', x_best[:5])
    print('Fitness: ', best_eval)
