from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env_utils import load_flatland_environment_from_file
import numpy as np


def my_controller(obs, _env):
    _action = {}
    for _idx, _ in enumerate(_env.agents):
        _action[_idx] = np.random.randint(0, 5)
    return _action


my_observation_builder = GlobalObsForRailEnv()
episode = 0
env = load_flatland_environment_from_file("small.pkl")

for i in range(2):

    print("==============")
    episode += 1
    print("[INFO] EPISODE_START : {}".format(episode))
    # NO WAY TO CHECK service/self.evaluation_done in client

    obs, info = env.reset(True, True)
    if not obs:
        """
        The remote env returns False as the first obs
        when it is done evaluating all the individual episodes
        """
        print("[INFO] DONE ALL, BREAKING")
        break

    while True:
        action = my_controller(obs, env)
        try:
            obs, all_rewards, done, info = env.step(action)
        except:
            print("[ERR] DONE BUT step() CALLED")

        # break
        if done['__all__']:
            print("[INFO] EPISODE_DONE : ", episode)
            print("[INFO] TOTAL_REW: ", sum(list(all_rewards.values())))
            break

print("Evaluation Complete...")
