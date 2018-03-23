import gym
from models import Net
import sys
import json
from simulate import simulate

if __name__ ==  "__main__":
    gamename = 'BipedalWalker-v2'
    env = gym.make(gamename)

    filename = sys.argv[1]

    with open(filename, "r") as f:
        data = json.load(f)

        weights = data[0]

    simulate(env, Net, weights, 4, None, render=True)



