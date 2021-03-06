import gym
from models import Net
import sys
import json
from simulate import simulate, evaluate
from gym.wrappers import Monitor

if __name__ ==  "__main__":
    # gamename = 'BipedalWalker-v2'
    gamename = 'BipedalWalkerHardcore-v2'
    env = gym.make(gamename)

    command = sys.argv[1]
    filename = sys.argv[2]

    with open(filename, "r") as f:
        data = json.load(f)
        weights = data[0]

    if command == "simulate":
        simulate(env, Net, weights, 4, None, render=True)
    elif command == "evaluate":
        print evaluate(env, Net, weights)
    elif command == "record":
        env = Monitor(env, directory='/tmp/' + gamename,
                      video_callable=lambda id: True,
                      write_upon_reset=True, force=True)
        simulate(env, Net, weights, 1, None, render=True)
    else:
        print "invalid command [simulate | evaluate]"



