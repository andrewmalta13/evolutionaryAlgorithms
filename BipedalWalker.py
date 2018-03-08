import gym
import numpy as np
from evolution import evolve
import cma
import models

def run_worker(weights, num_trials, render=False):
    nn = Net(weights)

    rewards = []
    for _ in xrange(num_trials):
        observation = env.reset()
        done = False
        best_reward = -float("inf")
        ticks = 0

        while not done:
            if render:
                env.render()
            action = nn.forward(observation)
            observation, reward, done, info = env.step(action)

            if reward > best_reward:
                best_reward = reward
                ticks = 0
            elif ticks == 250:
                break
            ticks += 1

        rewards.append(reward)

    return np.mean(rewards)


if __name__ == "__main__":
    env = gym.make("BipedalWalker-v2")
    np.random.seed(123)
    generation_size = 25
    num_generations = 100
    theta_dimension = Net(None).num_flat_features()

    e = evolve(theta_dimension, num_generations, generation_size, run_worker, 1)
    weights = e.winner_es()

    # res = cma.fmin(run_worker, theta_dimension * [0], 0.5,  {'verb_disp': 0})
    nn = models.Net(weights)

    print("end of generation {} with a best reward of {}".format(num_generations, run_worker(nn, 1, True)))












