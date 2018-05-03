import numpy as np


def simulate(env, model_cls, weights, num_trials, seed=None, render=False):
    m = model_cls(weights)
    rewards = []
    for _ in xrange(num_trials):
        observation = env.reset()
        done = False
        best_reward = -float("inf")
        total = 0

        while not done:
            if render:
                env.render()
            action = m.forward(observation)
            observation, reward, done, info = env.step(action)

            total += reward

        rewards.append(total)

    return np.mean(rewards), range(num_trials)

def evaluate(env, model_cls, weights):
    return simulate(env, model_cls, weights, 100, None)[0]