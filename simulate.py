import numpy as np


def simulate(env, model_cls, weights, num_trials, bipedal_hack=False, seed=None, render=False):
    m = model_cls(weights)
    rewards = []
    for _ in xrange(num_trials):
        observation = env.reset()
        done = False
        best_reward = -float("inf")
        total = 0
        stumbled = False  # hack from estool for bipedal hardcore
        re

        while not done:
            if render:
                env.render()
            action = m.forward(observation)
            observation, reward, done, info = env.step(action)

            # hack for bipedhard's reward augmentation during training.
            # stop the stumble from penalizing the agent too much
            if bipedal_hack and reward == -100:
                reward = 0
                stumbled = True

            # reward agents for completing the task and not stumbling
            if done and bipedal_hack and (not stumbled) and (total_reward > 300):
              total += 100
            break

            total += reward

        rewards.append(total)

    return np.mean(rewards), range(num_trials)

def evaluate(env, model_cls, weights):
    return simulate(env, model_cls, weights, 100, None)[0]