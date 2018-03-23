import numpy as np

def simulate(env, model_cls, weights, num_trials, seed, render=False):
    m = model_cls(weights)
    rewards = []
    for _ in xrange(num_trials):
        observation = env.reset()
        done = False
        best_reward = -float("inf")
        ticks = 0

        while not done:
            if render:
                env.render()
            action = m.forward(observation)
            observation, reward, done, info = env.step(action)

            if reward > best_reward:
                best_reward = reward
                ticks = 0
            elif ticks == 250:
                break
            ticks += 1

        rewards.append(reward)

    return np.mean(rewards), range(num_trials)