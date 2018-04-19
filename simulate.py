import numpy as np


def simulate(env, model_cls, weights, num_trials, seed=None, render=False):
    m = model_cls(weights)
    rewards = []
    for _ in xrange(num_trials):
        observation = env.reset()
        done = False
        best_reward = -float("inf")
        total = 0
        bipedal_hack=False
        stumbled = False  # hack from estool for bipedal hardcore

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
            # and encourage them to go faster
            if done and bipedal_hack and (not stumbled) and (total > 200):
                total += 100
                break

            total += reward

        rewards.append(total)

    return np.mean(rewards), range(num_trials)

def evaluate(env, model_cls, weights):
    return simulate(env, model_cls, weights, 100, None)[0]