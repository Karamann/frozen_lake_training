import numpy as np


def train_agent(env, agent, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, alpha, gamma)
            state = next_state
            total_reward += reward

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    print("Training completed.")
