import gym
from agent import Agent
from train import train_agent
import numpy as np

def print_q_table(q_table):
    for row in q_table:
        print(' '.join(f'{x:.2f}' for x in row))

def main():
    # Create the environment
    env = gym.make('FrozenLake-v1', is_slippery=False)

    # Create the agent
    agent = Agent(env.observation_space.n, env.action_space.n)

    # Train the agent
    episodes = 10000
    train_agent(env, agent, episodes)

    # Print the Q-table
    print("\nFinal Q-table:")
    print_q_table(agent.q_table)

    # Test the agent
    state, _ = env.reset()
    done = False
    total_reward = 0

    print("\nTesting the agent:")
    while not done:
        action = agent.choose_action(state, epsilon=0)
        print(f"State: {state}, Action: {action}")
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()