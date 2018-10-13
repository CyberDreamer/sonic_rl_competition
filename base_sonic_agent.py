# -*- coding: utf-8 -*-
from fnq_agent import FNQAgent
from retro_contest.local import make
import numpy as np

EPISODES = 1000

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    state_size = env.observation_space.shape[0]
    print('state_shape: ',  env.observation_space.shape)
    action_size = env.action_space.n
    print('action_size: ',  action_size)
    # exit()
    agent = FNQAgent(state_size, action_size)
    batch_size = 32

    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, time_t))
                break
        # train the agent with the experience of the episode
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)