# -*- coding: utf-8 -*-
"""
По сути это уже не базовый агент, потому что обучается он с постепенным увеличением длины уровня
"""

from cnq_agent import CNQAgent
from retro_contest.local import make
import numpy as np
import cv2

EPISODES = 2500

def train():
    # initialize gym environment and the agent
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print('state_shape: ',  env.observation_space.shape)
    print('action_size: ',  action_size)
    # exit()
    agent = CNQAgent(action_size, 'base_visual_sonic_agent')
    agent.load_model()
    batch_size = 50
    total_time = 0
    total_r = 0
    done_counter = 0
    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()
        # print(state.shape)
        # exit()
        state = np.reshape(state, (1, 224, 320, 3))

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
            action_space = env.action_space.sample()
            action_space[action] = 1
            
            next_state, reward, done, _ = env.step(action_space)
            total_r += reward
            next_state = np.reshape(next_state, (1, 224, 320, 3))
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole

            if done or total_time > 2000 * (1.0 + 0.01 * e):
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, total_r))
                total_time = 0
                total_r = 0
                done_counter += 1
                if done_counter % 20 == 0:
                    agent.save_model()
                break
        # train the agent with the experience of the episode
        total_time+=500
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    agent.save_model()

def test(agent):
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    state = env.reset()
    while True:
        action = agent.act(state)
        action_space = env.action_space.sample()
        action_space[action] = 1
        
        state, reward, done, _ = env.step(action_space)
        cv2.imshow('game', state)
        cv2.waitKey(20)
        # env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    # train()
    agent = CNQAgent(12, 'base_visual_sonic_agent')
    agent.load_model()
    test(agent)
