# -*- coding: utf-8 -*-
"""
Обучение происходит по картинке с экрана
За один шаг может быть выбрано несколько действий
Длина уровня постепенно растет (линейно)
Если агент не получает награду за N шагов, то исследовательский коэфф. увеличивается.
"""

# from cnq_agent import CNQAgent
# from cnq_agent_ma import CNQAgent
# from vgg19_agent import CNQAgent
from ar_agent import CNQAgent
from retro_contest.local import make
import numpy as np
import cv2
from data_writer import VideoWriter
import gym_remote.exceptions as gre
import gym_remote.client as grc

EPISODES = 2500
MAX_FRAMES = 18000
AGENT_NAME = 'agent_3_sa'
REPLAYS_DIR = 'D:\\PROJECTS\\GENERAL\\RESEARCH & CONTEST\\Retro GYM\\replays\\'
# AGENT_NAME = 'base_visual_sonic_agent'
EXPERIMENT_SIZE = 0

def train():
    ONE_TRY = 100
    RATIO = (MAX_FRAMES - ONE_TRY)/EPISODES

    # initialize gym environment and the agent
    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print('state_shape: ',  env.observation_space.shape)
    print('action_size: ',  action_size)
    # exit()
    agent = CNQAgent(action_size, AGENT_NAME)
    agent.epsilon_decay = 0.995
    # agent.load_model()
    batch_size = 20
    total_time = 0
    total_r = 0
    reward = 0
    action = np.zeros(12)

    prev_epsilon = agent.epsilon
    done_counter = 0
    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()
        local_r = []
        experiment_steps = EXPERIMENT_SIZE
        experiment_mode = False
        # print(state.shape)
        # exit()
        state = np.reshape(state, (1, 224, 320, 3))

        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(int(ONE_TRY)):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state, action, reward)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            total_r += reward
            next_state = np.reshape(next_state, (1, 224, 320, 3))
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole

            if done:
                break
            
            # local_r.append(reward)
            # if len(local_r) >= 20 and sum(local_r) < 10.0:
            #     prev_epsilon = agent.epsilon
            #     agent.epsilon = 0.9
            #     local_r = []
            #     # print('IN experiment_mode {} -> {}'.format(prev_epsilon, 0.9))
            #     experiment_mode = True

            # if experiment_mode:
            #     experiment_steps-=1
            #     if experiment_steps <= 0:
            #         experiment_mode = False
            #         # print('OUT experiment_mode {} -> {}'.format(0.9, prev_epsilon))
            #         experiment_steps = EXPERIMENT_SIZE
            #         agent.epsilon = prev_epsilon

        ONE_TRY +=RATIO
        # print('ONE TRY: {}'.format(ONE_TRY))
        # train the agent with the experience of the episode
        # print the score and break out of the loop
        print("episode: {}/{}, score: {}".format(e, EPISODES, round(total_r, 1)))
        total_r = 0
        done_counter += 1
        if done_counter % 40 == 0:
            recorder = VideoWriter(320, 224, 50, REPLAYS_DIR + str(done_counter) + '.avi')
            for state, action, reward, next_state, done in agent.memory: 
                # cv2.imshow('game', state[0])
                # cv2.waitKey(20)
                recorder.add_frame(state[0])
            print('model saved...')
            recorder.stop_and_release()
            agent.save_model()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    agent.save_model()

def valid():
    agent = CNQAgent(12, AGENT_NAME)
    agent.load_model()
    agent.epsilon = 0.1
    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    state = env.reset()
    while True:
        state = np.reshape(state, (1, 224, 320, 3))
        action = agent.act(state)   
        # print(action)  
        state, reward, done, _ = env.step(action)
        cv2.imshow('game', state)
        cv2.waitKey(20)
        if done:
            obs = env.reset()

def test():
    agent = CNQAgent(12, AGENT_NAME)
    agent.load_model()
    agent.epsilon = 0.1
    print('connecting to remote environment')
    # env = grc.RemoteEnv('tmp/sock')
    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    print('starting episode')
    state = env.reset()
    while True:
        state = np.reshape(state, (1, 224, 320, 3))
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        if done:
            print('episode complete')
            env.reset()


if __name__ == "__main__":
    train()
    # valid()

    # try:
    #     test()
    # except gre.GymRemoteError as e:
    #     print('exception', e)