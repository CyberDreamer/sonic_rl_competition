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
# from ar_agent import CNQAgent
# from agent_5 import CNQAgent
from agent_6 import CNQAgent
import numpy as np
import gym_remote.exceptions as gre
import gym_remote.client as grc
from collections import deque


EPISODES = 240
# MAX_FRAMES = 18000
MAX_FRAMES = 10000
AGENT_NAME = 'agent_6_II_repeat-1_exp-5-30-10_total-r'
REPLAYS_DIR = 'D:\\PROJECTS\\GENERAL\\RESEARCH & CONTEST\\Retro GYM\\replays\\'
# AGENT_NAME = 'base_visual_sonic_agent'
EXPERIMENT_SIZE = 5

def train():
    from retro_contest.local import make
    import cv2
    from data_writer import VideoWriter
    ONE_TRY = 600
    RATIO = (MAX_FRAMES - ONE_TRY)/EPISODES

    # initialize gym environment and the agent
    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print('state_shape: ',  env.observation_space.shape)
    print('action_size: ',  action_size)
    # exit()
    agent = CNQAgent(action_size, AGENT_NAME)
    agent.epsilon_decay = 0.95
    # agent.load_model()
    batch_size = 400
    total_time = 0
    reward = 0
    action = np.zeros(12)
    max_reward = 0

    prev_epsilon = agent.epsilon
    done_counter = 0
    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()
        local_speed = deque(maxlen=20)
        experiment_steps = EXPERIMENT_SIZE
        experiment_mode = False
        # print(state.shape)
        # exit()
        state = np.reshape(state, (1, 224, 320, 3))
        total_r = 0
        max_lives = 3
        local_r = 0
        pos_x = 0
        x1 = 0
        x2 = 0
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(int(ONE_TRY)):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state, action, local_r, pos_x)
            # print(action)
            local_r = 0
            for j in range(4):
                next_state, clear_r, done, info = env.step(action)

                x2 = info['x']
                speed = x2 - x1
                x1 = x2

                total_r += clear_r
                if speed > 0.0:
                    # reward = clear_r + clear_r * 0.01 * info['x']
                    reward = 5*clear_r
                else:
                    reward = 5*clear_r
                pos_x = info['x']
                if info['lives'] < max_lives:
                    max_lives = info['lives']
                    reward=-100

                # expectation = 2
                # if abs(np.array(local_speed).mean()) < expectation:
                #     reward=-5.0

                local_r+=reward
                local_speed.append(speed)      


            next_state = np.reshape(next_state, (1, 224, 320, 3))
            # Remember the previous state, action, reward, and done
            if local_r>2.0:
                agent.remember(state, action, local_r, next_state, done, info)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole

            if done:
                break
            
            # if len(local_speed) == 20 and np.array(local_speed).mean() < 2.0:
            #     prev_epsilon = agent.epsilon
            #     agent.epsilon = 0.8
            #     experiment_mode = True

            # if experiment_mode:
            #     experiment_steps-=1
            #     if experiment_steps <= 0:
            #         experiment_mode = False
            #         experiment_steps = EXPERIMENT_SIZE
            #         agent.epsilon = prev_epsilon

        ONE_TRY +=RATIO
        # print('ONE TRY: {}'.format(ONE_TRY))
        # train the agent with the experience of the episode
        # print the score and break out of the loop
        print("episode: {}/{}, score: {}".format(e, EPISODES, round(total_r, 1)))
        done_counter += 1
        if done_counter % 40 == 0:
            recorder = VideoWriter(320, 224, 30, REPLAYS_DIR + str(done_counter) + '.avi')
            for state, action, reward, next_state, done, info in agent.memory: 
                # cv2.imshow('game', state[0])
                # cv2.waitKey(20)
                frame = cv2.putText(state[0], str(reward), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                recorder.add_frame(frame)
            print('model saved...')
            recorder.stop_and_release()
            agent.save_model()
        if total_r > max_reward:
            # max_reward = total_r
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

        # batch_size+=10

    agent.save_model()

def valid():
    from retro_contest.local import make
    import cv2
    agent = CNQAgent(12, AGENT_NAME)
    agent.load_model()
    agent.epsilon = 0.1
    env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    state = env.reset()
    reward = 0
    action = np.zeros(12)
    while True:
        state = np.reshape(state, (1, 224, 320, 3))
        action = agent.act(state, action, reward)   
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
    env = grc.RemoteEnv('tmp/sock')
    # env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
    print('starting episode')
    state = env.reset()
    reward = 0
    action = np.zeros(12)
    while True:
        state = np.reshape(state, (1, 224, 320, 3))
        action = agent.act(state, action, reward)
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