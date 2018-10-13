# -*- coding: utf-8 -*-
"""
Обучение происходит по картинке с экрана
За один шаг может быть выбрано несколько действий
Длина уровня постепенно растет (линейно)
Если агент не получает награду за N шагов, то исследовательский коэфф. увеличивается.
"""

from agent_7 import CNQAgent
import numpy as np
import gym_remote.exceptions as gre
import gym_remote.client as grc
from collections import deque
import cv2

class Trainer:
    def __init__(self):
        self.TRY_COUNT = 500
        self.TARGET_REWARD = 6000
        # self.MAX_TIME = 18000
        self.MAX_TIME = 40000
        self.TIME_PER_TRY = 2000
        # self.AGENT_NAME = 'agent-7_best-sol_update-mean-reward'
        self.AGENT_NAME = 'agent-7_best-sol_update-mean-reward_7'
        self.REPLAYS_DIR = 'D:\\PROJECTS\\GENERAL\\RESEARCH & CONTEST\\Retro GYM\\replays\\'
        self.INERCIA = 1
        self.done = False
        self.max_lives = 3
        self.reward = 0
        self.Q_STEP = 4
        self.total_source_reward = 0
        self.rings = 0
        self.pos_x = 0
        self.acions_memory = []
        self.pos_y = 0

    def train_init(self):
        from retro_contest.local import make
        self.env = make(game='SonicTheHedgehog-Genesis',
                        state='SpringYardZone.Act2')
        self.screen_shape = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = CNQAgent(self.action_size, self.AGENT_NAME)
        self.agent.load_model()
        self.agent.epsilon_decay = 0.1
        self.prev_epsilon = self.agent.epsilon
        self.research_mode = False
        self.research_step = 0
        # print('state_shape: ',  env.observation_space.shape)
        # print('action_size: ',  action_size)
        self.memory = deque(maxlen=6000)
        self.train = True

    def save_experience(self):
        from data_writer import VideoWriter
        recorder = VideoWriter(320, 224, 20, self.REPLAYS_DIR + str(self.AGENT_NAME) + '.avi')
        for state, action, reward, next_state, done, info in self.memory:
            frame = cv2.putText(state, str(reward), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            recorder.add_frame(frame)
        print('replay saved...')
        recorder.stop_and_release()

    def view_frame(self):
        state = cv2.putText(self.state, str(self.action), (150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        state = cv2.putText(state, str(round(self.reward, 1)), (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        state = cv2.putText(state, str(self.info['x']), (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow('game', state)
        cv2.waitKey(20)


    def view_experience(self):
        for state, action, reward, next_state, done, info in self.memory:
            # state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            # info_screen = np.zeros(state.shape)
            # info_screen = cv2.putText(info_screen, str(round(reward, 1)), (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            state = cv2.putText(state, str(action), (150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            state = cv2.putText(state, str(round(reward, 1)), (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            state = cv2.putText(state, str(self.info['x']), (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            # frame = np.hstack((state, info_screen))
            frame = state
            # print(state)
            cv2.imshow('game', frame)
            cv2.waitKey(30)

    def update_epsilon_by_reward(self, reward, expectation=4.0):
        if abs(reward) < expectation:
            self.prev_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.9
        else :
            self.agent.epsilon = self.prev_epsilon

    def update_epsilon_by_mean_reward(self, reward, expectation=4.0):
        self.local_reward.append(reward)
        if abs(np.array(self.local_reward).mean()) < expectation:
            self.agent.epsilon = 0.9
        else :
            self.agent.epsilon = 0.01

    def update_epsilon_by_mean_reward_2(self, reward, expectation=4.0):
        self.local_reward.append(reward)
        self.research_step+=1

        if self.research_mode and self.research_step < 80:
            self.reward -= 10
            return

        self.research_mode = False
        self.research_step = 0
        self.agent.epsilon = self.prev_epsilon

        # if abs(np.array(self.local_reward).mean()) < expectation:
        if np.array(self.local_reward).mean() < expectation:
            self.prev_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.85
            self.research_mode = True
            self.reward -= 50

    def update_reward(self, source_reward):
        if self.info['x'] > self.pos_x:
            self.pos_x = self.info['x']

        delta_x = self.info['x'] - self.pos_x
        if delta_x > -250.0 and delta_x <= 0.0:
            source_reward = 0.0

        # self.reward = source_reward + 0.01 * self.info['x']
        # self.reward = source_reward + 0.1 * self.info['score'] + self.info['rings']
        self.reward = source_reward
        delta_rings = self.info['rings'] - self.rings
        if delta_rings < 0:
            self.reward += 20*delta_rings          
        self.rings = self.info['rings']

        if self.info['lives'] < self.max_lives:
            self.max_lives = self.info['lives']
            self.reward=-500
        # self.reward = source_reward
        # self.reward -= 1
        if self.info['y'] < self.pos_y:
            delta_y = self.pos_y - self.info['y']
            self.reward+=delta_y
            self.pos_y = self.info['y']


    def press_buttons(self, button_indexes, repeat=10):
        if self.done:
            return

        buttons = np.zeros(12)
        for ind in button_indexes:
            buttons[ind] = 1

        for i in range(repeat):
            self.next_state, source_reward, self.done, self.info = self.env.step(buttons)
            # print(self.info)
            self.total_source_reward += source_reward
            self.update_reward(source_reward)
            if self.train:
                self.memory.append((self.state, self.action, self.reward, self.next_state, self.done, self.info))
            self.state = self.next_state
            # source_reward += 0.01 * info['x']
            self.view_frame()
            if self.done:
                break


    def do(self):
        act_value = np.argmax(self.action)
        if act_value == 0:
            self.press_buttons([], self.INERCIA)
        if act_value == 1:
            self.press_buttons([7], 1)
        if act_value == 2:
            self.press_buttons([6], 1)
        if act_value == 3:
            self.press_buttons([5], 1)
        if act_value == 4:
            self.press_buttons([4, 0], self.INERCIA)
        if act_value == 5:
            self.press_buttons([4, 0, 7], self.INERCIA)
        if act_value == 6:
            self.press_buttons([4, 0, 6], self.INERCIA)
        # if act_value == 6:
        #     self.press_buttons(6, 7)
        #     self.press_buttons(7, 2)
        #     self.press_buttons(4, 5)
        #     self.press_buttons(0, 5)
        #     self.press_buttons(7, 5)

    def train(self):
        self.total_source_reward = 0
        current_try = 0
        max_reward = 0
        self.action = np.zeros(7)
        self.reward = 0
        pos_x = 0
        time_ratio = int((self.MAX_TIME - self.TIME_PER_TRY)/self.TRY_COUNT)
        self.train_init()

        while self.total_source_reward < self.TARGET_REWARD and current_try < self.TRY_COUNT:
            self.total_source_reward = 0
            self.max_lives = 3
            current_try += 1
            self.local_reward = deque(maxlen=50)
            self.state = self.env.reset()
            self.done = False
            self.memory.clear()

            for time in range(0, self.TIME_PER_TRY):
                self.action = self.agent.act(self.state, self.action, self.reward, pos_x)
                self.reward = 0
                # for i in range(self.INERCIA):
                    # next_state, source_reward, done, info = self.env.step(action)
                self.do()
                if self.done:
                    break

                # total_source_reward += self.reward
                # self.memory.append((self.state, action, self.reward, self.next_state, self.done, self.info))
                # self.state = self.next_state

                # self.update_epsilon_by_reward(reward)
                # self.update_epsilon_by_mean_reward(reward)
                self.update_epsilon_by_mean_reward_2(self.reward, expectation=0.5)

            self.TIME_PER_TRY += time_ratio
            print("try: {}, score: {}".format(current_try, round(self.total_source_reward, 1)))

            if self.total_source_reward > 0.2 * max_reward:
                max_reward = self.total_source_reward
                print('achieved best experience...')
                # self.view_experience()
                for state, action, reward, next_state, done, info in self.memory:
                    self.agent.remember(state, action, reward, next_state, done, info)
                print('training on batch {} ...'.format(len(self.memory)))
                self.agent.replay()
                # self.save_experience()
                self.agent.save_model()

        self.agent.save_model()
        self.save_experience()
        cv2.destroyWindow('game')

    def valid(self):
        from retro_contest.local import make
        agent = CNQAgent(12, self.AGENT_NAME)
        agent.load_model()
        agent.epsilon = 0.1
        self.env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
        self.state = self.env.reset()
        self.reward = 0
        self.action = np.zeros(7)
        self.train = False
        while True:
            self.action = agent.act(self.state, self.action, self.reward, 0)   
            # print(action)  
            self.do()
            self.state = self.next_state
            if self.done:
                obs = self.env.reset()
            frame = cv2.putText(self.state, str(round(self.reward,1)), (150, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            frame = cv2.putText(frame, str(self.action), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.imshow('AGENT VALIDATION', frame)
            cv2.waitKey(60)

if __name__ == '__main__':
    Trainer().train()
    # Trainer().valid()
