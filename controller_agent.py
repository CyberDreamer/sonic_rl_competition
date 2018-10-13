from retro_contest.local import make
import cv2
import numpy
import sys
import msvcrt

class Sonic:
    def __init__(self):
        self.buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        self.actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], [], ['UP', 'B', 'RIGHT']]

    def press(self, act, repeat=10):
        buttons = numpy.zeros(12)
        for b in act:
            buttons[self.buttons.index(b)] = 1

        for i in range(repeat):
            state, reward, done, info = self.env.step(buttons)
            # print(info)
            self.total_reward += reward
            self.draw(state, self.total_reward)
        return state, reward, done, info 
        

    def draw(self, state, reward='None'):
        state = cv2.putText(state, str(reward), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        cv2.imshow('game em', state)
        cv2.waitKey(40)

    def main(self):
        self.env = make(game='SonicTheHedgehog-Genesis', state='SpringYardZone.Act3')
        obs = self.env.reset()
        self.total_reward = 0
        done = False
        inercia = 5
        while True:
            key=msvcrt.getch()
            # print(k)
            if key==b'a':
                state, rew, done, info = self.press(self.actions[0], inercia)
            if key==b'd':
                state, rew, done, info = self.press(self.actions[1], inercia)
            if key==b'w':
                state, rew, done, info = self.press(self.actions[8], inercia)
            if key==b's':
                state, rew, done, info = self.press(self.actions[3], inercia)
            if key==b'e':
                state, rew, done, info = self.press(self.actions[4], inercia)
            if key==b'b':
                state, rew, done, info = self.press(self.actions[5], inercia)
            if key==b'\x00':
                state, rew, done, info = self.press(self.actions[7], inercia)
            # x1 = 0
            # x2 = 0
            # x2 = info['x']
            # speed = x2 - x1
            # x1 = x2
            # print('speed: ', speed)
            if done:
                obs = self.env.reset()

            # if k==ord('Esc'):
            #     break


if __name__ == '__main__':
    Sonic().main()
