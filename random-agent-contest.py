from retro_contest.local import make
import cv2
import numpy

class Sonic:
    def do(self, act, repeat=10):
        action = numpy.zeros(12)
        action[act] = True
        for i in range(repeat):
            state, reward, done, info = self.env.step(action)
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

        state, rew, done, info = self.do(7, 20)
        state, rew, done, info = self.do(4, 5)
        state, rew, done, info = self.do(0, 5)
        state, rew, done, info = self.do(6, 20)
        x1 = 0
        x2 = 0
        x2 = info['x']
        speed = x2 - x1
        x1 = x2
        print('speed: ', speed)
        if done:
            obs = self.env.reset()

        for i in range(500):
            action = numpy.zeros(12)
            state, rew, done, info = self.env.step(action)
            self.draw(state, self.total_reward)
            if done:
                obs = self.env.reset()


if __name__ == '__main__':
    Sonic().main()
