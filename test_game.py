import retro
import cv2
import time

def main():
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
        # cv2.waitKey(1000)
        time.sleep(0.1)

if __name__ == '__main__':
    main()