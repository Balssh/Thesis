from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import gymnasium as gym
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64
from collections import deque
from matplotlib import pyplot as plt


class Dino(gym.Env):
    def __init__(self, screen_width: int = 84, screen_height: int = 84):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._skip = 4
        self._web_driver = webdriver.Chrome()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_width, self.screen_height),
            dtype=np.uint8,
        )

        self._obs_buffer = np.zeros(
            (2,) + self.observation_space.shape, dtype=self.observation_space.dtype
        )
        self.actions_map = [Keys.RIGHT, Keys.UP, Keys.DOWN]
        self._web_driver.get("https://chromedino.com/")
        WebDriverWait(self._web_driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas"))
        )

    def _get_image(self):
        LEADING_TEXT = "data:image/png;base64,"
        _img = self._web_driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL()"
        )
        _img = _img[len(LEADING_TEXT) :]
        return np.array(Image.open(BytesIO(base64.b64decode(_img))))

    def _get_observation(self):
        image = cv2.cvtColor(self._get_image(), cv2.COLOR_RGB2GRAY)
        image = image[:500, :480]
        image = cv2.resize(
            image, (self.screen_width, self.screen_height), interpolation=cv2.INTER_AREA
        )
        return image

    def _get_score(self):
        return int(
            "".join(
                self._web_driver.execute_script(
                    "return Runner.instance_.distanceMeter.digits"
                )
            )
        )

    def _get_done(self):
        return self._web_driver.execute_script("return Runner.instance_.crashed")

    def reset(self, seed=None, options=None):
        time.sleep(0.75)  # delay so that the game can reload
        super().reset(seed=seed)
        self._web_driver.find_element(By.TAG_NAME, "body").send_keys(Keys.UP)
        return self._get_observation(), {}

    def step(self, action):
        total_reward = 0

        for i in range(self._skip):
            self._web_driver.find_element(By.TAG_NAME, "body").send_keys(
                self.actions_map[action]
            )
            observation = self._get_observation()
            if i == self._skip - 2:
                self._obs_buffer[0] = observation
            if i == self._skip - 1:
                self._obs_buffer[1] = observation
            done = self._get_done()
            reward = 1 if self._get_score() > 0 else 0
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, False, {}


# dino = Dino()
# dino.reset()
# for i in range(100):
#     time.sleep(1)
#     dino._get_ready()
#     if dino._get_done():
#         dino.reset()
# print(dino._get_score())
