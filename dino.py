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
        # super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        options = Options()
        options.page_load_strategy = "normal"
        self._web_driver = webdriver.Chrome(options=options)

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_width, self.screen_height),
            dtype=np.uint8,
        )
        self.actions_map = [Keys.RIGHT, Keys.UP, Keys.DOWN]

    def _get_image(self):
        LEADING_TEXT = "data:image/png;base64,"
        _img = self._web_driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL()"
        )
        _img = _img[len(LEADING_TEXT) :]
        return np.array(Image.open(BytesIO(base64.b64decode(_img))))

    def _get_observation(self):
        image = cv2.cvtColor(self._get_image(), cv2.RGB2GRAY)
        # image = np.reshape(image, (3, image.shape[0], image.shape[1]))
        # image = image[:500, :480]
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
        super().reset(seed=seed)
        self._web_driver.get("https://chromedino.com/")
        WebDriverWait(self._web_driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas"))
        )
        time.sleep(0.02)
        self._web_driver.find_element(By.TAG_NAME, "body").send_keys(Keys.UP)
        return self._get_observation(), {}

    def step(self, action):
        self._web_driver.find_element(By.TAG_NAME, "body").send_keys(
            self.actions_map[action]
        )
        observation = self._get_observation()
        done = self._get_done()
        reward = self._get_score()
        time.sleep(0.02)
        return observation, reward, done, False, {}


# dino = Dino()
# obs = dino.reset()[0]
# time.sleep(1)
# for i in range(100):
#     time.sleep(1)
#     if dino._get_done():
#         dino.reset()
#     dino.step(2)
#     print(dino._get_score())
