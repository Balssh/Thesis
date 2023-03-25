from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

# from selenium.webdriver.common.action_chains import ActionChains
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
    def __init__(self, screen_width: int = 120, screen_height: int = 120):
        # super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._web_driver = webdriver.Chrome()

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, self.screen_width, self.screen_height),
            dtype="uint8",
        )
        # self.current_key = None
        self.actions_map = [Keys.RIGHT, Keys.SPACE, Keys.DOWN]
        # self.actions_chain = ActionChains(self._web_driver)
        self.state_queue = deque(maxlen=4)

    def _get_image(self):
        LEADING_TEXT = "data:image/png;base64,"
        _img = self._web_driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL()"
        )
        _img = _img[len(LEADING_TEXT) :]
        return np.array(Image.open(BytesIO(base64.b64decode(_img))))

    def _get_observation(self):
        image = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2GRAY)
        image = image[:500, :400]
        image = cv2.resize(image, (self.screen_width, self.screen_height))
        image = np.reshape(image, (1, self.screen_width, self.screen_height))
        # print(image.shape)
        # self.state_queue.append(image)
        # if len(self.state_queue) < 4:
        #     return np.stack([image] * 4, axis=-1)
        # else:
        #     return np.stack(self.state_queue, axis=-1)
        return image

    def _get_score(self):
        return int(
            "".join(
                self._driver.execute_script(
                    "return Runner.instance_.distanceMeter.digits"
                )
            )
        )

    def _get_done(self):
        return self._web_driver.execute_script("return Runner.instance_.crashed")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._web_driver.get("https://chromedino.com/")
        WebDriverWait(self._web_driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas"))
        )

        self._web_driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.SPACE)
        return self._get_observation(), {}

    def step(self, action):
        self._web_driver.find_element(By.CSS_SELECTOR, "body").send_keys(
            self.actions_map[action]
        )
        observation = self._get_observation()
        done = self._get_done()
        reward = 1 if not done else -1

        time.sleep(0.02)
        return observation, reward, done, {"score": self._get_score()}

    # def render(self, mode: str = "human"):
    #     img = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2RGB)
    #     if mode == "rgb_array":
    #         return img
    #     elif mode == "human":
    #         from gym.Env import rendering

    #         if self.viewer is None:
    #             self.viewer = rendering.SimpleImageViewer()
    #         self.viewer.imshow(img)
    #         return self.viewer.isopen

    # def close(self):
    #     if self.viewer is not None:
    #         self.viewer.close()
    #         self.viewer = None


# dino = Dino()
# print(dino.reset())
# plt.show()
