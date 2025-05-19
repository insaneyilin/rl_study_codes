import argparse
import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import random
import time

import numpy as np
import pygame
from backgroud import Cloud, Desert, Score
from events import ADD_ENEMY
from gameover import GameOver
from PIL import Image
from speed import Speed, SpeedRatio
from sprites.collision import detect_collision_by_alpha_channel
from sprites.enemies import Cactus, Pterodactyl
from sprites.player import TRex, TRexStatus

_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_PATHS = {
    "desert":
    os.path.join(_CURRENT_FILE_DIR, "./images/desert.png"),
    "cloud":
    os.path.join(_CURRENT_FILE_DIR, "./images/cloud.png"),
    "cactuses": [
        os.path.join(_CURRENT_FILE_DIR, "./images/cactus/cactus_1.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/cactus/cactus_2.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/cactus/cactus_3.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/cactus/cactus_4.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/cactus/cactus_5.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/cactus/cactus_6.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/cactus/cactus_7.png"),
    ],
    "pterodactyl": [
        os.path.join(_CURRENT_FILE_DIR,
                     "./images/pterodactyl/pterodactyl_1.png"),
        os.path.join(_CURRENT_FILE_DIR,
                     "./images/pterodactyl/pterodactyl_2.png"),
    ],
    "dinosaur": [
        os.path.join(_CURRENT_FILE_DIR, "./images/t-rex/standing_1.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/t-rex/standing_2.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/t-rex/creeping_1.png"),
        os.path.join(_CURRENT_FILE_DIR, "./images/t-rex/creeping_2.png"),
    ],
    "gameover":
    os.path.join(_CURRENT_FILE_DIR, "./images/gameover.png"),
    "restart":
    os.path.join(_CURRENT_FILE_DIR, "./images/restart.png"),
    "font":
    os.path.join(_CURRENT_FILE_DIR, "./fonts/DinkieBitmap-7pxDemo.ttf"),
}


class TRexRunner:

    def __init__(self, num_history_frames=4):
        self.assets_paths = _ASSETS_PATHS
        pygame.init()
        pygame.display.set_caption("T-Rex Runner Pygame")
        self.screen_size = (1000, 350)

        self.screen = pygame.display.set_mode(self.screen_size)
        self.screen.fill((255, 255, 255))

        self.speed_ratio = SpeedRatio()
        self.background_speed = Speed(8.0, self.speed_ratio)
        self.cloud_speed = Speed(1.0, self.speed_ratio)

        self.desert_image = pygame.image.load(self.assets_paths["desert"])
        self.cloud_image = pygame.image.load(self.assets_paths["cloud"])
        self.cactus_images = [
            pygame.image.load(img_path)
            for img_path in self.assets_paths["cactuses"]
        ]
        self.pterodactyl_images = [
            pygame.image.load(img_path)
            for img_path in self.assets_paths["pterodactyl"]
        ]
        self.dinosaur_images = [
            pygame.image.load(img_path)
            for img_path in self.assets_paths["dinosaur"]
        ]
        self.gameover_image = pygame.image.load(self.assets_paths["gameover"])
        self.restart_image = pygame.image.load(self.assets_paths["restart"])

        self.desert = Desert(self.desert_image, speed=self.background_speed)
        self.clouds = [
            Cloud(self.cloud_image, 10, 50, speed=self.cloud_speed),
            Cloud(self.cloud_image, 500, 70, speed=self.cloud_speed),
        ]
        self.t_rex = TRex(self.dinosaur_images)
        self.enemies = pygame.sprite.Group()
        self.enemies.add(
            Cactus(self.cactus_images[0], speed=self.background_speed))
        self.score = Score(font=pygame.font.Font(self.assets_paths["font"],
                                                 30),
                           speed=self.background_speed)
        self.gameover = GameOver(self.gameover_image, self.restart_image)

        self.num_history_frames = num_history_frames
        self.history_frames = []
        self.record_frames = []
        self.over = False

        self.fps = 60
        self.clock = pygame.time.Clock()

    def begin(self):
        self.history_frames = []
        return self.get_history_frames(0)

    def step(self, action, record=False, record_path=None):
        # The state is the history frames, we need temporal information.
        state = self.get_history_frames(action, record, record_path)
        reward = 0.0
        if self.gameover.is_gameover:
            reward = -100.0
        elif self.t_rex.status == TRexStatus.RUNNING:
            reward = 0.05
        elif self.t_rex.status == TRexStatus.JUMPING:
            reward = 0.005
        elif self.t_rex.status == TRexStatus.CREEPING:
            reward = 0.005

        done = self.gameover.is_gameover
        return state, reward, done

    def get_history_frames(self, action, record=False, record_path=None):
        frame = self._step(action)
        if record:
            self.record_frames.append(frame)

        # Resize current frame to 500x175 (half the original size) and convert to binary.
        frame = frame.resize((500, 175))
        frame = (np.array(frame) < 128)

        # If the history frames are empty, fill it with the current frame.
        if len(self.history_frames) == 0:
            self.history_frames = [frame] * self.num_history_frames
        else:
            # Keep the last num_history_frames frames.
            self.history_frames.pop(0)
            self.history_frames.append(frame)

        if record and self.gameover.is_gameover:
            # Save the recorded frames to a GIF file.
            self.record_frames[0].save(record_path,
                                       save_all=True,
                                       append_images=self.record_frames[1:],
                                       duration=20,
                                       loop=0)
            self.record_frames = []

        return np.array(self.history_frames)  # (4, 175, 500) shape

    def _step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.over = True
            elif event.type == pygame.QUIT:
                self.over = True
            elif event.type == ADD_ENEMY:
                # if score.value > 100 and random.random() < 0.2:
                if random.random() < 0.5:
                    self.enemies.add(
                        Pterodactyl(self.pterodactyl_images,
                                    speed=self.background_speed))
                else:
                    cactus_image = random.choice(self.cactus_images)
                    self.enemies.add(
                        Cactus(cactus_image, speed=self.background_speed))

        if self.gameover.is_gameover:
            self.gameover.is_gameover = False

            for enemy in self.enemies:
                enemy.kill()
            self.enemies.add(
                Cactus(self.cactus_images[0], speed=self.background_speed))
            self.score.clear()
            self.speed_ratio.reset()
        else:
            self.speed_ratio.update()

            self.desert.update(self.screen)
            for enemy in self.enemies:
                enemy.update(self.screen)
            for cloud in self.clouds:
                cloud.update(self.screen)
            self.t_rex.update(self.screen)
            self.score.update(self.screen)

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_SPACE] or pressed[pygame.K_UP]:
                action = 1
            elif pressed[pygame.K_DOWN]:
                action = 2
            self.t_rex.handle_action(action)

            for enemy in self.enemies:
                if detect_collision_by_alpha_channel(self.t_rex,
                                                     enemy,
                                                     self.screen,
                                                     plot_mask=False):
                    self.gameover.is_gameover = True
                    break

        # Update the screen. (Swap the buffers to display the new frame)
        pygame.display.flip()

        # Convert the screen to a grayscale image.
        state = pygame.image.tobytes(self.screen, 'RGB')
        state = Image.frombytes('RGB', self.screen.get_size(),
                                state).convert('L')
        return state

    def play(self):
        while not self.over:
            self._step(0)  # 0: no action
            self.clock.tick(self.fps)

    def close(self):
        pygame.quit()

    def get_random_action(self):
        # 0: no action, 1: jump, 2: creep
        return random.randint(0, 2)

    def observation_shape(self):
        # The observation shape is (batch_size, num_history_frames, 175, 500)
        return (1, self.num_history_frames, 175, 500)

    def num_actions(self):
        # The number of actions is 3: no action, jump, creep
        return 3


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="T-Rex Runner Pygame")
    argparser.add_argument("--fps", type=int, default=60)
    argparser.add_argument(
        "--no_render",
        action="store_true",
        default=False,
        help="Do not render the game, only run the game loop.")
    args = argparser.parse_args()
    if args.no_render:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = TRexRunner()
    state = env.begin()
    while True:
        # action = random.randint(0, 2)  # Test random actions
        action = 0
        state, reward, done = env.step(action)
        print(reward, done)

        if env.over:
            env.close()
            break
