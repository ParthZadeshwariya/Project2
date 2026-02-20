import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class DroneEnv(gym.Env):

    def __init__(self):
        super(DroneEnv, self).__init__()

        self.grid_size = 10

        # 6 movement actions
        self.action_space = spaces.Discrete(6)

        # state:
        # [dx, dy, dz, front, back, left, right, up, down]
        self.observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(9,),
            dtype=np.float32
        )

        self.max_steps = 250

    # --------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------
    def reset(self, seed=None, options=None):

        self.drone_pos = np.array([0, 0, 0], dtype=np.float32)

        # random goal
        self.goal = np.array([
            random.randint(6, 9),
            random.randint(6, 9),
            random.randint(3, 6)
        ], dtype=np.float32)

        # random obstacles
        self.obstacles = []
        for _ in range(15):
            self.obstacles.append(np.array([
                random.uniform(1, 9),
                random.uniform(1, 9),
                random.uniform(1, 5)
            ]))

        self.steps = 0

        return self._get_obs(), {}

    # --------------------------------------------------
    # OBSTACLE SENSING
    # --------------------------------------------------
    def get_obstacle_distances(self):

        directions = [
            np.array([1,0,0]),
            np.array([-1,0,0]),
            np.array([0,1,0]),
            np.array([0,-1,0]),
            np.array([0,0,1]),
            np.array([0,0,-1]),
        ]

        distances = []

        for d in directions:

            min_dist = 5

            for obs in self.obstacles:
                diff = obs - self.drone_pos
                proj = np.dot(diff, d)

                if proj > 0:
                    dist = np.linalg.norm(diff)
                    min_dist = min(min_dist, dist)

            distances.append(min_dist)

        return distances

    # --------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------
    def _get_obs(self):

        dx, dy, dz = self.goal - self.drone_pos
        obstacle_info = self.get_obstacle_distances()

        return np.array(
            [dx, dy, dz] + obstacle_info,
            dtype=np.float32
        )

    # --------------------------------------------------
    # STEP FUNCTION
    # --------------------------------------------------
    def step(self, action):

        old_distance = np.linalg.norm(self.goal - self.drone_pos)

        # movement
        if action == 0:
            self.drone_pos[1] += 1
        elif action == 1:
            self.drone_pos[1] -= 1
        elif action == 2:
            self.drone_pos[0] -= 1
        elif action == 3:
            self.drone_pos[0] += 1
        elif action == 4:
            self.drone_pos[2] += 1
        elif action == 5:
            self.drone_pos[2] -= 1

        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size-1)

        self.steps += 1
        done = False

        # collision check
        for obs in self.obstacles:
            if np.linalg.norm(self.drone_pos - obs) < 0.6:
                reward = -100
                done = True
                return self._get_obs(), reward, done, False, {}

        new_distance = np.linalg.norm(self.goal - self.drone_pos)

        reward = -0.1

        if new_distance < old_distance:
            reward += 1
        else:
            reward -= 1

        # goal reached
        if new_distance < 0.5:
            reward = 100
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}
