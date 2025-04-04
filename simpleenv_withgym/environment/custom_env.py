import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BotanicalExplorerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Box(low=0, high=1, shape=(size, size, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 0:Up, 1:Right, 2:Down, 3:Left

        self.agent_pos = None
        self.target_pos = None
        self.obstacles = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.target_pos = [self.size - 1, self.size - 1]
        self.obstacles = [[1, 1], [2, 2], [3, 1]]
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        delta = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}[action]
        next_pos = [self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]]

        if (0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size and next_pos not in self.obstacles):
            self.agent_pos = next_pos

        done = self.agent_pos == self.target_pos
        reward = 10 if done else -0.1
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        grid = np.zeros((self.size, self.size, 3), dtype=np.float32)
        for pos in self.obstacles:
            grid[pos[0], pos[1]] = [1, 0, 0]
        grid[self.target_pos[0], self.target_pos[1]] = [0, 1, 0]
        grid[self.agent_pos[0], self.agent_pos[1]] = [0, 0, 1]
        return grid

    def render(self):
        if self.render_mode == "human":
            print(np.where(self._get_obs()[:, :, 2] == 1, 'A', ' ').tolist())

    def close(self):
        pass
