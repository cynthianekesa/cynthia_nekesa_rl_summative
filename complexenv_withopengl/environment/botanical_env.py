import numpy as np
import random
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any

# Plant types with mock medicinal info
PLANTS = {
    "Aloe Vera": {"medicinal": "Treats burns and skin irritations", "rarity": 0.7},
    "Echinacea": {"medicinal": "Boosts immune system", "rarity": 0.5},
    "Ginseng": {"medicinal": "Increases energy and reduces stress", "rarity": 0.3},
    "Turmeric": {"medicinal": "Anti-inflammatory properties", "rarity": 0.6},
    "Mint": {"medicinal": "Aids digestion and relieves nausea", "rarity": 0.8}
}

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    IDENTIFY = 4

class Plant:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.name = random.choice(list(PLANTS.keys()))
        self.identified = False
        self.rarity = PLANTS[self.name]["rarity"]
        
    def get_info(self) -> Dict[str, Any]:
        return PLANTS[self.name]

class BotanicalExplorerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, grid_size: int = 10, num_plants: int = 15):
        super().__init__()
        
        # Game parameters
        self.grid_size = grid_size
        self.num_plants = num_plants
        
        # Observation space: grid with 3 channels (agent pos, unidentified plants, identified plants)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(3, grid_size, grid_size), 
            dtype=np.float32
        )
        
        # Action space: 5 possible actions (up, down, left, right, identify)
        self.action_space = spaces.Discrete(5)
        
        # Reward range
        self.reward_range = (-10, 15)
        
        # Game state
        self.agent_pos = None
        self.plants = None
        self.score = 0
        self.identified_count = 0
        self.current_plant_info = None
        self.steps = 0
        self.max_steps = 100
        
    def _get_obs(self) -> np.ndarray:
        """Convert the game state to an observation array"""
        obs = np.zeros((3, self.grid_size, self.grid_size))
        
        # Channel 0: Agent position
        obs[0, self.agent_pos[1], self.agent_pos[0]] = 1
        
        # Channel 1: Unidentified plants
        for plant in self.plants:
            if not plant.identified:
                obs[1, plant.y, plant.x] = 1
        
        # Channel 2: Identified plants
        for plant in self.plants:
            if plant.identified:
                obs[2, plant.y, plant.x] = 1
                
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information about the environment"""
        return {
            'score': self.score,
            'identified_count': self.identified_count,
            'agent_pos': self.agent_pos,
            'steps': self.steps,
            'plants': [(p.x, p.y, p.name, p.identified) for p in self.plants],
            'current_plant_info': self.current_plant_info.get_info() if self.current_plant_info else None
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> tuple:
        super().reset(seed=seed)
        
        # Reset game state
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.plants = []
        self.score = 0
        self.identified_count = 0
        self.steps = 0
        self.current_plant_info = None
        
        # Generate plants
        positions = set()
        while len(positions) < self.num_plants:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in positions and (x, y) != tuple(self.agent_pos):
                positions.add((x, y))
                self.plants.append(Plant(x, y))
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> tuple:
        terminated = False
        truncated = False
        reward = 0
        
        # Execute action
        if action == Action.UP.value and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == Action.DOWN.value and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == Action.LEFT.value and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == Action.RIGHT.value and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == Action.IDENTIFY.value:
            reward = self._identify_plant()
        
        # Increment step count
        self.steps += 1
        
        # Check termination conditions
        if self.identified_count == self.num_plants:
            terminated = True
            reward += 10  # Bonus for identifying all plants
        elif self.steps >= self.max_steps:
            truncated = True
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _identify_plant(self) -> float:
        """Identify plant at current position and return reward"""
        for plant in self.plants:
            if plant.x == self.agent_pos[0] and plant.y == self.agent_pos[1] and not plant.identified:
                plant.identified = True
                self.identified_count += 1
                self.current_plant_info = plant
                
                # Calculate reward
                reward = 10  # Base reward for correct identification
                if plant.rarity < 0.5:
                    reward += 5  # Bonus for rare plants
                self.score += reward
                return reward
        
        # Penalty for trying to identify where there's no plant or already identified
        return -2

# Register the environment
gym.register(
    id='BotanicalExplorer-v0',
    entry_point='botanical_env:BotanicalExplorerEnv',
    max_episode_steps=100,
)