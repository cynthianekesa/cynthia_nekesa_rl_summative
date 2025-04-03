import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.botanical_env import BotanicalExplorerEnv  # My custom environment

# Configuration
config = {
    "total_timesteps": 100_000,
    "learning_rate": 1e-4,
    "buffer_size": 50_000,
    "batch_size": 128,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "target_update_interval": 1000,
    "train_freq": 4,
    "gradient_steps": 1,
    "policy_kwargs": {
        "net_arch": [128, 128]  # Neural network architecture
    },
    "log_dir": "./dqn_logs",
    "save_dir": "./dqn_models",

}

def create_env():
    """Create and wrap the environment"""
    env = BotanicalExplorerEnv(grid_size=10, num_plants=15)
    env = Monitor(env)  # For tracking episode stats
    return env

def setup_callbacks(env, save_dir):
    """Configure training callbacks"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Stop training when mean reward reaches threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=200, 
        verbose=1
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=stop_callback,
        eval_freq=5000,
        best_model_save_path=save_dir,
        verbose=1
    )
    
    return [eval_callback]

def train_dqn():
    # Create environment
    env = DummyVecEnv([create_env])
    
    # Initialize model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        target_update_interval=config["target_update_interval"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=config["log_dir"]
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(env, config["save_dir"])
    
    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model.save(os.path.join(config["save_dir"], "dqn_botanical_final"))
    return model

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["save_dir"], exist_ok=True)
    
    # Start training
    print("Starting DQN training...")
    trained_model = train_dqn()
    print("Training completed!")