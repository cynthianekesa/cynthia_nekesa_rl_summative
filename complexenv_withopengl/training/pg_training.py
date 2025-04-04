import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.botanical_env import BotanicalExplorerEnv  # Your custom environment

# Configuration
config = {
    "policy": "MlpPolicy",
    "total_timesteps": 500_000,
    "n_steps": 2048,               # Steps per environment per update
    "batch_size": 64,              # Minibatch size
    "learning_rate": 3e-4,
    "ent_coef": 0.01,              # Entropy coefficient for exploration
    "clip_range": 0.2,             # PPO clipping parameter
    "n_epochs": 10,                # Number of optimization epochs per update
    "gamma": 0.99,                 # Discount factor
    "gae_lambda": 0.95,            # Factor for trade-off of bias vs variance
    "max_grad_norm": 0.5,          # Maximum gradient norm
    "vf_coef": 0.5,                # Value function coefficient
    "policy_kwargs": {
        "net_arch": dict(pi=[128, 128], vf=[128, 128])  # Network architecture
    },
    "log_dir": "./ppo_logs",
    "save_dir": "./ppo_models",
    "eval_freq": 20_000,           # Evaluate every N timesteps
    "n_eval_episodes": 5           # Episodes per evaluation
}

def make_env():
    """Create and wrap the environment"""
    env = BotanicalExplorerEnv(grid_size=10, num_plants=15)
    env = Monitor(env)  # For tracking episode statistics
    return env

def setup_callbacks(env, save_path):
    """Configure training callbacks"""
    os.makedirs(save_path, exist_ok=True)
    
    # Stop training when mean reward reaches threshold
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=250, 
        verbose=1
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=stop_callback,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        best_model_save_path=save_path,
        verbose=1
    )
    
    return eval_callback

def train_ppo():
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Initialize PPO model
    model = PPO(
        config["policy"],
        env,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        ent_coef=config["ent_coef"],
        clip_range=config["clip_range"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        max_grad_norm=config["max_grad_norm"],
        vf_coef=config["vf_coef"],
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
    model.save(os.path.join(config["save_dir"], "ppo_botanical_final"))
    return model

if __name__ == "__main__":
    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["save_dir"], exist_ok=True)
    
    print("Starting PPO training...")
    trained_model = train_ppo()
    print("Training completed!")