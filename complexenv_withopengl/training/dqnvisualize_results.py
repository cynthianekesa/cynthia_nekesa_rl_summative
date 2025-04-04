import os
import sys
import numpy as np
from stable_baselines3 import DQN
from environment.botanical_env import BotanicalExplorerEnv
from environment.botanical_opengl_renderer import OpenGLRenderer

def visualize_model(model_path="dqn_models/dqn_botanical_final.zip", num_episodes=2):
    """Memory-efficient visualization with frame sampling"""
    
    # 1. Load model and setup environment
    model = DQN.load(model_path)
    env = BotanicalExplorerEnv(grid_size=10, num_plants=15)
    renderer = OpenGLRenderer(env)
    
    try:
        # 2. Configure GIF writer (lower resolution)
        output_path = "trained_agent.gif"
        fps = 5  # Reduced frames per second
        frame_skip = 3  # Only save every 3rd frame
        
        with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                frame_count = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # Only capture and save every Nth frame
                    if frame_count % frame_skip == 0:
                        renderer.render_frame()
                        frame = np.array(renderer._get_last_frame())  # Implement this in your renderer
                        writer.append_data(frame)
                    
                    frame_count += 1
                
                print(f"Completed episode {episode + 1}")
    
    finally:
        renderer.close()

# Add this to your botanical_opengl_renderer.py:
# def _get_last_frame(self):
#     """Get last rendered frame as numpy array"""
#     buffer = glReadPixels(0, 0, self.window_size, self.window_size, 
#                         GL_RGB, GL_UNSIGNED_BYTE)  # Use RGB instead of RGBA
#     return np.frombuffer(buffer, dtype=np.uint8).reshape(
#         self.window_size, self.window_size, 3)

if __name__ == "__main__":
    visualize_model()

# ALternative code

import os
import sys
from stable_baselines3 import DQN

# Import your existing renderer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.botanical_env import BotanicalExplorerEnv
from environment.botanical_opengl_renderer import OpenGLRenderer  # Your existing renderer

def visualize_trained_agent(model_path, num_episodes=3):
    """Visualizes agent using your existing OpenGL renderer"""
    model = DQN.load(model_path)
    env = BotanicalExplorerEnv(grid_size=10, num_plants=15)
    renderer = OpenGLRenderer(env)  # Using YOUR renderer
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                # 1. Get agent action
                action, _ = model.predict(obs, deterministic=True)
                
                # 2. Step environment
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 3. Render frame (using your existing renderer)
                renderer.render_frame()
            
            print(f"Rendered episode {episode + 1}")
        
        # 4. Save animation
        renderer.save_gif("trained_agent.gif")
    
    finally:
        renderer.close()

if __name__ == "__main__":
    visualize_trained_agent("dqn_models/dqn_botanical_final.zip")