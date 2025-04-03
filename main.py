from botanical_env import BotanicalExplorerEnv
from botanical_opengl_renderer import OpenGLRenderer
import numpy as np
import time

def simulate_episode(env, renderer):
    obs, info = env.reset()
    renderer.clear_frames()
    
    # Initial frame
    renderer.render_frame()
    
    for _ in range(20):  # Reduced steps for quicker demo
        action = np.random.randint(0, 5)  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        renderer.render_frame()
        
        if terminated or truncated:
            break
    
    # Final frame
    renderer.render_frame()
    return len(renderer.frames)

def main():
    # Create environment
    env = BotanicalExplorerEnv(grid_size=10, num_plants=15)
    renderer = OpenGLRenderer(env)
    
    try:
        # Simulate and render an episode
        num_frames = simulate_episode(env, renderer)
        print(f"Rendered {num_frames} frames")
        
        # Save output
        renderer.save_frame("final_frame.png")
        renderer.save_gif("episode.gif", duration=0.3)
        print("Saved final_frame.png and episode.gif")
    finally:
        renderer.close()

if __name__ == "__main__":
    main()