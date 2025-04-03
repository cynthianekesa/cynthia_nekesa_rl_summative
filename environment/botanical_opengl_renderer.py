import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
from PIL import Image
import imageio
import os
from typing import List
from environment.botanical_env import BotanicalExplorerEnv

class OpenGLRenderer:
    def __init__(self, env: BotanicalExplorerEnv):
        self.env = env
        self.cell_size = 0.2
        self.window_size = 800
        self.frames: List[np.ndarray] = []
        
        # Initialize Pygame with OpenGL context
        pygame.init()
        self.display = (self.window_size, self.window_size)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        
        # Set up OpenGL perspective
        gluPerspective(45, (self.display[0]/self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
        
        # Colors (RGBA)
        self.colors = {
            'background': (0.9, 0.9, 0.9, 1.0),
            'grid': (0.7, 0.7, 0.7, 1.0),
            'agent': (1.0, 0.0, 0.0, 1.0),
            'unidentified': (0.13, 0.54, 0.13, 1.0),
            'identified': (1.0, 1.0, 0.0, 1.0)
        }
        
    def _setup_view(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        grid_half = (self.env.grid_size * self.cell_size) / 2
        glOrtho(-grid_half, grid_half, -grid_half, grid_half, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
    def _draw_grid(self):
        glColor4f(*self.colors['grid'])
        glLineWidth(1.0)
        
        half_size = (self.env.grid_size * self.cell_size) / 2
        glBegin(GL_LINES)
        for i in range(self.env.grid_size + 1):
            x = -half_size + i * self.cell_size
            glVertex2f(x, -half_size)
            glVertex2f(x, half_size)
            
            y = -half_size + i * self.cell_size
            glVertex2f(-half_size, y)
            glVertex2f(half_size, y)
        glEnd()
    
    def _draw_plants(self):
        half_size = self.env.grid_size // 2
        glBegin(GL_QUADS)
        for plant in self.env.plants:
            x = (plant.x - half_size) * self.cell_size
            y = (plant.y - half_size) * self.cell_size
            
            if plant.identified:
                glColor4f(*self.colors['identified'])
            else:
                glColor4f(*self.colors['unidentified'])
                
            glVertex2f(x, y)
            glVertex2f(x + self.cell_size, y)
            glVertex2f(x + self.cell_size, y + self.cell_size)
            glVertex2f(x, y + self.cell_size)
        glEnd()
    
    def _draw_agent(self):
        half_size = self.env.grid_size // 2
        x = (self.env.agent_pos[0] - half_size) * self.cell_size + self.cell_size/2
        y = (self.env.agent_pos[1] - half_size) * self.cell_size + self.cell_size/2
        
        glColor4f(*self.colors['agent'])
        glBegin(GL_TRIANGLE_FAN)
        for i in range(360):
            angle = np.radians(i)
            glVertex2f(
                x + np.cos(angle) * self.cell_size/3,
                y + np.sin(angle) * self.cell_size/3
            )
        glEnd()
    
    def _capture_frame(self):
        buffer = glReadPixels(0, 0, *self.display, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", self.display, buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        self.frames.append(np.array(image))
    
    def render_frame(self, filename: str = None):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.colors['background'])
        
        self._setup_view()
        self._draw_grid()
        self._draw_plants()
        self._draw_agent()
        
        pygame.display.flip()
        self._capture_frame()
        
        if filename:
            self.save_frame(filename)
    
    def save_frame(self, filename: str):
        if not self.frames:
            return
        imageio.imwrite(filename, self.frames[-1])
    
    def save_gif(self, filename: str, duration: float = 0.5):
        if len(self.frames) < 2:
            return
        imageio.mimsave(filename, self.frames, duration=duration)
    
    def clear_frames(self):
        self.frames = []
    
    def close(self):
        pygame.quit()