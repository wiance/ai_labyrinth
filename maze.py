import numpy as np
import matplotlib.pyplot as plt
import time

class Maze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze    
        self.maze_height = maze.shape[0]
        self.maze_width = maze.shape[1]
        self.start_position = start_position
        self.goal_position = goal_position

    def show_maze(self):
        plt.figure(figsize=(5,5))

        plt.imshow(self.maze, cmap='gray')

        plt.text(self.start_position[0], self.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[0], self.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)

        plt.xticks([]), plt.yticks([])
        plt.show()

