import numpy as np
from maze import Maze

maze_layout = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0]
])

maze = Maze(maze_layout, (0, 0), (4, 4))

maze.show_maze()