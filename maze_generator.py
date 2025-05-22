import numpy as np
import random
from maze import Maze

class MazeGenerator:    
    def __init__(self):
        """Inicializuoja labirinto generatorių"""
        pass
    
    def generate_random_maze(self, size):
        """Generuoja atsitiktinį labirintą
        
        Args:
            size: labirinto dydis (size x size)
            
        Returns:
            Maze: sugeneruotas labirintas
        """
        maze = Maze(size)
        
        for row in range(size):
            for col in range(size):
                if random.random() < 0.3:
                    maze.set_wall(row, col)
        
        maze.set_start(0, 0)
        maze.set_end(size-1, size-1)
        
        self._ensure_path(maze)
        
        return maze
    
    def _ensure_path(self, maze):
        """Užtikrina, kad egzistuoja kelias nuo pradžios iki pabaigos
        Mokymosi fazė
        
        Args:
            maze: labirintas
        """
        for row in range(maze.size):
            maze.set_path(row, 0)
        
        for col in range(maze.size):
            maze.set_path(maze.size-1, col)

    
    def generate_maze_with_dfs(self, size):
        """Generuoja labirintą naudojant gilumos paiešką (DFS)
        
        Args:
            size: labirinto dydis (size x size)
            
        Returns:
            Maze: sugeneruotas labirintas
        """
        return self.generate_random_maze(size)