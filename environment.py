import numpy as np

class Environment:
    """Labirinto aplinka reinforcement learning algoritmams"""
    
    def __init__(self, maze):
        """Inicializuoja aplinką
        
        Args:
            maze: labirintas (Maze objektas)
        """
        self.maze = maze
        self.grid = maze.grid
        self.start = maze.start
        self.end = maze.end
        self.size = maze.size
        
        self.current_pos = self.start
        self.steps = 0
        
        self.max_steps = self.size * self.size * 2
    
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    def reset(self):
        """Atstato aplinką į pradinę būseną
        
        Returns:
            state: pradinė būsena
        """
        self.current_pos = self.start
        self.steps = 0
        return self._get_state()
    
    def step(self, action):
        self.steps += 1
        
        new_row = self.current_pos[0] + self.actions[action][0]
        new_col = self.current_pos[1] + self.actions[action][1]
        new_pos = (new_row, new_col)
        
        reward = -0.05
        done = False
        
        if not self._is_valid_position(new_pos):
            reward = -1.0 
            return self._get_state(), reward, done, {}
        
        old_distance = abs(self.current_pos[0] - self.end[0]) + abs(self.current_pos[1] - self.end[1])
        new_distance = abs(new_row - self.end[0]) + abs(new_col - self.end[1])
        
        self.current_pos = new_pos
        
        if new_distance < old_distance:
            reward = 0.5
        
        if self.current_pos == self.end:
            reward = 10.0
            done = True
        
        if self.steps >= self.max_steps:
            done = True
        
        return self._get_state(), reward, done, {}
    
    def _is_valid_position(self, position):
        """Patikrina, ar pozicija yra leistina
        
        Args:
            position: pozicijos koordinatės (eilutė, stulpelis)
            
        Returns:
            bool: ar pozicija leistina
        """
        row, col = position
        
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        
        if self.grid[row, col] == 1:
            return False
        
        return True
    
    def _get_state(self):
        """Gauna dabartinę būseną
        
        Returns:
            state: dabartinė būsena (3D masyvas)
        """
        state = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        state[:, :, 0] = self.grid
        
        state[self.current_pos[0], self.current_pos[1], 1] = 1
        
        state[self.end[0], self.end[1], 2] = 1
        
        # (1, aukštis, plotis, kanalai)
        return np.expand_dims(state, axis=0)
    
    def get_path(self):
        """Grąžina dabartinę agento poziciją
        
        Returns:
            tuple: dabartinė pozicija (eilutė, stulpelis)
        """
        return self.current_pos