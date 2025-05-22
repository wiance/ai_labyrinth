import numpy as np

class Maze:
    """Labirinto klasė, skirta saugoti ir manipuliuoti labirintą"""
    
    def __init__(self, size):
        """Inicializuoja labirintą
        
        Args:
            size: labirinto dydis (size x size)
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start = (0, 0)
        self.end = (size-1, size-1)
    
    def set_wall(self, row, col):
        """Nustato sieną nurodytoje pozicijoje"""
        self.grid[row, col] = 1
    
    def set_path(self, row, col):
        """Nustato kelią nurodytoje pozicijoje"""
        self.grid[row, col] = 0
    
    def is_wall(self, row, col):
        """Patikrina, ar nurodytoje pozicijoje yra siena"""
        return self.grid[row, col] == 1
    
    def is_path(self, row, col):
        """Patikrina, ar nurodytoje pozicijoje yra kelias"""
        return self.grid[row, col] == 0
    
    def set_start(self, row, col):
        """Nustato pradžios tašką"""
        self.start = (row, col)
        self.set_path(row, col)
    
    def set_end(self, row, col):
        """Nustato pabaigos tašką"""
        self.end = (row, col)
        self.set_path(row, col)
    
    def get_neighbors(self, row, col):
        """Grąžina visus gretimus taškus
        
        Args:
            row: eilutės indeksas
            col: stulpelio indeksas
            
        Returns:
            list: gretimų taškų sąrašas [(eilutė, stulpelis), ...]
        """
        neighbors = []
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)] 
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
