import numpy as np

class Environment:
    """Labirinto aplinka reinforcement learning algoritmams"""
    
    def __init__(self, maze):
        """Inicializuoja aplinką
        
        Args:
            maze: labirintas (Maze objektas)
        """
        self.maze = maze
        self.grid = maze.grid  # Nuoroda į labirinto grid masyvą
        self.start = maze.start  # Pradžios pozicija
        self.end = maze.end  # Pabaigos pozicija
        self.size = maze.size  # Labirinto dydis
        
        # Agento būsena
        self.current_pos = self.start  # Dabartinė pozicija
        self.steps = 0  # Žingsnių skaičius
        
        # Maksimalus žingsnių skaičius - dvigubas labirinto dydis
        self.max_steps = self.size * self.size * 2
        
        # Galimi veiksmai: 0-aukštyn, 1-dešinėn, 2-žemyn, 3-kairėn
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
        
        # Gauname naujas koordinates
        new_row = self.current_pos[0] + self.actions[action][0]
        new_col = self.current_pos[1] + self.actions[action][1]
        new_pos = (new_row, new_col)
        
        # Patikrinti, ar naujos koordinatės yra leistinos
        # Pradinė bauda
        reward = -0.05  # Sumažinama iš -0.1 į -0.01
        done = False
        
        # Jei išėjome už ribų arba atsitrenkėme į sieną
        if not self._is_valid_position(new_pos):
            reward = -1.0  # Bauda už neteisingą veiksmą
            return self._get_state(), reward, done, {}
        
        # Apskaičiuoti atstumo pokytį iki tikslo
        old_distance = abs(self.current_pos[0] - self.end[0]) + abs(self.current_pos[1] - self.end[1])
        new_distance = abs(new_row - self.end[0]) + abs(new_col - self.end[1])
        
        # Atnaujinti poziciją
        self.current_pos = new_pos
        
        # Atlygis pagal atstumo pokytį - ar priartėjome prie tikslo?
        if new_distance < old_distance:
            reward = 0.5  # Teigiamas atlygis už priartėjimą
        
        # Patikrinti, ar pasiekėme tikslą
        if self.current_pos == self.end:
            reward = 10.0  # Didelis atlygis už tikslą!
            done = True
        
        # Patikrinti maksimalų žingsnių skaičių
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
        
        # Patikrinti, ar neišėjome už labirinto ribų
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        
        # Patikrinti, ar neatsitrenkėme į sieną
        if self.grid[row, col] == 1:
            return False
        
        return True
    
    def _get_state(self):
        """Gauna dabartinę būseną
        
        Returns:
            state: dabartinė būsena (3D masyvas)
        """
        # Sukurti 3D masyvą (aukštis, plotis, 3 kanalai)
        state = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # 1 kanalas: sienos (1 = siena, 0 = kelias)
        state[:, :, 0] = self.grid
        
        # 2 kanalas: dabartinė pozicija (1 = dabartinė pozicija, 0 = kitur)
        state[self.current_pos[0], self.current_pos[1], 1] = 1
        
        # 3 kanalas: tikslas (1 = tikslas, 0 = kitur)
        state[self.end[0], self.end[1], 2] = 1
        
        # Pridėti batch dimensiją (1, aukštis, plotis, kanalai)
        return np.expand_dims(state, axis=0)
    
    def get_path(self):
        """Grąžina dabartinę agento poziciją
        
        Returns:
            tuple: dabartinė pozicija (eilutė, stulpelis)
        """
        return self.current_pos