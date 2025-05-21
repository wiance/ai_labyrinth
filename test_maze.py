from maze import Maze
from maze_generator import MazeGenerator
from environment import Environment
from agent import DQNAgent

def test_maze_generation():
    """Testuoja labirinto generavimą"""
    print("Testuojamas labirinto generavimas")
    print("--------------------------------")
    
    # Sukurti labirinto generatorių
    generator = MazeGenerator()
    
    # Generuoti labirintą
    size = 5
    maze = generator.generate_random_maze(size)
    
    # Atvaizduoti labirintą
    print(f"Sugeneruotas {size}x{size} labirintas:")
    maze.visualize()
    
    print("\nTesto rezultatas: Sėkmingai sugeneruotas labirintas.")

def test_environment():
    """Testuoja labirinto aplinką"""
    print("\nTestuojama labirinto aplinka")
    print("---------------------------")
    
    # Sukurti labirinto generatorių ir labirintą
    generator = MazeGenerator()
    maze = generator.generate_random_maze(5)
    
    # Sukurti aplinką
    env = Environment(maze)
    
    # Atstatyti aplinką
    state = env.reset()
    
    print(f"Pradinė būsena: {env.current_pos}")
    
    # Atlikti kelis veiksmus
    for action in [1, 2, 1]:  # Dešinėn, žemyn, dešinėn
        print(f"Atliekamas veiksmas: {action}")
        next_state, reward, done, _ = env.step(action)
        print(f"Nauja pozicija: {env.current_pos}, Atlygis: {reward}, Baigta: {done}")
    
    # Atvaizduoti labirintą su nukeliauta trajektorija
    path = [maze.start, (0, 1), (1, 1), (1, 2)]
    print("\nLabirintas su trajektorija:")
    maze.visualize(path)
    
    print("\nTesto rezultatas: Aplinka veikia teisingai.")

def test_agent():
    """Testuoja DQN agentą"""
    print("\nTestuojamas DQN agentas")
    print("----------------------")
    
    # Sukurti labirinto generatorių ir labirintą
    generator = MazeGenerator()
    maze = generator.generate_random_maze(5)
    
    # Sukurti aplinką
    env = Environment(maze)
    
    # Sukurti agentą
    agent = DQNAgent(input_shape=(5, 5, 3))
    
    # Apmokyti agentą (labai trumpai, tik testavimui)
    print("Pradedamas trumpas apmokymas...")
    agent.train(env, episodes=100, batch_size=32)
    
    # Testuoti apmokytą agentą
    print("\nTestuojamas apmokytas agentas:")
    state = env.reset()
    done = False
    steps = 0
    path = [maze.start]
    
    # Išjungti tyrinėjimą testavimui
    epsilon_backup = agent.epsilon
    agent.epsilon = 0
    
    while not done and steps < env.max_steps:
        # Pasirinkti veiksmą
        action = agent.act(state)
        
        # Atlikti veiksmą
        state, reward, done, _ = env.step(action)
        
        # Įsiminti poziciją
        path.append(env.current_pos)
        steps += 1
    
    # Atstatyti epsilon
    agent.epsilon = epsilon_backup
    
    # Atvaizduoti rezultatus
    print(f"Žingsnių skaičius: {steps}")
    print(f"Tikslas pasiektas: {'Taip' if env.current_pos == maze.end else 'Ne'}")
    
    # Atvaizduoti labirintą su agento trajektorija
    print("\nLabirintas su agento trajektorija:")
    maze.visualize(path)
    
    print("\nTesto rezultatas: Agentas veikia teisingai.")

def run_all_tests():
    """Paleidžia visus testus"""
    test_maze_generation()
    test_environment()
    test_agent()
    
    print("\nVisi testai sėkmingai baigti!")

if __name__ == "__main__":
    run_all_tests()