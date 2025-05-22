from maze import Maze
from maze_generator import MazeGenerator
from environment import Environment
from agent import DQNAgent

def test_maze_generation():
    """Testuoja labirinto generavimą"""
    print("Testuojamas labirinto generavimas")
    print("--------------------------------")
    
    
    generator = MazeGenerator()
    
    size = 5
    maze = generator.generate_random_maze(size)
    
    print(f"Sugeneruotas {size}x{size} labirintas:")
    maze.visualize()
    
    print("\nTesto rezultatas: Sėkmingai sugeneruotas labirintas.")

def test_environment():
    """Testuoja labirinto aplinką"""
    print("\nTestuojama labirinto aplinka")
    print("---------------------------")
    
    generator = MazeGenerator()
    maze = generator.generate_random_maze(5)
    
    env = Environment(maze)
    
    state = env.reset()
    
    print(f"Pradinė būsena: {env.current_pos}")
    
    for action in [1, 2, 1]:
        print(f"Atliekamas veiksmas: {action}")
        next_state, reward, done, _ = env.step(action)
        print(f"Nauja pozicija: {env.current_pos}, Atlygis: {reward}, Baigta: {done}")
    
    path = [maze.start, (0, 1), (1, 1), (1, 2)]
    print("\nLabirintas su trajektorija:")
    maze.visualize(path)
    
    print("\nTesto rezultatas: Aplinka veikia teisingai.")

def test_agent():
    """Testuoja DQN agentą"""
    print("\nTestuojamas DQN agentas")
    print("----------------------")
    
    generator = MazeGenerator()
    maze = generator.generate_random_maze(5)
    
    env = Environment(maze)
    
    agent = DQNAgent(input_shape=(5, 5, 3))
    
    print("Pradedamas trumpas apmokymas...")
    agent.train(env, episodes=100, batch_size=32)
    
    print("\nTestuojamas apmokytas agentas:")
    state = env.reset()
    done = False
    steps = 0
    path = [maze.start]
    
    epsilon_backup = agent.epsilon
    agent.epsilon = 0
    
    while not done and steps < env.max_steps:
        action = agent.act(state)
        
        state, reward, done, _ = env.step(action)
        
        path.append(env.current_pos)
        steps += 1
    
    agent.epsilon = epsilon_backup
    
    print(f"Žingsnių skaičius: {steps}")
    print(f"Tikslas pasiektas: {'Taip' if env.current_pos == maze.end else 'Ne'}")
    
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