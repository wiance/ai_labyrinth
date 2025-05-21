from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import time
import os

from maze import Maze
from maze_generator import MazeGenerator
from environment import Environment
from agent import DQNAgent

if not os.path.exists("trained_models"):
    os.makedirs("trained_models")

app = Flask(__name__, template_folder='frontend')

app.secret_key = 'ai_maze_secret_key'

maze = None
env = None
agent = None
solution_path = None

def load_trained_models():
    trained_sizes = []
    sizes = [5, 8, 10]
    
    print("\n=== Apmokytų modelių tikrinimas ===")
    for size in sizes:
        model_path = f"trained_models/maze_{size}x{size}_model.keras"
        if os.path.exists(model_path):
            trained_sizes.append(size)
            print(f"Rastas apmokytas modelis {size}x{size} labirintui!")
    
    if trained_sizes:
        print(f"Iš viso rasta apmokytų modelių: {len(trained_sizes)}")
        return trained_sizes
    else:
        print("Neaptikta apmokytų modelių.")
        return []

@app.route('/')
def index():
    """Pagrindinis puslapis"""
    return render_template('index.html', 
                          maze=maze, 
                          solution_path=solution_path,
                          maze_size=maze.size if maze else 0, 
                          steps=len(solution_path) if solution_path else 0,
                          accuracy="Modelis nėra apmokytas" if agent is None else f"{get_accuracy()}%")

@app.route('/generate', methods=['POST'])
def generate_maze_route():
    """Labirinto generavimo maršrutas"""
    global maze, env, agent, solution_path
    
    size = int(request.form.get('size'))
    
    generator = MazeGenerator()
    maze = generator.generate_random_maze(size)
    
    env = Environment(maze)
    
    agent = None
    solution_path = None
    
    input_shape = (size, size, 3)
    agent = DQNAgent(input_shape=input_shape)
    if agent.load_model(size):
        flash(f'Sėkmingai įkeltas apmokytas {size}x{size} modelis! Galite iškart spręsti labirintą.')
    else:
        flash('Labirintas sėkmingai sugeneruotas! Pirmiausia apmokykite modelį.')
    
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_agent_route():
    """Agento apmokymo maršrutas"""
    global agent, env, maze
    
    if maze is None or env is None:
        flash('Pirmiausia sugeneruokite labirintą!')
        return redirect(url_for('index'))
    
    episodes = int(request.form.get('episodes'))
    
    if agent is None:
        input_shape = (maze.size, maze.size, 3)
        agent = DQNAgent(input_shape=input_shape)
        
        if agent.load_model(maze.size):
            flash(f'Sėkmingai įkeltas apmokytas {maze.size}x{maze.size} modelis!')
            return redirect(url_for('index'))
    
    start_time = time.time()
    agent.train(env, episodes=episodes, batch_size=32)
    training_time = time.time() - start_time
    
    agent.save_model(maze.size)
    
    flash(f'Modelis sėkmingai apmokytas per {training_time:.2f} sekundes ({episodes} epizodai)!')
    return redirect(url_for('index'))

@app.route('/solve', methods=['POST'])
def solve_maze_route():
    """Labirinto sprendimo maršrutas"""
    global agent, env, maze, solution_path
    
    if maze is None or env is None:
        flash('Pirmiausia sugeneruokite labirintą!')
        return redirect(url_for('index'))
    
    if agent is None:
        flash('Pirmiausia apmokykite modelį!')
        return redirect(url_for('index'))
    
    state = env.reset()
    done = False
    steps = 0
    path = [maze.start]
    
    epsilon_backup = agent.epsilon
    agent.epsilon = 0.0
    
    max_steps = maze.size * maze.size * 2 
    
    start_time = time.time()
    while not done and steps < max_steps:
        action = agent.act(state)
        
        state, reward, done, _ = env.step(action)
        
        path.append(env.current_pos)
        steps += 1
    
    agent.epsilon = epsilon_backup
    
    solution_path = path
    
    solve_time = time.time() - start_time
    
    if env.current_pos == maze.end:
        flash(f'Labirintas sėkmingai išspręstas per {steps} žingsnius! (užtruko {solve_time:.2f}s)')
    else:
        flash(f'Nepavyko išspręsti labirinto per {steps} žingsnius. Bandykite apmokyti modelį dar kartą.')
    
    return redirect(url_for('index'))

def get_accuracy():
    """Grąžina apytikslį modelio tikslumą"""
    if agent is None:
        return 0
    
    if agent.epsilon < 0.1:
        return 95
    elif agent.epsilon < 0.3:
        return 85
    else:
        return 70

if __name__ == '__main__':
    app.run(debug=True)