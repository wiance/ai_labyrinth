import optuna
import numpy as np
import random
import tensorflow as tf
import os
from collections import deque
from tensorflow.keras.models import Sequential #pyright: ignore
from tensorflow.keras.layers import Dense, Flatten #pyright: ignore
from tensorflow.keras.optimizers import Adam #pyright: ignore

# TensorFlow optimizavimai
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(False)

# Išjungti visas GPU, kad būtų pastovus veikimas
tf.config.set_visible_devices([], 'GPU')

from maze_generator import MazeGenerator
from environment import Environment
from agent import DQNAgent

def objective(trial):
    """Optuna objektyvinė funkcija su 5x5 labirintu"""
    print(f"\nPradedamas Trial {trial.number}")
    
    gamma = trial.suggest_float('gamma', 0.9, 0.99, step=0.03)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.8, 0.95, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
    
    fixed_params = {
        'batch_size': 16,
        'hidden_units': 32,
        'max_steps': 50,  # 5x5 labirintui
        'episodes': 10
    }

    generator = MazeGenerator()
    maze = generator.generate_random_maze(5)
    
    for row in range(5):
        for col in range(5):
            if maze.is_wall(row, col):
                if random.random() < 0.3:  # 30% šansų pašalinti sieną
                    maze.set_path(row, col)
    
    env = Environment(maze)
    
    input_shape = (5, 5, 3)
    agent = DQNAgent(input_shape=input_shape)
    agent.gamma = gamma
    agent.epsilon_decay = epsilon_decay
    
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(fixed_params['hidden_units'], activation='relu'))
    model.add(Dense(4, activation='linear'))
    model.compile(
        loss='mse', 
        optimizer=Adam(learning_rate=learning_rate),
        run_eagerly=False
    )
    agent.model = model
    
    success_count = 0
    scores = []
    
    print(f"Trial {trial.number}: Pradedamas apmokymas ({fixed_params['episodes']} epizodų)")
    
    for episode in range(fixed_params['episodes']):
        state = env.reset()
        score = 0
        done = False
        steps = 0
        
        while not done and steps < fixed_params['max_steps']:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            steps += 1
            
            # Apmokyti modelį kas 10 žingsnių
            if len(agent.memory) > fixed_params['batch_size'] and steps % 10 == 0:
                # Optimizuotas apmokymas
                batch_size = fixed_params['batch_size']
                batch = random.sample(agent.memory, batch_size)
                
                # Paruošti duomenis vienam apmokymui
                states = np.zeros((batch_size, *input_shape))
                targets = np.zeros((batch_size, 4))
                
                for i, (s, a, r, ns, d) in enumerate(batch):
                    states[i] = s[0]
                    target = model.predict(s, verbose=0)[0]
                    if d:
                        target[a] = r
                    else:
                        target[a] = r + gamma * np.max(model.predict(ns, verbose=0)[0])
                    targets[i] = target
            
                model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        if env.current_pos == env.end:
            success_count += 1
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= epsilon_decay
        
        scores.append(score)
        print(f"Trial {trial.number}, Epizodas {episode+1}/{fixed_params['episodes']}: " +
              f"Rezultatas={score:.2f}, Epsilon={agent.epsilon:.2f}, Žingsniai={steps}")
    
    avg_score = np.mean(scores)
    success_rate = success_count / fixed_params['episodes']
    
    print(f"Trial {trial.number} baigtas: Sėkmė={success_rate*100:.1f}%, Vidutinis rezultatas={avg_score:.2f}")
    
    return success_rate * 0.7 + avg_score / 100 * 0.3

def optimize_and_save():
    print("Pradedamas hiperparametrų optimizavimas")
    
    study = optuna.create_study(direction='maximize')
    
    study.optimize(objective, n_trials=8)
    
    print("\nGeriausi parametrai:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    optimize_and_save()