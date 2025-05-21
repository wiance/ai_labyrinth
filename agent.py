import numpy as np
import random
import tensorflow as tf
import time
import os

from tensorflow.keras.models import Sequential #pyright: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D #pyright: ignore
from tensorflow.keras.optimizers import Adam #pyright: ignore
from tensorflow.keras.models import load_model #pyright: ignore

from collections import deque

# TensorFlow optimizacijos
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.run_functions_eagerly(False)

class DQNAgent:
    """Deep Q-Network (DQN) agento klasė su optimizuotu CNN"""

    def __init__(self, input_shape=(10, 10, 3), action_space=4):
        """Inicializuoja DQN agentą"""
        self.input_shape = input_shape
        self.action_space = action_space

        maze_size = input_shape[0]

        if maze_size <= 5:
            self.memory = deque(maxlen=2000)
            self.gamma = 0.95
            self.epsilon_decay = 0.92
            self.learning_rate = 0.002
            print(f"Naudojami parametrai mažam {maze_size}x{maze_size} labirintui")

        elif maze_size <= 8:
            self.memory = deque(maxlen=2000)
            self.epsilon_decay = 0.9
            self.gamma = 0.93
            self.learning_rate = 0.003
            print(f"Naudojami parametrai vidutiniam {maze_size}x{maze_size} labirintui")
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01

        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()

        maze_size = self.input_shape[0]

        if maze_size <= 5:
            model.add(Conv2D(16, (3, 3), activation="relu", input_shape=self.input_shape, padding='same'))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(self.action_space, activation="linear"))

        elif maze_size <= 8:
            model.add(Conv2D(12, (3, 3), activation="relu", input_shape=self.input_shape, padding='same'))
            model.add(Flatten())
            model.add(Dense(48, activation="relu"))
            model.add(Dense(24, activation="relu"))
            model.add(Dense(self.action_space, activation="linear"))

        model.compile(
            loss="mse", 
            optimizer=Adam(learning_rate=self.learning_rate),
            run_eagerly=False
        )
        
        return model

    def remember(self, state, action, reward, next_state, done):
        """Įsimena patirtį"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Pasirenka veiksmą pagal epsilon-greedy metodą"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        """Optimizuotas mokymas su atsižvelgimu į jūsų sistemą"""
        if len(self.memory) < batch_size:
            return
            
        # Naudoti batch processing - efektyviau su moderniu CPU
        minibatch = random.sample(self.memory, batch_size)
        
        # Paruošti duomenis vienam apmokymui
        states = np.zeros((batch_size, *self.input_shape))
        targets = np.zeros((batch_size, self.action_space))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state[0]
            target = self.model.predict(state, verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                next_value = np.amax(self.model.predict(next_state, verbose=0)[0])
                target[action] = reward + self.gamma * next_value
                
            targets[i] = target
        
        # Vienas efektyvus apmokymo žingsnis
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

        # Sumažinti epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=75, batch_size=32):
        """Apmokyti agentą su optimizuotais parametrais"""
        scores = []
        total_start_time = time.time()
        
        print(f"Pradedamas CNN modelio apmokymas: {episodes} epizodai...")

        for episode in range(episodes):
            episode_start_time = time.time()
            state = env.reset()
            score = 0
            done = False
            steps = 0
            
            # Adaptyvus žingsnių limitas pagal labirinto dydį
            max_steps = min(50, env.size * env.size)
            
            while not done and steps < max_steps:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
                steps += 1

                # Mokytis kas 10 žingsnių
                if len(self.memory) > batch_size and steps % 10 == 0:
                    self.replay(batch_size)
                    
                # Pridėti mažą pauzę, kad CPU nesiektų 100%
                if steps % 20 == 0:
                    time.sleep(0.001)
            
            episode_time = time.time() - episode_start_time
            scores.append(score)
            
            print(
                f"Epizodas: {episode+1}/{episodes}, Žingsniai: {steps}, "
                f"Rezultatas: {score:.2f}, Epsilon: {self.epsilon:.2f}, "
                f"Laikas: {episode_time:.2f}s"
            )
            
            # Išsaugoti gerą modelį
            if score > 9.0 and env.current_pos == env.end:
                self.save_model(env.size)
                print(f"Išsaugotas geras modelis po epizodo {episode+1}")
        
        total_time = time.time() - total_start_time
        print(f"\nApmokymas baigtas! Bendras laikas: {total_time:.2f}s")
        print(f"Vidutinis laikas per epizodą: {total_time/episodes:.2f}s")
        
        return scores

    def save_model(self, size):
        """Išsaugo apmokytą modelį"""
        model_path = f"trained_models/maze_{size}x{size}_model.keras"
        self.model.save(model_path)
        print(f"Modelis sėkmingai išsaugotas: {model_path}")
        
    def load_model(self, size):
        """Įkelia apmokytą modelį, jei jis egzistuoja"""
        model_path = f"trained_models/maze_{size}x{size}_model.keras"
        try:
            if os.path.exists(model_path):
                print(f"Rastas modelio failas: {model_path}")
                self.model = load_model(model_path)
                print(f"Sėkmingai įkeltas apmokytas modelis!")
                # Nustatyti mažą epsilon reikšmę
                self.epsilon = 0.1
                return True
            else:
                print(f"Modelio failo nerasta: {model_path}")
                return False
        except Exception as e:
            print(f"Klaida įkeliant modelį: {e}")
            return False