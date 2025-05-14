import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore
from tensorflow.keras.layers import Dense # pyright: ignore
from tensorflow.keras.optimizers import Adam # pyright: ignore

class DQNAgent:
    def __init__(self, maze, learning_rate=0.001, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        self.maze = maze
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes
        
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=2, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4, activation='linear'))
        
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_exploration_rate(self, current_episode):
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)
        return exploration_rate

    def get_action(self, state, current_episode):
        exploration_rate = self.get_exploration_rate(current_episode)
        
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)
        
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def update_model(self, state, action, next_state, reward):
        q_values = self.model.predict(np.array([state]))
        next_q_values = self.model.predict(np.array([next_state]))
        
        q_values[0][action] = reward + self.discount_factor * np.max(next_q_values[0])
        
        self.model.fit(np.array([state]), q_values, epochs=1, verbose=0)
