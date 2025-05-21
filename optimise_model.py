
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential #pyright: ignore
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout #pyright: ignore
from tensorflow.keras.optimizers import Adam #pyright: ignore
import os

# Kad TensorFlow nekeltų bereikalingų pranešimų
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Supaprastinta labirinto aplinka greitam testavimui
class SimpleEnvironment:
    """Supaprastinta labirinto aplinka greitam testavimui"""
    
    def __init__(self, size=5):
        self.size = size
        self.reset()
    
    def reset(self):
        """Iš naujo nustato aplinką"""
        # Sukurti supaprastintą reprezentaciją labirinto būsenai
        self.state = np.zeros((self.size, self.size, 3))
        
        # Atsitiktinai parinkti agento poziciją
        self.agent_pos = (0, 0)
        
        # Tikslo pozicija visada kampe
        self.goal_pos = (self.size-1, self.size-1)
        
        # Atsitiktinai sugeneruoti sienas (1 - siena, 0 - kelias)
        # Paprasta versija - tiesiog keletas atsitiktinių sienų
        self.walls = np.zeros((self.size, self.size))
        
        wall_count = self.size  # Sienų skaičius proporcingas labirinto dydžiui
        for _ in range(wall_count):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            # Nestatyti sienų starto ir tikslo pozicijose
            if (x, y) != self.agent_pos and (x, y) != self.goal_pos:
                self.walls[y, x] = 1
        
        # Atnaujinti būseną
        self._update_state()
        
        return self._get_observation()
    
    def _update_state(self):
        """Atnaujina būseną pagal dabartinę poziciją"""
        # Išvalyti būseną
        self.state = np.zeros((self.size, self.size, 3))
        
        # Raudonas kanalas - sienos
        self.state[:, :, 0] = self.walls
        
        # Žalias kanalas - agentas
        self.state[self.agent_pos[1], self.agent_pos[0], 1] = 1
        
        # Mėlynas kanalas - tikslas
        self.state[self.goal_pos[1], self.goal_pos[0], 2] = 1
    
    def _get_observation(self):
        """Grąžina stebėjimą kaip 3D masyvą"""
        # Grąžina būseną tokiu formatu, kokį tikisi DQN (batch dydis 1)
        return np.expand_dims(self.state, axis=0)
    
    def step(self, action):
        """Vykdo veiksmą aplinkoje
        
        Args:
            action: 0 - aukštyn, 1 - dešinėn, 2 - žemyn, 3 - kairėn
        
        Returns:
            observation: nauja būsena
            reward: atlygis
            done: ar žaidimas baigtas
            info: papildoma informacija
        """
        # Dabartinė pozicija
        x, y = self.agent_pos
        
        # Nauja pozicija pagal veiksmą
        if action == 0:  # Aukštyn
            new_pos = (x, max(0, y-1))
        elif action == 1:  # Dešinėn
            new_pos = (min(self.size-1, x+1), y)
        elif action == 2:  # Žemyn
            new_pos = (x, min(self.size-1, y+1))
        elif action == 3:  # Kairėn
            new_pos = (max(0, x-1), y)
        else:
            raise ValueError(f"Neteisingas veiksmas: {action}")
        
        # Patikrinti, ar nauja pozicija yra siena
        new_x, new_y = new_pos
        done = False  # Inicializuojame done su numatyta reikšme False
        
        if self.walls[new_y, new_x] == 1:
            # Atsitrenkė į sieną - pasilikti toje pačioje pozicijoje
            reward = -1  # Bauda už atsitrenkimą į sieną
        else:
            # Jei ne siena, atnaujinti poziciją
            self.agent_pos = new_pos
            
            # Atlygiai
            if self.agent_pos == self.goal_pos:
                reward = 10  # Didelis atlygis už tikslo pasiekimą
                done = True
            else:
                reward = -0.1  # Maža bauda už kiekvieną žingsnį
        
        # Atnaujinti būseną
        self._update_state()
        
        # Grąžinti stebėjimą, atlygį, baigties požymį ir tuščią info žodyną
        return self._get_observation(), reward, done, {}

# Testuojama modifikuota DQNAgent versija
class DQNAgentTest:
    """Deep Q-Network (DQN) agento klasė su modifikuojama struktūra"""

    def __init__(self, input_shape=(5, 5, 3), action_space=4, model_type="simple"):
        """Inicializuoja DQN agentą
        
        Args:
            input_shape: įvesties forma (aukštis, plotis, kanalai)
            action_space: galimų veiksmų skaičius
            model_type: modelio tipas ('simple', 'medium', 'complex', 'conv')
        """
        self.input_shape = input_shape  # Įvesties forma
        self.action_space = action_space  # Galimų veiksmų skaičius
        self.memory = deque(maxlen=2000)  # Patirties atmintis
        self.epsilon = 1.0  # Epsilon-godusis parametras
        self.epsilon_min = 0.01  # Minimali epsilon reikšmė
        self.gamma = 0.93  # Diskonto faktorius
        self.epsilon_decay = 0.8  # Epsilon mažėjimo greitis
        self.learning_rate = 0.009  # Mokymosi greitis
        self.model_type = model_type
        
        self.model = self._build_model()
        print(f"Sukurtas {self.model_type} modelis:")
        self.model.summary()

    def _build_model(self):
        """Sukuria neuroninį tinklą pagal nurodytą tipą"""
        model = Sequential()
        
        if self.model_type == "simple":
            # Paprastas modelis - 1 paslėptas sluoksnis
            model.add(Flatten(input_shape=self.input_shape))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(self.action_space, activation="linear"))
            
        elif self.model_type == "medium":
            # Vidutinio sudėtingumo modelis - 2 paslėpti sluoksniai
            model.add(Flatten(input_shape=self.input_shape))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(self.action_space, activation="linear"))
            
        elif self.model_type == "complex":
            # Sudėtingesnis modelis - 3 paslėpti sluoksniai su dropout
            model.add(Flatten(input_shape=self.input_shape))
            model.add(Dense(128, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(self.action_space, activation="linear"))
            
        elif self.model_type == "conv":
            # Konvoliucinis modelis
            model.add(Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape, padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(16, (3, 3), activation="relu", padding='same'))
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(self.action_space, activation="linear"))
            
        else:
            raise ValueError(f"Nežinomas modelio tipas: {self.model_type}")
        
        # Kompiliuoti modelį
        model.compile(
            loss="mse", optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        return model

    def remember(self, state, action, reward, next_state, done):
        """Įsimena patirtį"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Pasirenka veiksmą pagal epsilon-godųjį metodą"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=32):
        """Apmoko modelį naudojant patirties atkartojimą"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                next_value = np.amax(self.model.predict(next_state, verbose=0)[0])
                target = reward + self.gamma * next_value
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=20, batch_size=16, max_steps=50):
        """Apmoko agentą nurodytame aplinkos modelyje"""
        scores = []
        steps_per_episode = []
        
        for episode in range(episodes):
            state = env.reset()
            score = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # Pasirinkti veiksmą
                action = self.act(state)
                
                # Atlikti veiksmą
                next_state, reward, done, _ = env.step(action)
                
                # Įsiminti patirtį
                self.remember(state, action, reward, next_state, done)
                
                # Pereiti į kitą būseną
                state = next_state
                
                # Pridėti atlygį prie rezultato
                score += reward
                steps += 1
                
                # Apmokyti modelį
                if len(self.memory) > batch_size and steps % 10 == 0:
                    self.replay(batch_size)
            
            scores.append(score)
            steps_per_episode.append(steps)
            
            print(
                f"Epizodas: {episode+1}/{episodes}, Žingsniai: {steps}, Rezultatas: {score:.2f}, Epsilon: {self.epsilon:.2f}"
            )
        
        return scores, steps_per_episode

def test_model_types():
    """Testuoja skirtingus modelių tipus ir palygina rezultatus"""
    # Parametrai
    maze_size = 5
    episodes = 50  # Mažesnis epizodų skaičius greitam testavimui
    model_types = ["simple", "medium", "complex", "conv"]
    
    # Rezultatai
    all_scores = {}
    all_steps = {}
    
    # Testuoti kiekvieną modelio tipą
    for model_type in model_types:
        print(f"\n==== Testuojamas modelio tipas: {model_type} ====")
        
        # Sukurti aplinką
        env = SimpleEnvironment(size=maze_size)
        
        # Sukurti agentą
        agent = DQNAgentTest(
            input_shape=(maze_size, maze_size, 3),
            action_space=4,
            model_type=model_type
        )
        
        # Apmokyti agentą
        scores, steps = agent.train(env, episodes=episodes)
        
        # Išsaugoti rezultatus
        all_scores[model_type] = scores
        all_steps[model_type] = steps
    
    # Atvaizduoti rezultatus
    plt.figure(figsize=(15, 10))
    
    # Atlygių grafikas
    plt.subplot(2, 1, 1)
    for model_type, scores in all_scores.items():
        plt.plot(scores, label=model_type)
    plt.title('Vidutinis atlygis per epizodus')
    plt.xlabel('Epizodas')
    plt.ylabel('Atlygis')
    plt.legend()
    plt.grid(True)
    
    # Žingsnių grafikas
    plt.subplot(2, 1, 2)
    for model_type, steps in all_steps.items():
        plt.plot(steps, label=model_type)
    plt.title('Žingsnių skaičius per epizodus')
    plt.xlabel('Epizodas')
    plt.ylabel('Žingsnių skaičius')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Išvesti papildomą statistiką
    print("\n==== Rezultatai ====")
    for model_type in model_types:
        avg_score = np.mean(all_scores[model_type][-10:])  # Paskutinių 10 epizodų vidurkis
        avg_steps = np.mean(all_steps[model_type][-10:])
        print(f"{model_type.capitalize()} modelis:")
        print(f"  Vidutinis atlygis (paskutiniai 10 epizodų): {avg_score:.2f}")
        print(f"  Vidutinis žingsnių skaičius (paskutiniai 10 epizodų): {avg_steps:.2f}")

if __name__ == "__main__":
    # Nustatyti atsitiktinių skaičių sėklą, kad rezultatai būtų pakartojami
    np.random.seed(42)
    random.seed(42)
    
    # Paleisti testavimą
    print("Pradedamas modelių testavimas...")
    test_model_types()