import numpy as np
import pickle
import os
import re
from multiprocessing import Pool
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Policy:
    def __init__(self, game):
        self.game = game
        self.max_steps = game.get_current_state()[3]
        self.max_line_length = game._max_length

        if os.path.exists('zuma_policy_nn.pkl'):
            self.load_policy('zuma_policy_nn.pkl')
        else:
            self._initialize_model()
            self._generate_policy()
            self.save_policy('zuma_policy_nn.pkl')

    def _initialize_model(self):
        """Initialize the neural network model."""
        self.model = Sequential([
            Dense(64, input_shape=(11,), activation='relu'),  # Input: last 10 balls + current ball
            Dense(64, activation='relu'),
            Dense(12, activation='linear')  # Output: Q-values for each action
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def _compress_state(self, line, ball):
        """Compress the state into a fixed-size vector."""
        if not line:
            return [0] * 10 + [ball]  # Pad with zeros if the line is empty

        # Limit state to the last 10 balls
        compressed = line[-10:] if len(line) > 10 else line
        compressed = compressed + [0] * (10 - len(compressed))  # Pad with zeros if necessary
        return compressed + [ball]  # Add the current ball

    def _find_matches(self, line, index, ball):
        """Find the number of matches if a ball is inserted at the given index."""
        if index == -1:
            return 0

        if len(line) >= self.max_line_length:
            return -1

        new_line = line.copy()
        new_line.insert(index, ball)

        count = 1
        left = index - 1
        right = index + 1

        while left >= 0 and new_line[left] == ball:
            count += 1
            left -= 1

        while right < len(new_line) and new_line[right] == ball:
            count += 1
            right += 1

        if count >= 2:
            return count
        return 0

    def _find_chain_potential(self, line, index, ball):
        """Estimate the potential for chain reactions after inserting a ball."""
        new_line = line.copy()
        new_line.insert(index, ball)

        total_cleared = 0
        while True:
            burstable = re.finditer(r'1{3,}|2{3,}|3{3,}|4{3,}', ''.join([str(i) for i in new_line]))
            cleared = False
            for group in burstable:
                if index in range(group.span()[0], group.span()[1]):
                    total_cleared += (group.span()[1] - group.span()[0])
                    new_line = new_line[:group.span()[0]] + new_line[group.span()[1]:]
                    cleared = True
                    break
            if not cleared:
                break
        return total_cleared

    def _train_episode(self, episode):
        """Train a single episode and update the neural network."""
        self.game.reset(generate_new_game=True)
        state = self.game.get_current_state()
        epsilon = max(0.01, 1.0 * (0.999 ** episode))  # Decay epsilon based on episode

        while not state[2] >= self.max_steps:
            line, ball = state[:2]
            state_vector = self._compress_state(line, ball)
            valid_actions = list(range(-1, len(line) + 1))

            if np.random.random() < epsilon:
                action = np.random.choice(valid_actions)
            else:
                q_values = self.model.predict(np.array([state_vector]), verbose=0)[0]
                action = valid_actions[np.argmax(q_values[:len(valid_actions)])]

            new_line, new_ball, reward, done = self.game.submit_next_action(action)

            if not done:
                next_state_vector = self._compress_state(new_line, new_ball)
                next_q_values = self.model.predict(np.array([next_state_vector]), verbose=0)[0]
                target = reward + 0.9 * np.max(next_q_values[:len(valid_actions)])
            else:
                target = reward

            target_vector = self.model.predict(np.array([state_vector]), verbose=0)[0]
            target_vector[action + 1] = target
            self.model.fit(np.array([state_vector]), np.array([target_vector]), verbose=0)

            state = (new_line, new_ball, state[2], state[3])

    def _generate_policy(self):
        """Generate the policy using neural network-based Q-learning."""
        episodes = 1000000  # Increased number of episodes
        with Pool() as pool:
            pool.map(self._train_episode, range(episodes))

    def choose_next_action(self, state):
        """Choose the next action based on the current state and policy."""
        line, ball = state
        state_vector = self._compress_state(line, ball)
        valid_actions = list(range(-1, len(line) + 1))

        q_values = self.model.predict(np.array([state_vector]), verbose=0)[0]
        return valid_actions[np.argmax(q_values[:len(valid_actions)])]

    def save_policy(self, file):
        """Save the policy to a file."""
        self.model.save(file)

    def load_policy(self, file):
        """Load the policy from a file."""
        from tensorflow.keras.models import load_model
        self.model = load_model(file)