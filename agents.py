import random

import numpy as np
import tensorflow as tf


class SmartAgent:
    def __init__(self, num_features: int, num_actions: int, saved_model_path = None):
        self.model = self.build_model(num_features, num_actions)
        self.memory = []

        if saved_model_path != None:
            self.model.load_weights(saved_model_path)

    def build_model(self, num_features: int, num_actions: int) -> tf.keras.Sequential:
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(num_features,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(num_actions, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()
        return model

    def select_action(self, observation: np.ndarray, actions: np.ndarray) -> int:
        predictions = self.model.predict(np.array([observation]), verbose=0)
        return actions[np.argmax(predictions[0])]

    def update_model(self):
        filtered_memory = self.filter_memory(filtering_ratio=0.3)

        train_x = []
        train_y = []
        for interaction in filtered_memory:
            train_x.append(interaction.observation)
            train_y.append(interaction.action - 16)

        train_y_cat = tf.keras.utils.to_categorical(train_y, 8)

        self.model.fit(np.array(train_x), train_y_cat, epochs=20)

        self.memory.clear()

    def filter_memory(self, filtering_ratio: float):
        self.memory.sort(key=lambda x: x.reward, reverse=True)

        filtered_length = int(len(self.memory) * filtering_ratio)

        filtered_memory = []

        for i in range(filtered_length):
            filtered_memory.append(self.memory[i])

        return filtered_memory

    def save_model(self, episode: int) -> None:
        self.model.save('smart_agent_{}.h5'.format(episode))


class BaselineAgent:
    def select_action(self, observation: np.ndarray, actions: np.ndarray):
        """The minimizer agent always selects the baseline temperature."""
        return actions[-1]


class MinimizerAgent:
    def select_action(self, observation: np.ndarray, actions: np.ndarray):
        """The minimizer agent always selects the lowest possible temperature."""
        return actions[0]


class RandomAgent:
    def select_action(self, observation: np.ndarray, actions: np.ndarray):
        """The random agent always chooses a random temperature."""
        return random.choice(actions)
