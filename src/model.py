import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from collections import deque
import random

from environment import Game2048, Movement


class DQNModel(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize the DQN model.

        :param state_size: The size of the input state vector.
        :param action_size: The number of possible actions.
        """
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        :param x: Input tensor (state).
        :return: Output tensor (Q-values for each action).
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = torch.relu(self.fc3(x))
        # x = torch.softmax(self.fc4(x), dim=-1)
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize the DQN agent.

        :param state_size: The size of the input state vector.
        :param action_size: The number of possible actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state: np.ndarray, action: int, reward: int, next_state: np.ndarray, done: bool) -> None:
        """
        Store an experience in the replay memory.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The state after taking the action.
        :param done: Whether the episode is finished.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """
        Select an action based on the current policy.

        :param state: The current state.
        :return: The selected action.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size: int) -> None:
        """
        Train the model on a batch of experiences.

        :param batch_size: The number of experiences to sample from memory for training.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state)[0]).item())
            target_f = self.model(state).clone().detach()
            target_f[0][action] = target
            output = self.model(state)
            loss = self.criterion(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str) -> None:
        """
        Load the model weights from a file.

        :param name: The file name to load the model weights from.
        """
        self.model.load_state_dict(torch.load(name))

    def save(self, name: str) -> None:
        """
        Save the model weights to a file.

        :param name: The file name to save the model weights to.
        """
        torch.save(self.model.state_dict(), name)
        
if __name__ == "__main__":
    env = Game2048()
    movement = Movement(env)
    state_size = env.size * env.size
    action_size = 4  # Left, Right, Up, Down
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action, movement)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"episode: {e}/{episodes}, score: {env.score}, e: {agent.epsilon:.2}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
               
    env.render()

        # if e % 50 == 0:
        #     agent.save(f"2048-dqn-{e}.pth")
