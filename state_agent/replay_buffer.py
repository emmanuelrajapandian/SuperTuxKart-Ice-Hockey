import random
from collections import deque
import torch

from state_agent.player import extract_features

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        reward = torch.tensor(reward)
        action = torch.stack(action)
        done = torch.tensor(done)

        return state, action, reward[:, None], next_state, done[:, None]

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    from state_agent.environment import GameEnvironment

    replay_buffer = ReplayBuffer(capacity=1000)

    game_environment = GameEnvironment(
        first_team="image_jurgen_agent",
        second_team="image_jurgen_agent",
        num_frames=100,
        max_score=3,
        num_players=1
    )

    state = game_environment.setup()

    for i in range(10):

      next_state, reward, done = game_environment.step()

      replay_buffer.push(state, torch.Tensor([0.5, -.3, 0.4]), reward, next_state, done)

    state, action, reward, next_state, done = replay_buffer.sample(2)

    print("State:")
    print(state.shape)

    print("Action:")
    print(action.shape)

    print("Reward:")
    print(reward.shape)

    print("Next State:")
    print(next_state.shape)
      