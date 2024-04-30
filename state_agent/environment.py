
import logging
import numpy as np
import importlib
from tournament import utils
from state_agent.player import extract_features
import torch
import os

TRACK_NAME = 'icy_soccer_field'

def to_native(o):
    import pystk
    _type_map = {pystk.Camera.Mode: int,
                 pystk.Attachment.Type: int,
                 pystk.Powerup.Type: int,
                 float: float,
                 int: int,
                 list: list,
                 bool: bool,
                 str: str,
                 memoryview: np.array,
                 property: lambda x: None}

    def _to(v):
        if type(v) in _type_map:
            return _type_map[type(v)](v)
        else:
            return {k: _to(getattr(v, k)) for k in dir(v) if k[0] != '_'}
    return _to(o)

class GameEnvironment():
    """
    Runs the game environment and records the state in record_state file.
    """
    
    def __init__(
        self,
        first_team="my_agent",
        second_team="geoffrey_agent",
        num_players=2,
        num_frames=1200,
        max_score=3,
        initial_ball_location=[0,0],
        initial_ball_velocity=[0,0],
        record=False,
        record_name='video'
    ):
        super(GameEnvironment, self).__init__()
        import pystk
        self._pystk = pystk

        self.RECORD_EVERY_N_EPISODE = 1

        if first_team == 'my_agent':
          self.team_id = 0
          self.enemy_team_id = 1
          self.second_team = importlib.import_module(second_team).Team()
        else:
           self.team_id = 1
           self.enemy_team_id = 0
           self.first_team = importlib.import_module(first_team).Team()

        self.num_players = num_players
        self.num_frames = num_frames
        self.max_score = max_score
        self.initial_ball_location = initial_ball_location
        self.initial_ball_velocity = initial_ball_velocity
        self.record = record
        self.reward_record = []
        self.record_name = record_name

        if self.record:
           self.record_iteration = 0 
        

    def get_game_state(self):
        team1_state = [to_native(p) for p in self.state.players[0::2]]
        team2_state = [to_native(p) for p in self.state.players[1::2]]
        soccer_state = to_native(self.state.soccer)

        return team1_state, team2_state, soccer_state
    

    def get_info(self, team1_state, team2_state, soccer_state, team_id):
       features = []
       if team_id == 1:
         team1_state, team2_state = team2_state, team1_state
        
       for i in range(len(team1_state)):
         features.append(extract_features(team1_state[i], soccer_state, team2_state, team_id))

       return features
    

    def reset(self):
        self.recorder = None
        try:
           self.race
           self.race.stop()
           del self.race
        except Exception:
           'continue'

        if self.team_id == 0:
          t1_cars = ['tux'] * self.num_players
          t2_cars = self.second_team.new_match(self.enemy_team_id, self.num_players)
        elif self.team_id == 1:
          t2_cars = ['tux'] * self.num_players
          t1_cars = self.first_team.new_match(self.enemy_team_id, self.num_players)

        try:
          self._pystk.init(self._pystk.GraphicsConfig.none())
        except ValueError:
           self._pystk.clean()
           self._pystk.init(self._pystk.GraphicsConfig.none())

        RaceConfig = self._pystk.RaceConfig

        self.race_config = RaceConfig(track=TRACK_NAME, mode=RaceConfig.RaceMode.SOCCER, num_kart=2 * self.num_players)
        self.race_config.players.pop()
        for i in range(self.num_players):
            self.race_config.players.append(self._make_config(0, t1_cars[i % len(t1_cars)]))
            self.race_config.players.append(self._make_config(1, t2_cars[i % len(t2_cars)]))

        self.iteration = 0

        self.initial_ball_location = self.initial_ball_location
        self.initial_ball_velocity = self.initial_ball_velocity
        
        self.race = self._pystk.Race(self.race_config)
        self.race.start()
        self.race.step()

        self.state = self._pystk.WorldState()
        self.state.update()
        self.state.set_ball_location((self.initial_ball_location[0], 1, self.initial_ball_location[1]),
                                (self.initial_ball_velocity[0], 0, self.initial_ball_velocity[1]))
        
        self.state.update()

        team1_state, team2_state, soccer_state = self.get_game_state()
        feature_state = self.get_info(team1_state, team2_state, soccer_state, team_id=self.team_id) # torch tensor

        return feature_state
        

    def step(self, actions=None): 
        done = 0
        
        # Transform 3rd action, brake, into binary True or False
        for action in actions:
          if action['brake'] > 0.5:
            action['brake'] = True
          else:
            action['brake'] = False

        team1_state, team2_state, soccer_state = self.get_game_state()

        if self.team_id == 0:
           team1_actions = actions
           team2_actions = self.second_team.act(team2_state, team1_state, soccer_state)
        elif self.team_id == 1:
           team2_actions = actions
           team1_actions = self.first_team.act(team1_state, team2_state, soccer_state)

        # Assemble the actions
        actions = []
        for i in range(self.num_players):
            a1 = team1_actions[i] if team1_actions is not None and i < len(team1_actions) else {}
            a2 = team2_actions[i] if team2_actions is not None and i < len(team2_actions) else {}
            actions.append(a1)
            actions.append(a2)

        result_from_action_bool = self.race.step([self._pystk.Action(**a) for a in actions])
        self.state.update()

        if (not result_from_action_bool and self.num_players) or sum(self.state.soccer.score) >= self.max_score or self.iteration >= self.num_frames:
          self.race.stop()
          del self.race
          print('deleted race object')
          done = 1

        self.iteration = self.iteration + 1

        team1_next_state, team2_next_state, soccer_next_state = self.get_game_state()

        reward = self.reward_fn(current_score=soccer_state['score'], next_score=soccer_next_state['score'],
                               ball_location=soccer_state['ball']['location'], goal_line=soccer_state['goal_line'],
                               team_id=self.team_id) # TODO: make this function support multiple players
        
        self.reward_record.append(reward)
        
        feature_state = self.get_info(team1_next_state, team2_next_state, soccer_next_state, self.team_id) # torch tensor

        return feature_state, reward, done, {}
    
    def stop(self):
       
       del self.first_team
       try:
          self.race
          self.race.stop()
          del self.race
       except Exception as e:
          print('could not stop')
          print(e)


    def _make_config(self, team_id, kart):
        PlayerConfig = self._pystk.PlayerConfig
        controller = PlayerConfig.Controller.PLAYER_CONTROL
        return PlayerConfig(controller=controller, team=team_id, kart=kart)
    

    def reward_fn(self, current_score, next_score, ball_location, goal_line, team_id):
        """For now we keep it simple. 
        This is a combination of:
          1. If agent scores
          1. How close is the ball to oponents goal line
        """
        goal_reward = 0
        goal_penalty = 0
        
        # Reward for scoring goals
        if team_id == 0:
           our_score = current_score[0]
           our_next_score = next_score[0]
           their_score = current_score[1]
           their_next_score = next_score[1]
           our_goal_line = goal_line[1] # where we have to score
           their_goal_line = goal_line[0] # where they have to score
        elif team_id == 1:
           our_score = current_score[1]
           our_next_score = next_score[1]
           their_score = current_score[0]
           their_next_score = next_score[0]
           our_goal_line = goal_line[0] # where we have to score
           their_goal_line = goal_line[1] # where they have to score

        if our_next_score > our_score:  # If we score good
            goal_reward = 100

        if their_next_score > their_score:  # If they score, boo
            goal_penalty = -100

        scoring_reward = 0
        if our_score > their_score:
           scoring_reward = 1
        elif our_score < their_score:
           scoring_reward = -1

        # Calculate proximity to the opponent's goal line
        opponent_goal_center = np.mean(our_goal_line, axis=0) # Assumming second element is the oponent goal line
        distance_to_goal = np.linalg.norm(np.array(ball_location) - np.array(opponent_goal_center))

        # Own goal line
        goal_center = np.mean(their_goal_line, axis=0) # Assumming second element is the oponent goal line
        own_distance_to_goal = np.linalg.norm(np.array(ball_location) - np.array(goal_center))
        
        # Normalize distance to a reasonable range for rewards; adjust as necessary
        max_distance = 64.5 + 64.5 
        proximity_reward = abs(max_distance - distance_to_goal) / max_distance
        proximity_penalty = abs(max_distance - own_distance_to_goal) / max_distance

        # Combine rewards
        total_reward = goal_reward + goal_penalty + 1.5 *  proximity_reward + (-1.5) * proximity_penalty + scoring_reward

        return total_reward


def compute_returns(rewards, gamma=0.9):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


import ray
import gc
# gc.set_debug(gc.DEBUG_LEAK)


@ray.remote
def generate_episode_with_return_parallel(policy, agent_to_play_against, tid, bloc):
    if tid == 1:
      ge = GameEnvironment(first_team=agent_to_play_against, second_team='my_agent', num_players=2, max_score=3, record=False, num_frames=1200, initial_ball_location=bloc)
    elif tid == 0:
      ge = GameEnvironment(first_team='my_agent', second_team=agent_to_play_against, num_players=2, max_score=3, record=False, num_frames=1200, initial_ball_location=bloc)

    
    states, actions, rewards = generate_episode(ge, policy)  # Generate a single episode
    returns = compute_returns(rewards)  # Calculate the returns
    ge.stop()
    ge._pystk.clean()
    del ge._pystk    
    del ge
    gc.collect()
    return states, actions, returns


def generate_episode_with_return(policy, agent_to_play_against, tid, bloc):
   
    if tid == 1:
      ge = GameEnvironment(first_team=agent_to_play_against, second_team='my_agent', num_players=2, max_score=3, record=False, num_frames=1200, initial_ball_location=bloc)
    elif tid == 0:
      ge = GameEnvironment(first_team='my_agent', second_team=agent_to_play_against, num_players=2, max_score=3, record=False, num_frames=1200, initial_ball_location=bloc)

    
    states, actions, rewards = generate_episode(ge, policy)  # Generate a single episode
    returns = compute_returns(rewards)  # Calculate the returns
    ge.stop()
    ge._pystk.clean()
    del ge
    gc.collect()
    return states, actions, returns

# @profile
def generate_episode(env, policy):
    import torch.distributions as dist

    state = env.reset()
    done = False
    actions = []
    states = []
    rewards = []
    while not done:
        actions_to_take = []
        for i, s in enumerate(state):
          with torch.no_grad():
            action = policy(s)
            # Sample continuous actions
            acc = dist.Normal(action[0], .1).sample().clamp(0, 1)
            steer = dist.Normal(action[1], .1).sample().clamp(-1, 1)
            brake = dist.Normal(action[2], 0.1).sample().clamp(0, 1)
            if brake < 0.5:
              brake_act = False
            else:
              brake_act = True

          if i == 0:
            actions.append(action)
            states.append(s)

          actions_to_take.append(dict(acceleration=acc, steer=steer, brake=brake_act))

        next_state, reward, done, _ = env.step(actions_to_take)
        rewards.append(torch.tensor(reward))
        
        state = next_state
    return states, actions, rewards


def save_actor(model, path, path2, after_save_device='cpu'):
    model.eval()
    torch.save(model.state_dict(), path2)
    scripted_model = torch.jit.script(model)
    scripted_model.save(path)
    model.to(after_save_device)




def main():
  import logging
  import ray
  import torch.distributions as dist

  logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

  

  from state_agent.model import Actor

  agent = Actor(state_dim=17)
  agent.load_state_dict(torch.load("state_agent/imitation_model_no_jit.pt"), strict=True)
  agent.eval()

  import torch.optim as optim
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  num_episodes = 100  # Set the total number of episodes to train
  num_episodes_parallel = 10
  n_iterations = 50 # Iterations to run the training loop
  batch_size = 128
  learning_rate = 3e-5
  ball_locations = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],]

  for i_episode in range(num_episodes):
      logging.info(f"Episode {i_episode}")
      logging.info("Generating data...")
      optimizer = optim.Adam(agent.parameters(), lr=learning_rate)  # Set appropriate learning rate

      ray.init()
      episode_ids = [generate_episode_with_return_parallel.remote(agent, opponent_agent, team_id, bloc) for opponent_agent, team_id, bloc in zip(['jurgen_agent'] * 8, [1,1,1,1, 0,0,0,0], [[0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],[0, 1],
            [0, -1],
            [1, 0],
            [-1, 0]
            ])
      ]

      trajectories = ray.get(episode_ids)
      ray.shutdown()


      # print("Unreachable objects:", gc.garbage)
      states = []
      actions = []
      returns = []
      for trajectory in trajectories:
        states_, actions_, returns_ = trajectory
        states.extend(states_)
        actions.extend(actions_)
        returns.extend(returns_)

      agent.to(device)
      states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states]).to(device)
      actions = torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in actions]).to(device)
      returns = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in returns]).to(device)



      returns = (returns - returns.mean()) / (returns.std() + 1e-9)
         
      avg_expected_log_return = []
      logging.info("Training loop...")
      for it in range(n_iterations):
         batch_ids = torch.randint(0, len(states), (batch_size,))
         states_batch = states[batch_ids]
         actions_batch = actions[batch_ids]
         returns_batch = returns[batch_ids]

         optimizer.zero_grad()

         output = agent(states_batch)

         # Normally we use a standard deviation that is bigger so we encourage exploration
         # Set it this low since we just want to fine tune our already trained imitation learning agent
         acc_dist = dist.Normal(output[:, 0], 0.1)
         steer_dist = dist.Normal(output[:, 1], 0.1)
         brake_dist = dist.Normal(output[:, 2], 0.1)
      
         log_prob_acc = acc_dist.log_prob(actions_batch[:, 0]) 
         log_prob_steer = steer_dist.log_prob(actions_batch[:, 1])
         log_prob_brake = brake_dist.log_prob(actions_batch[:, 2])
         expected_log_return = (returns_batch * (log_prob_acc + log_prob_steer + log_prob_brake)).mean()

         (-expected_log_return).backward()
         optimizer.step()

         avg_expected_log_return.append(expected_log_return.item())

      logging.info(f"Average return for episode {i_episode}: {np.mean(avg_expected_log_return)}")
      agent.to('cpu')

      if i_episode % 10 == 0:
        save_actor(agent, f"state_agent/reinforce_trained_model_{i_episode}.pt", f"state_agent/reinforce_trained_model_no_jit_{i_episode}.pt")
        save_actor(agent, f"state_agent/reinforce_trained_model.pt", f"state_agent/reinforce_trained_model.pt")
        os.system("python -m grader state_agent -v")
      
  # ray.shutdown()

if __name__ == "__main__":
   main()