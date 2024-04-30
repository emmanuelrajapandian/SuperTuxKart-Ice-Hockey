import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from state_agent.model import Actor
from state_agent.player import extract_features
import os
import shutil
import logging
from pathlib import Path
from jurgen_agent.player import extract_featuresV2
import random
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool


logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def reset_directories():
  imitation_recordings_dir = "./state_agent/imitation_recordings"
  shutil.rmtree(imitation_recordings_dir, ignore_errors=True)
  os.mkdir(imitation_recordings_dir)
  return imitation_recordings_dir


# Define a function to execute a command
def execute_command(cmd):
  os.system(cmd)


def collect_data(player, games_to_play, dirs, logging):
  # TODO: consider making this a parallel computation. Maybe threadhing would be enough

  if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
     hide_print = ''
     print_command_to_console = True
  else:
     hide_print = '>/dev/null 2>&1'
     print_command_to_console = False

  if games_to_play > 1:
     games_to_play = '-j ' + str(games_to_play)
  else:
     games_to_play =  ''

  commands = [
     f"python -m tournament.runner {player}  yoshua_agent  -s {dirs}/state_p_yo_player1.pkl -p 2 {games_to_play} {hide_print}",
     f"python -m tournament.runner yoshua_agent {player} -s {dirs}/state_yo_p_player2.pkl -p 2 {games_to_play} {hide_print}",

     f"python -m tournament.runner {player}  geoffrey_agent  -s {dirs}/state_p_g_player1.pkl -p 2 {games_to_play} {hide_print}",
     f"python -m tournament.runner geoffrey_agent {player} -s {dirs}/state_g_p_player2.pkl -p 2 {games_to_play} {hide_print}",

     f"python -m tournament.runner {player}  yann_agent  -s {dirs}/state_p_y_player1.pkl -p 2 {games_to_play} {hide_print}",
     f"python -m tournament.runner yann_agent {player} -s {dirs}/state_y_p_player2.pkl -p 2 {games_to_play} {hide_print}",


     f"python -m tournament.runner jurgen_agent  yoshua_agent  -s {dirs}/state_j_yo_player1.pkl -p 2 {games_to_play} {hide_print}",
     f"python -m tournament.runner yoshua_agent jurgen_agent -s {dirs}/state_yo_j_player2.pkl -p 2 {games_to_play} {hide_print}",

     f"python -m tournament.runner jurgen_agent  geoffrey_agent  -s {dirs}/state_j_g_player1.pkl -p 2 {games_to_play} {hide_print}",
     f"python -m tournament.runner geoffrey_agent jurgen_agent -s {dirs}/state_g_j_player2.pkl -p 2 {games_to_play} {hide_print}",

     f"python -m tournament.runner jurgen_agent  yann_agent  -s {dirs}/state_j_y_player1.pkl -p 2 {games_to_play} {hide_print}",
     f"python -m tournament.runner yann_agent jurgen_agent -s {dirs}/state_y_j_player2.pkl -p 2 {games_to_play} {hide_print}",
  ]
  
  # Determine the number of processes to use
  num_processes = 4

  # Use a process pool to execute commands in parallel
  with Pool(processes=num_processes) as pool:
      pool.map(execute_command, commands)

   

def load_recording(path):
    import pickle
    objects = []
    if 'player1' in str(path):
       team_id = 0
       player_state = 'team1_state'
       opponent_state = 'team2_state'
    else:
       team_id = 1
       player_state = 'team2_state'
       opponent_state = 'team1_state'
    
    with (open(path, "rb")) as openfile:
        while True:
            try:
                o = pickle.load(openfile)

                objects.append(dict(
                  player_state = o[player_state][0],
                  opponent_state = o[opponent_state],
                  soccer_state = o['soccer_state'],
                  team_id = team_id
                ))

                objects.append(dict(
                  player_state = o[player_state][1],
                  opponent_state = o[opponent_state],
                  soccer_state = o['soccer_state'],
                  team_id = team_id
                ))
            except EOFError:
                break
            
    return objects


def predict_jurgen(model, data):
  s = []
  for d in data:
      d = extract_featuresV2(d['player_state'], d['soccer_state'], d['opponent_state'], d['team_id'])
      with torch.no_grad():
        s.append(torch.tensor(model(d)).float())
  data = torch.stack(s)
  return data


def format_training_data(data):
  s = []
  for d in data:
    s.append(extract_features(d['player_state'], d['soccer_state'], d['opponent_state'], d['team_id']))
  return torch.stack(s)


def save_actor(model, path, path2, after_save_device='cpu'):
    model.eval()
    torch.save(model.state_dict(), path2)
    scripted_model = torch.jit.script(model)
    scripted_model.save(path)
    # model.train() 
    model.to(after_save_device)
   


def train():
  games_to_play = 3
  batch_size = 128
  epochs = 100
  loops = 51
  patience = 5
  save_after_loops = 5

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")

  logging.info("Loading jurgen_agent...")
  image_jurgen_agent = torch.jit.load("jurgen_agent/jurgen_agent.pt")
  image_jurgen_agent = image_jurgen_agent
  image_jurgen_agent.eval()

  logging.info("Initializing state agent...")
  agent = Actor()
  save_actor(model=agent, path="state_agent/imitation_model.pt", path2="state_agent/imitation_model_no_jit.pt", after_save_device=device)


  for l in range(loops):
    # Set the agent to evaluation mode so dropout layers are not used
    agent.eval()
    
    dir = reset_directories() 
    dir = "./state_agent/imitation_recordings"
    logging.info("Collecting data...")
    collect_data(player='state_agent', games_to_play=games_to_play, dirs=dir, logging=logging)

    logging.info("Shaping training data...")

    data = []
    dir = Path(dir)
    for pkl in dir.iterdir():
      if pkl.suffix != '.pkl':
          continue
      data.extend(load_recording(pkl))

    # Shuffle training data
    random.shuffle(data)
    
    logging.debug("Getting labels from expert...")
    labels = predict_jurgen(image_jurgen_agent, data).to(device) # this is a tensor

    # get training data in format for network
    data = format_training_data(data)
    data = data.to(device)
    logging.debug(f"Training data shape: {str(data.shape)}")
    logging.debug(f"Labels shape: {str(labels.shape)}")

    # Train network
    logging.info(f"Training model loop {l}...")
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=patience)

    for epoch in range(epochs):  # num_epochs is the number of epochs you want to train for
      loss_values = 0
      for batch_idx, (data, target) in enumerate(dataloader):
        
        loss = F.mse_loss(agent(data), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values = loss.item() + loss_values

      if l % save_after_loops == 0:
        print(f"Loop {l}, Average loss of: {loss_values / batch_idx}, learning_rate: {optimizer.param_groups[0]['lr']}")
      scheduler.step(loss_values)
    
    if l % save_after_loops == 0:
      save_actor(model=agent, path="state_agent/imitation_model.pt", path2="state_agent/imitation_model_no_jit.pt", after_save_device=device)
      save_actor(model=agent, path="state_agent/imitation_model.pt", path2=f"state_agent/imitation_model_no_jit_{l}.pt", after_save_device=device)
      os.system(f"python -m grader state_agent -v")

    
  save_actor(model=agent, path="state_agent/imitation_model.pt", path2="state_agent/imitation_model_no_jit.pt", after_save_device=device)


if __name__ == "__main__":
   train()
