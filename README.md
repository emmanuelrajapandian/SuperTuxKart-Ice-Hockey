## RL-DL agent to play SuperTuxKart Ice Hockey

We design an automated agent to play SuperTuxKart Ice Hockey which is a game featuring a vast state space, a diverse action space, and sparse rewards, presenting a highly formidable challenge. The objective of the agent is to maximize goal scoring in any 
difficulty and achieve victory in the match if possible. Our approach involves using imitation learning, combining both Behavioural Cloning and DAgger, to mimic other agents and learn the optimal strategy for playing the game. We employ REINFORCE on top of 
our best imitation agent to adjust the variables of the agent's policy in an approach that increases the likelihood of actions that result in higher rewards i.e., an effective goal-scoring strategy. Our system is designed to exploit potential simplifications 
in this complex environment, with the ultimate aim of creating proficient players. We train a neural net model, inspired by the principles of imitation learning, to support a controller network to play ice hockey. Our system is state-based, focusing on the 
state of the game rather than the visual input from the player’s field of view.


### Gameplay 
Our team, represented by the blue players, faces off against the formidable Jurgen agent, represented by the red players, known as the strongest contender in the pool. Our agent demonstrates commendable performance both in defense and attack. This accomplishment was made possible through the integration of REINFORCE, layered atop imitation learning from the top-performing agent.

<p align="center">
  <img src="https://github.com/emmanuelrajapandian/SuperTuxKart-Ice-Hockey/blob/main/tournament-run.gif" alt="animated" />
</p>
