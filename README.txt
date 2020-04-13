All code is contained in Jupyter notebooks frozen_lake.ipynb and tower_of_hanoi.ipynb.

Their names correspond to the MDP they are used for.

Code can be found here: https://github.com/luclement/ml-assignment4

MDPs:

1. Frozen Lake is an OpenAI Gym environment which can be found here: https://gym.openai.com/envs/FrozenLake-v0/
  a. Source adapted from: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
2. Tower of Hanoi is a custom OpenAI Gym environment
  a. Source adapted from: https://github.com/AlexMatthers/toh-gym
  b. Original source: https://github.com/xadahiya/toh-gym

Install:

1. Run "pip install -r requirements.txt" to install required dependencies
2. Make sure you have Jupyter installed by running "jupyter --version" and you should see jupyter-notebook installed
3. To start the notebook, run: jupyter notebook
4. Navigate to desired notebook and run from the top down

Code references:
1. Piazza post on converting gym env.P to transition and reward matrices for MDPToolbox: https://piazza.com/class/k51r1vdohil5g3?cid=709 (see comments seciton)
2. Code for visualizing grid world: https://github.com/wesley-smith/CS7641-assignment-4/blob/f3d86e37504dda563f65b3267610a30f09d01c77/helpers.py

Packages used:
1. OpenAI Gym: https://gym.openai.com/
2. MDPToolbox: https://pymdptoolbox.readthedocs.io/en/latest/api/mdptoolbox.html
3. MDPToolbox Hiive Fork: https://github.com/hiive/hiivemdptoolbox
5. jupyter: https://jupyter.org/
6. numpy: https://numpy.org/
7. matplotlib: https://matplotlib.org/
