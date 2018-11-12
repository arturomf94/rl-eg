# @title Necessary imports and globals.
#matrix math
import numpy as np
#load files
import os
#DQN for baselines
from dopamine.agents.dqn import dqn_agent
#high level agent-environment excecution engine
from dopamine.atari import run_experiment
#visualization + data downloading
from dopamine.colab import utils as colab_utils
#warnings
from absl import flags

import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#where to store training logs
BASE_PATH = '/tmp/colab_dope_run'  # @param
#which arcade environment?
GAME = 'Asterix'  # @param

# @title Create a  new agent from scratch.

#define where to store log data
LOG_PATH = os.path.join(BASE_PATH, 'basic_agent', GAME)

class BasicAgent(object):
  """This agent randomly selects an action and sticks to it. It will change
  actions with probability switch_prob."""
  def __init__(self, sess, num_actions, switch_prob=0.1):
    #tensorflow session
    self._sess = sess
    #how many possible actions can it take?
    self._num_actions = num_actions
    # probability of switching actions in the next timestep?
    self._switch_prob = switch_prob
    #initialize the action to take (randomly)
    self._last_action = np.random.randint(num_actions)
    #not debugging
    self.eval_mode = False

  #How select an action?
  #we define our policy here
  def _choose_action(self):
    if np.random.random() <= self._switch_prob:
      self._last_action = np.random.randint(self._num_actions)
    return self._last_action

  #when it checkpoints during training, anything we should do?
  def bundle_and_checkpoint(self, unused_checkpoint_dir, unused_iteration):
    pass

  #loading from checkpoint
  def unbundle(self, unused_checkpoint_dir, unused_checkpoint_version,
               unused_data):
    pass

  #first action to take
  def begin_episode(self, unused_observation):
    return self._choose_action()

  #cleanup
  def end_episode(self, unused_reward):
    pass

  #we can update our policy here
  #using the reward and observation
  #dynamic programming, Q learning, monte carlo methods, etc.
  def step(self, reward, observation):
    return self._choose_action()

def create_basic_agent(sess, environment, summary_writer = None):
  """The Runner class will expect a function of this type to create an agent."""
  return BasicAgent(sess, num_actions=environment.action_space.n,
                     switch_prob=0.2)

# Create the runner class with this agent. We use very small numbers of steps
# to terminate quickly, as this is mostly meant for demonstrating how one can
# use the framework. We also explicitly terminate after 110 iterations (instead
# of the standard 200) to demonstrate the plotting of partial runs.
basic_runner = run_experiment.Runner(LOG_PATH,
                                      create_basic_agent,
                                      game_name=GAME,
                                      num_iterations=200,
                                      training_steps=10,
                                      evaluation_steps=10,
                                      max_steps_per_episode=100)


# @title Train Basic Agent.
print('Will train basic agent, please be patient, may be a while...')
basic_runner.run_experiment()
print('Done training!')



# @title Load baseline data
content = os.path.join(LOG_PATH, 'content')
os.system('mkdir -p ' + content)
# os.system('mkdir -p content')
os.system("gsutil -q -m cp -r gs://download-dopamine-rl/preprocessed-benchmarks/* " + content)
# os.system("gsutil -q -m cp -R gs://download-dopamine-rl/preprocessed-benchmarks/* /content/")
# !gsutil -q -m cp -R gs://download-dopamine-rl/preprocessed-benchmarks/* /content/
# experimental_data = colab_utils.load_baselines('/content')
experimental_data = colab_utils.load_baselines(content)

# @title Load the training logs.
basic_data = colab_utils.read_experiment(log_path=LOG_PATH, verbose=True)
basic_data['agent'] = 'BasicAgent'
basic_data['run_number'] = 1
# experimental_data[GAME] = experimental_data[GAME].merge(basic_data,
#                                                         how='outer')
experimental_data[GAME] = pd.concat([experimental_data[GAME], basic_data])

# @title Plot training results.

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16,8))
sns.tsplot(data=experimental_data[GAME], time='iteration', unit='run_number',
           condition='agent', value='train_episode_returns', ax=ax)
plt.title(GAME)
plt.show()
