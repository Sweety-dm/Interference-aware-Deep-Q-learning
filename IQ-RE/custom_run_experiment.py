""" Inherited and customized TrainRunner class from dopamine &
Create custom agent."""


from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import iteration_statistics
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent

import tensorflow as tf
import gin.tf
import os
import numpy as np

import IQ_rainbow_agent
import IQ_dqn_agent
import SRNN_dqn_agent
import SRNN_rainbow_agent


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None,
                        debug_mode=False):
  """Creates an agent.

    Args:
      sess: A `tf.compat.v1.Session` object for running associated ops.
      environment: A gym environment (e.g. Atari 2600).
      agent_name: str, name of the agent to create.
      summary_writer: A Tensorflow summary writer to pass to the agent
        for in-agent training statistics in Tensorboard.
      debug_mode: bool, whether to output Tensorboard summaries. If set to true,
        the agent will output in-episode statistics to Tensorboard. Disabled by
        default as this results in slower training.

    Returns:
      agent: An RL agent.

    Raises:
      ValueError: If `agent_name` is not in supported list.
    """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess,
                              num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(sess,
                                      num_actions=environment.action_space.n,
                                      summary_writer=summary_writer)
  elif agent_name == 'IQ_dqn':
    return IQ_dqn_agent.IQDQNAgent(sess,
                                         num_actions=environment.action_space.n,
                                         summary_writer=summary_writer)
  elif agent_name == 'IQ_rainbow':
    return IQ_rainbow_agent.IQRainbowAgent(sess,
                                                 num_actions=environment.action_space.n,
                                                 summary_writer=summary_writer)
  elif agent_name == 'SRNN_dqn':
    return SRNN_dqn_agent.SRNNDQNAgent(sess,
                                       num_actions=environment.action_space.n,
                                       summary_writer=summary_writer)
  elif agent_name == 'SRNN_rainbow':
    return SRNN_rainbow_agent.SRNNRainbowAgent(sess,
                                               num_actions=environment.action_space.n,
                                               summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))

@gin.configurable
def create_runner_checkpoint(base_dir):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    create_agent: function, function used to create the agent.

  Returns:
    runner: A `Runner` like object.

  """
  assert base_dir is not None
  return BuildRunner(base_dir, create_agent)


@gin.configurable
class BuildRunner(run_experiment.TrainRunner):
  """The SaveRunner which adds saving functionality.
  """

  def __init__(self,
               base_dir,
               create_agent_fn):
    """Initialize the SaveRunner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
    """
    tf.logging.info('Creating SaveRunner ...')
    super(BuildRunner, self).__init__(base_dir, create_agent_fn)

    self.training_performance_iteration = []

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    num_episodes_train, average_reward_train = self._run_train_phase(
        statistics)
    self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train)

    self.training_performance_iteration.append(average_reward_train)
    np.save(os.path.join(self._base_dir, 'training_performance.npy'), np.array(self.training_performance_iteration))

    # record context centers
    # context_centers = self._agent.context_center
    # np.save(os.path.join(self._base_dir, 'context_centers.npy'), np.array(context_centers))

    return statistics.data_lists

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      if iteration == 199:
          self._log_experiment(iteration, statistics)
          self._checkpoint_experiment(iteration)


