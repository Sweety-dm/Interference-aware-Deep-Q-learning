# coding=utf-8
# Copyright 2020 The Intelligent Computing Lab, THU

"""Launcher for training original agent.

This file trains the initial (DQN / Rainbow) agent and
calculated interferences during RL model training.
"""
import os
from absl import app
from absl import flags

import tensorflow as tf
import custom_run_experiment


# flags: Helps to dynamically change parameters in code from the command line
FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_atari_environment.game_name="Pong"").')

def main(unused_argv):
  """This main function acts as a wrapper around a gin-configurable experiment.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.logging.set_verbosity(tf.logging.INFO)  # DEBUG < INFO < WARN < ERROR < FATAL
  custom_run_experiment.run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = custom_run_experiment.create_runner_checkpoint(FLAGS.base_dir)
  runner.run_experiment()

if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
