"""Compact implementation of a IQ_Rainbow agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import gin.tf

from dopamine.discrete_domains import atari_lib
from dopamine.agents.dqn.dqn_agent import linearly_decaying_epsilon
from dopamine.agents.rainbow.rainbow_agent import project_distribution

import IQ_dqn_agent
import network_model
import random_encoder
import prioritized_replay_buffer
from IQ_dqn_agent import current_state_head, head_others


@gin.configurable
class IQRainbowAgent(IQ_dqn_agent.IQDQNAgent):
  """A compact implementation of a IQ_Rainbow agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=network_model.IQ_RainbowNetwork,
               image_encoder=random_encoder.RandomEncoder,
               num_contexts=3,
               feature_dim=50,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.
    """
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    self._support = tf.linspace(-vmax, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    # TODO(b/110897128): Make agent optimizer attribute private.
    self.optimizer = optimizer

    IQ_dqn_agent.IQDQNAgent.__init__(
        self,
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        image_encoder=image_encoder,
        num_contexts=num_contexts,
        feature_dim=feature_dim,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)

  def _create_network(self, name):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(self.num_contexts, self.num_actions,
                           self._num_atoms, self._support,
                           name=name)
    return network

  def _build_networks(self):
    super(IQRainbowAgent, self)._build_networks()

    self._replay_net_logits = current_state_head(self._replay_net_outputs_.logits,
                                                 self._replay.state_labels)
    self._replay_net_other_logits = head_others(self._replay_net_outputs_.logits,
                                                self._replay.state_labels,
                                                self.num_contexts)
    self._replay_target_net_other_logits = head_others(self._replay_target_net_outputs_.logits,
                                                       self._replay.state_labels,
                                                       self.num_contexts)

    self._replay_next_target_net_probabilities = current_state_head(
        self._replay_next_target_net_outputs_.probabilities,
        self._replay.next_state_labels)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A `WrappedPrioritizedReplayBuffer` object.

    Raises:
      ValueError: if given an invalid replay scheme.
    """
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).
    return prioritized_replay_buffer.InheritedWrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_distribution(self):
    """Builds the C51 target distribution as per Bellemare et al. (2017).

    First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
    is the support of the next state distribution:

      * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
      * 0 otherwise (duplicated num_atoms times).

    Second, we compute the next-state probabilities, corresponding to the action
    with highest expected value.

    Finally we project the Bellman target (support + probabilities) onto the
    original support.

    Returns:
      target_distribution: tf.tensor, the target distribution from the replay.
    """
    batch_size = self._replay.batch_size

    # size of rewards: batch_size x 1
    rewards = self._replay.rewards[:, None]

    # size of tiled_support: batch_size x num_atoms
    tiled_support = tf.tile(self._support, [batch_size])
    tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

    # size of target_support: batch_size x num_atoms

    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: batch_size x 1
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = gamma_with_terminal[:, None]

    target_support = rewards + gamma_with_terminal * tiled_support

    # size of next_qt_argmax: 1 x batch_size
    next_qt_argmax = tf.argmax(
        self._replay_next_target_net_outputs, axis=1)[:, None]
    batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
    # size of next_qt_argmax: batch_size x 2
    batch_indexed_next_qt_argmax = tf.concat(
        [batch_indices, next_qt_argmax], axis=1)

    # size of next_probabilities: batch_size x num_atoms
    next_probabilities = tf.gather_nd(
        self._replay_next_target_net_probabilities,
        batch_indexed_next_qt_argmax)

    return project_distribution(target_support, next_probabilities,
                                self._support)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    target_distribution = tf.stop_gradient(self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_net_logits,
                                        reshaped_actions)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)

    # distillation loss
    if self.num_contexts >= 2:
      batch_indices = tf.range(tf.shape(self._replay_net_other_logits)[0])[:, None]
      batch_indices = tf.reshape(tf.tile(input=batch_indices,
                                         multiples=[1, self.num_contexts - 1]), [-1, 1])
      head_indices = tf.range(self.num_contexts - 1)[:, None]
      head_indices = tf.reshape(tf.tile(input=head_indices,
                                        multiples=[tf.shape(self._replay_net_other_logits)[0], 1]), [-1, 1])
      distill_indices = tf.concat([batch_indices, head_indices], 1)

      replay_actions = tf.reshape(tf.tile(input=self._replay.actions[:, None],
                                          multiples=[1, self.num_contexts - 1]), [-1, 1])
      distill_reshaped_actions = tf.concat([distill_indices, replay_actions], 1)

      chosen_action_other_logits = tf.gather_nd(self._replay_net_other_logits,
                                                distill_reshaped_actions)
      chosen_target_action_other_logits = tf.gather_nd(self._replay_target_net_other_logits,
                                                       distill_reshaped_actions)

      distillation_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=chosen_target_action_other_logits,
          logits=chosen_action_other_logits)
      distillation_loss = tf.reshape(distillation_loss,
                                     [tf.shape(self._replay_net_other_logits)[0], -1])
      distillation_loss = tf.reduce_sum(distillation_loss, 1)

      # total loss
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
      forgetting_inhibition_rate = 1 - epsilon

      loss = loss + forgetting_inhibition_rate * distillation_loss

    if self._replay_scheme == 'prioritized':
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.sqrt(loss + 1e-10))

      # Weight the loss by the inverse priorities.
      loss = loss_weights * loss
    else:
      update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('CrossEntropyLoss', tf.reduce_mean(loss))
      return self.optimizer.minimize(tf.reduce_mean(loss))

  def _store_transition(self,
                        last_observation,
                        last_label,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    """Stores a transition when in training mode.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer (last_observation, action, reward,
    is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      last_label: int, the context that last_observation belongs.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
    """
    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.memory.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(last_observation, last_label, action, reward, is_terminal, priority)



