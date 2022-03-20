"""Compact implementation of a SRNN_DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf
import gin.tf

from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib

import network_model


@gin.configurable
class SRNNDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the SRNN_DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=network_model.SRNN_DQNNetwork,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               eval_mode=False,
               use_staging=True,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):

    dqn_agent.DQNAgent.__init__(
        self,
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
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
        eval_mode=eval_mode,
        use_staging=use_staging,
        max_tf_checkpoints_to_keep=max_tf_checkpoints_to_keep,
        optimizer=optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency,
        allow_partial_reload=allow_partial_reload)

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_net_features: The replayed states' features.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs, _ = self.online_convnet(self.state_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs, self._replay_net_features = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs, _ = self.target_convnet(
        self._replay.next_states)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        reduction_indices=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)

    # loss of Sparsity
    beta = 0.1
    beta_hat = tf.reduce_mean(self._replay_net_features, axis=1)
    l_rec = (1 / (beta_hat + 0.0001)) - beta / (tf.square(beta_hat) + 0.0001)
    l_rec = l_rec * tf.cast(beta_hat > beta, dtype=tf.float32)
    loss_sparse = l_rec * beta_hat

    loss = loss + 0.1 * loss_sparse

    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))





