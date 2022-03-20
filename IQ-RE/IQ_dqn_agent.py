"""Compact implementation of a IQ_DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np

import tensorflow.compat.v1 as tf
import gin.tf

from sklearn.cluster import KMeans
from collections import Counter

from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib

import network_model
import random_encoder
import replay_buffer


def current_state_head(q_values, state_label):
    batch_size = tf.shape(q_values)[0]
    indices = tf.concat([tf.reshape(tf.range(batch_size), [batch_size, 1]),
                         tf.reshape(state_label, [batch_size, 1])],
                        axis=1)
    q_current_head = tf.gather_nd(q_values, indices)
    return q_current_head

def head_others(q_values, state_label, num_contexts):
    batch_size = tf.shape(q_values)[0]
    other_labels = tf.ones(shape=[batch_size, num_contexts]) - \
                   tf.one_hot(state_label, num_contexts)

    indices = tf.reshape(tf.where(other_labels), [batch_size, num_contexts - 1, 2])
    q_others = tf.gather_nd(q_values, indices)
    return q_others

def context_decision(num_contexts, state_encode, context_center):
    dist = []
    for i in range(num_contexts):
        dist.append(np.linalg.norm(state_encode - context_center[i, :]))
    obs_label = dist.index(np.min(dist))
    return obs_label

@gin.configurable
class IQDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the IQ_DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=network_model.IQ_DQNNetwork,
               image_encoder=random_encoder.RandomEncoder,
               num_contexts=3,
               feature_dim=50,
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
    self.num_contexts = num_contexts
    self.feature_dim = feature_dim
    self.image_encoder = image_encoder
    self.context_center = np.zeros((num_contexts, feature_dim))
    self.context_center_target = self.context_center
    self.context_states_count = [0] * num_contexts
    self.recent_buffer = np.empty([min_replay_history, feature_dim])

    with tf.device(tf_device):
      self.state_label_ph = tf.placeholder(tf.int32, (),
                                           name='state_label_ph')

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

    with tf.device(tf_device):
      self._build_image_encoder()

    tf.logging.info('\t num_contexts: %d', num_contexts)
    tf.logging.info('\t feature_dim: %d', feature_dim)


  def _create_network(self, name):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(self.num_contexts, self.num_actions, name=name)
    return network

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_net_others: The replayed states' output in other heads.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
      self._replay_target_net_others: The replayed states' target output in other heads.
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')

    self._net_outputs = self.online_convnet(self.state_ph)
    _current_headq_value = current_state_head(self._net_outputs.q_values,
                                              self.state_label_ph)
    self._q_argmax = tf.argmax(_current_headq_value, axis=1)[0]

    self._replay_net_outputs_ = self.online_convnet(self._replay.states)
    self._replay_net_outputs = current_state_head(self._replay_net_outputs_.q_values,
                                                  self._replay.state_labels)
    self._replay_net_others = head_others(self._replay_net_outputs_.q_values,
                                          self._replay.state_labels,
                                          self.num_contexts)

    self._replay_next_target_net_outputs_ = self.target_convnet(self._replay.next_states)
    self._replay_next_target_net_outputs = current_state_head(self._replay_next_target_net_outputs_.q_values,
                                                              self._replay.next_state_labels)

    self._replay_target_net_outputs_ = self.target_convnet(self._replay.states)
    self._replay_target_net_others = head_others(self._replay_target_net_outputs_.q_values,
                                                 self._replay.state_labels,
                                                 self.num_contexts)

  def _build_image_encoder(self):
    self.encoder = self.image_encoder(self.feature_dim, name='Encoder')
    self._encoder_outputs = self.encoder(self.state_ph)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperReplayBuffer object.
    """
    return replay_buffer.InheritedWrappedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs, 1)
    return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs * replay_action_one_hot,
        reduction_indices=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target,
        replay_chosen_q,
        reduction=tf.losses.Reduction.NONE)

    # distillation loss
    if self.num_contexts >= 2:
      replay_action_one_hot_encodes = replay_action_one_hot
      for i in range(self.num_contexts - 2):
        if i == 0:
            replay_action_one_hot_encodes = tf.stack([replay_action_one_hot_encodes,
                                                      replay_action_one_hot],
                                                     axis=1)
        else:
            replay_action_one_hot_encodes = tf.concat([replay_action_one_hot_encodes,
                                                       tf.stack([replay_action_one_hot], axis=1)],
                                                      axis=1)

      replay_chosen_others_q = tf.reduce_sum(
          self._replay_net_others * replay_action_one_hot_encodes,
          reduction_indices=2)
      replay_chosen_others_target_q = tf.reduce_sum(
          self._replay_target_net_others * replay_action_one_hot_encodes,
          reduction_indices=2)
      replay_chosen_others_target_q = tf.stop_gradient(replay_chosen_others_target_q)
      distillation_loss = tf.losses.huber_loss(
          replay_chosen_others_target_q,
          replay_chosen_others_q,
          reduction=tf.losses.Reduction.NONE)

      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
      forgetting_inhibition_rate = 1 - epsilon

      loss = loss + forgetting_inhibition_rate * distillation_loss

    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._last_state_label = self.state_label
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self._last_state_label, self.action, reward, False)
      self._train_step()

    self.action = self._select_action()
    return self.action

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(self._observation, self.state_label, self.action, reward, True)

  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # Choose the action with highest Q-value at the current state.
      return self._sess.run(self._q_argmax, {self.state_ph: self.state,
                                             self.state_label_ph: self.state_label})

  def _record_observation(self, observation):
    self._observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.state = np.roll(self.state, -1, axis=-1)
    self.state[0, ..., -1] = self._observation

    if self.num_contexts > 1:
      self.state_encode = self._sess.run(self._encoder_outputs,
                                         {self.state_ph: self.state})[0]

      # if self.training_steps % self.target_update_period == 0:
      #   self.context_center_target = self.context_center
      self.state_label = context_decision(self.num_contexts,
                                          self.state_encode,
                                          self.context_center)
      if not self.eval_mode:
        if self.training_steps < self.min_replay_history:
          self.recent_buffer[self.training_steps, :] = self.state_encode
        elif self.training_steps == self.min_replay_history:
          kmeans = KMeans(n_clusters=self.num_contexts, random_state=0).fit(self.recent_buffer)
          context_states_count_dict = Counter(kmeans.labels_)
          self.context_states_count = [context_states_count_dict[i] for i in range(self.num_contexts)]
          self.context_center = kmeans.cluster_centers_
        else:
          self.context_states_count[self.state_label] += 1
          self.context_center[self.state_label, :] += (1 / self.context_states_count[self.state_label]) * (
                  self.state_encode - self.context_center[self.state_label, :])
          # self.context_center[self.state_label, :] += 0.01 * (
          #             self.state_encode - self.context_center[self.state_label, :])
    else:
      self.state_encode = self.state
      self.state_label = 0

  def _store_transition(self, last_observation, last_label, action, reward, is_terminal):
    """Stores an experienced transition.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer:
      (last_observation, action, reward, is_terminal).

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      last_label: int, the context that last_observation belongs.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    self._replay.add(last_observation, last_label, action, reward, is_terminal)

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)

    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    bundle_dictionary['training_steps'] = self.training_steps
    bundle_dictionary['context_center'] = self.context_center
    return bundle_dictionary





