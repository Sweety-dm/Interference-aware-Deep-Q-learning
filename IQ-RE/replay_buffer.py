from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import gin.tf

from dopamine.replay_memory import circular_replay_buffer


ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

@gin.configurable
class InheritedOutOfGraphReplayBuffer(circular_replay_buffer.OutOfGraphReplayBuffer):
  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32,
               state_label_shape=(),
               state_label_dtype=np.int32):

    self._state_label_shape = state_label_shape
    self._state_label_dtype = state_label_dtype

    circular_replay_buffer.OutOfGraphReplayBuffer.__init__(
        self,
        observation_shape=observation_shape,
        stack_size=stack_size,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype,
        terminal_dtype=terminal_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype)

  def get_storage_signature(self):
    storage_elements = [
        ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
        ReplayElement('state_label', self._state_label_shape, self._state_label_dtype),
        ReplayElement('action', self._action_shape, self._action_dtype),
        ReplayElement('reward', self._reward_shape, self._reward_dtype),
        ReplayElement('terminal', (), self._terminal_dtype)
    ]
    for extra_replay_element in self._extra_storage_types:
      storage_elements.append(extra_replay_element)
    return storage_elements

  def add(self, observation, state_label, action, reward, terminal, *args):
    self._check_add_types(observation, state_label, action, reward, terminal, *args)
    if self.is_empty() or self._store['terminal'][self.cursor() - 1] == 1:
      for _ in range(self._stack_size - 1):
        # Child classes can rely on the padding transitions being filled with
        # zeros. This is useful when there is a priority argument.
        self._add_zero_transition()
    self._add(observation, state_label, action, reward, terminal, *args)

  def sample_transition_batch(self, batch_size=None, indices=None):
    if batch_size is None:
      batch_size = self._batch_size
    if indices is None:
      indices = self.sample_index_batch(batch_size)
    assert len(indices) == batch_size

    transition_elements = self.get_transition_elements(batch_size)
    batch_arrays = self._create_batch_arrays(batch_size)
    for batch_element, state_index in enumerate(indices):
      trajectory_indices = [(state_index + j) % self._replay_capacity
                            for j in range(self._update_horizon)]
      trajectory_terminals = self._store['terminal'][trajectory_indices]
      is_terminal_transition = trajectory_terminals.any()
      if not is_terminal_transition:
        trajectory_length = self._update_horizon
      else:
        # np.argmax of a bool array returns the index of the first True.
        trajectory_length = np.argmax(trajectory_terminals.astype(np.bool), 0) + 1
      next_state_index = state_index + trajectory_length
      trajectory_discount_vector = (self._cumulative_discount_vector[:trajectory_length])
      trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                          next_state_index)

      # Fill the contents of each array in the sampled batch.
      assert len(transition_elements) == len(batch_arrays)
      for element_array, element in zip(batch_arrays, transition_elements):
        if element.name == 'state':
          element_array[batch_element] = self.get_observation_stack(state_index)
        elif element.name == 'reward':
          # compute the discounted sum of rewards in the trajectory.
          element_array[batch_element] = np.sum(
              trajectory_discount_vector * trajectory_rewards, axis=0)
        elif element.name == 'next_state':
          element_array[batch_element] = self.get_observation_stack(
              (next_state_index) % self._replay_capacity)
        elif element.name in ('next_state_label', 'next_action', 'next_reward'):
          element_array[batch_element] = (
              self._store[element.name.lstrip('next_')][(next_state_index) %
                                                        self._replay_capacity])
        elif element.name == 'terminal':
          element_array[batch_element] = is_terminal_transition
        elif element.name == 'indices':
          element_array[batch_element] = state_index
        elif element.name in self._store.keys():
          element_array[batch_element] = (
              self._store[element.name][state_index])
        # We assume the other elements are filled in by the subclass.

    return batch_arrays

  def get_transition_elements(self, batch_size=None):
    """Returns a 'type signature' for sample_transition_batch.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
    Returns:
      signature: A namedtuple describing the method's return type signature.
    """
    batch_size = self._batch_size if batch_size is None else batch_size

    transition_elements = [
        ReplayElement('state', (batch_size,) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('state_label', (batch_size,) + self._state_label_shape,
                      self._state_label_dtype),
        ReplayElement('action', (batch_size,) + self._action_shape,
                      self._action_dtype),
        ReplayElement('reward', (batch_size,) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('next_state', (batch_size,) + self._state_shape,
                      self._observation_dtype),
        ReplayElement('next_state_label', (batch_size,) + self._state_label_shape,
                      self._state_label_dtype),
        ReplayElement('next_action', (batch_size,) + self._action_shape,
                      self._action_dtype),
        ReplayElement('next_reward', (batch_size,) + self._reward_shape,
                      self._reward_dtype),
        ReplayElement('terminal', (batch_size,), self._terminal_dtype),
        ReplayElement('indices', (batch_size,), np.int32)
    ]
    for element in self._extra_storage_types:
      transition_elements.append(
          ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                        element.type))
    return transition_elements


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class InheritedWrappedReplayBuffer(circular_replay_buffer.WrappedReplayBuffer):
  """Wrapper of InheritedOutOfGraphReplayBuffer with an in graph sampling mechanism.

  Usage:
    To add a transition:  call the add function.

    To sample a batch:    Construct operations that depend on any of the
                          tensors is the transition dictionary. Every sess.run
                          that requires any of these tensors will sample a new
                          transition.
  """

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32,
               state_label_shape=(),
               state_label_dtype=np.int32):

    """Initializes WrappedReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      wrapped_memory: The 'inner' memory data structure. If None,
        it creates the standard DQN replay memory.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.
      state_label_shape: tuple of ints, the shape of the context label vector. Empty tuple
        means the context label is a scalar.
      state_label_dtype: np.dtype, type of elements in the label.

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
    if replay_capacity < update_horizon + 1:
      raise ValueError(
          'Update horizon ({}) should be significantly smaller '
          'than replay capacity ({}).'.format(update_horizon, replay_capacity))
    if not update_horizon >= 1:
      raise ValueError('Update horizon must be positive.')
    if not 0.0 <= gamma <= 1.0:
      raise ValueError('Discount factor (gamma) must be in [0, 1].')

    self.batch_size = batch_size

    # Mainly used to allow subclasses to pass self.memory.
    if wrapped_memory is not None:
      self.memory = wrapped_memory
    else:
      self.memory = InheritedOutOfGraphReplayBuffer(
          observation_shape,
          stack_size,
          replay_capacity,
          batch_size,
          update_horizon,
          gamma,
          max_sample_attempts,
          observation_dtype=observation_dtype,
          terminal_dtype=terminal_dtype,
          extra_storage_types=extra_storage_types,
          action_shape=action_shape,
          action_dtype=action_dtype,
          reward_shape=reward_shape,
          reward_dtype=reward_dtype,
          state_label_shape=state_label_shape,
          state_label_dtype=state_label_dtype)

    self.create_sampling_ops(use_staging)

  def add(self, observation, state_label, action, reward, terminal, *args):
    """Adds a transition to the replay memory.

    Since the next_observation in the transition will be the observation added
    next there is no need to pass it.

    If the replay memory is at capacity the oldest transition will be discarded.

    Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: np.dtype, acts as a boolean indicating whether the transition
                was terminal (1) or not (0).
      *args: extra contents with shapes and dtypes according to
        extra_storage_types.
    """
    self.memory.add(observation, state_label, action, reward, terminal, *args)

  def unpack_transition(self, transition_tensors, transition_type):
    """Unpacks the given transition into member variables.

    Args:
      transition_tensors: tuple of tf.Tensors.
      transition_type: tuple of ReplayElements matching transition_tensors.
    """
    self.transition = collections.OrderedDict()
    for element, element_type in zip(transition_tensors, transition_type):
      self.transition[element_type.name] = element

    # TODO(bellemare): These are legacy and should probably be removed in
    # future versions.
    self.states = self.transition['state']
    self.state_labels = self.transition['state_label']
    self.actions = self.transition['action']
    self.rewards = self.transition['reward']
    self.next_states = self.transition['next_state']
    self.next_state_labels = self.transition['next_state_label']
    self.next_actions = self.transition['next_action']
    self.next_rewards = self.transition['next_reward']
    self.terminals = self.transition['terminal']
    self.indices = self.transition['indices']


