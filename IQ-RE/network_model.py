"""
## Networks
We are subclassing keras.models.Model in our network definitions. Each network
class has two main functions: `.__init__` and `.call`. When we create our
network the `__init__` function is called and necessary layers are defined. Once
we create our network, we can create the output operations by doing `call`s to
our network with different inputs. At each call, the same parameters will be
used.

## Network Types
Network types are namedtuples that define the output signature of the networks
used. Please use the appropriate signature as needed when defining new networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow.compat.v1 as tf

from dopamine.discrete_domains import atari_lib


DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = collections.namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])

@gin.configurable(whitelist=['num_contexts'])
class IQ_DQNNetwork(atari_lib.NatureDQNNetwork):
  """
  Multi-head Network.
  The convolutional network used to compute the agent's Q-values."""

  def __init__(self,
               num_contexts,
               num_actions,
               name=None):
    self.num_contexts = num_contexts

    atari_lib.NatureDQNNetwork.__init__(self,
        num_actions=num_actions,
        name=name)
    self.dense = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu,
                                       name='fully_connected')

  def single_head_output_layer(self, embedding):
    x = self.dense(embedding)
    x = self.dense2(x)
    return tf.stack([x], axis=1)

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    embedding = self.flatten(x)
    embedding = self.dense1(embedding)

    q_outs = self.single_head_output_layer(embedding)
    if self.num_contexts > 1:
      for context in range(1, self.num_contexts):
        q_out = self.single_head_output_layer(embedding)
        q_outs = tf.concat([q_outs, q_out], axis=1)

    return DQNNetworkType(q_outs)


class IQ_RainbowNetwork(atari_lib.RainbowNetwork):
  """Multi-head Network.
  The convolutional network used to compute agent's return distributions."""

  def __init__(self,
               num_contexts,
               num_actions,
               num_atoms,
               support,
               name=None):
    self.num_contexts = num_contexts

    atari_lib.RainbowNetwork.__init__(
        self,
        num_actions=num_actions,
        num_atoms=num_atoms,
        support=support,
        name=name)

    self.dense = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu,
                                       name='fully_connected')

  def single_head_output_layer(self, embedding):
    x = self.dense(embedding)
    x = self.dense2(x)
    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return tf.stack([q_values], axis=1), \
           tf.stack([logits], axis=1), \
           tf.stack([probabilities], axis=1)

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    embedding = self.flatten(x)
    embedding = self.dense1(embedding)

    q_values, logits,  probabilities = self.single_head_output_layer(embedding)
    if self.num_contexts > 1:
      for context in range(1, self.num_contexts):
        q_value, logit, probabilitie = self.single_head_output_layer(embedding)
        q_values = tf.concat([q_values, q_value], axis=1)
        logits = tf.concat([logits, logit], axis=1)
        probabilities = tf.concat([probabilities, probabilitie], axis=1)

    return RainbowNetworkType(q_values, logits, probabilities)


class SRNN_DQNNetwork(atari_lib.NatureDQNNetwork):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self,
               num_actions,
               name=None):

    atari_lib.NatureDQNNetwork.__init__(
        self,
        num_actions=num_actions,
        name=name)

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    feature = self.dense1(x)

    return DQNNetworkType(self.dense2(feature)), feature


class SRNN_RainbowNetwork(atari_lib.RainbowNetwork):
  """The convolutional network used to compute agent's return distributions."""

  def __init__(self,
               num_actions,
               num_atoms,
               support,
               name=None):

    atari_lib.RainbowNetwork.__init__(
        self,
        num_actions=num_actions,
        num_atoms=num_atoms,
        support=support,
        name=name)

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
      output the feature of representation layer.
    """
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    feature = self.dense1(x)

    out = self.dense2(feature)
    logits = tf.reshape(out, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities), feature
