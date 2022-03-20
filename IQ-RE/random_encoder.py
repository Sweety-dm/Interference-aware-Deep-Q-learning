from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from dopamine.discrete_domains import atari_lib


class RandomEncoder(atari_lib.NatureDQNNetwork):
  def __init__(self,
               feature_dim,
               name=None):
    atari_lib.NatureDQNNetwork.__init__(
        self,
        num_actions=feature_dim,
        name=name)

  def call(self, state):
    x = tf.cast(state, tf.float32)
    x = tf.div(x, 255.)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    embedding = self.flatten(x)

    return self.dense2(embedding)


