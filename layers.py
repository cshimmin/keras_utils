import numpy as np
import tensorflow as tf

from keras import layers

class AngleQuadrature(layers.Layer):
    def __init__(self, idxs, axis=-1, **kwargs):
        super(AngleQuadrature, self).__init__(**kwargs)
        
        if type(idxs) is int:
            idxs = [idxs]
        self.idxs = idxs
        self.axis = axis
        
    def call(self, inputs, training=None):
        features_in = tf.split(inputs, inputs.shape[self.axis], axis=self.axis)
        features_out = []
        for i,f in enumerate(features_in):
            if i in self.idxs:
                features_out.append(tf.sin(f))
                features_out.append(tf.cos(f))
            else:
                features_out.append(f)
        return tf.concat(features_out, axis=self.axis)
    
    def compute_output_shape(self, input_shape):
        output_shape = np.array(input_shape)
        output_shape[self.axis] += len(self.idxs)
        return tuple(output_shape)

# Layer to add a random angular offset per batch entry,
# for a specific index or list of indices along the given axis
class RandomizeAngle(layers.Layer):
    def __init__(self, idxs, axis=-1, train_only=True, **kwargs):
        super(RandomizeAngle, self).__init__(**kwargs)
        if type(idxs) is int:
            idxs = [idxs]
        self.idxs = idxs
        self.axis = axis
        self.train_only = train_only
    
    def call(self, inputs, training=None):
        features_in = tf.split(inputs, inputs.shape[self.axis], axis=self.axis)
        phi_offsets = tf.random_uniform((tf.shape(inputs)[0],) + (1,)*(len(inputs.shape)-1), -np.pi, np.pi)
        features_noised = []
        for i,f in enumerate(features_in):
            if i in self.idxs:
                features_noised.append(f + phi_offsets)
            else:
                features_noised.append(f)
        noised = tf.concat(features_noised, axis=self.axis)
        if self.train_only:
            return K.in_train_phase(noised, inputs, training=training)
        else:
            return noised
    
    def compute_output_shape(self, input_shape):
        return input_shape
