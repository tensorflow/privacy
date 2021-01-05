import tensorflow as tf


def make_dp_model_class(cls):
  class DPModelClass(cls):
    def __init__(self, l2_norm_clip, noise_multiplier, use_xla=True, *args, **kwargs):
        super(DPModelClass, self).__init__(*args, **kwargs)
        self._l2_norm_clip = l2_norm_clip
        self._noise_multiplier = noise_multiplier

        if use_xla:
            self.train_step = tf.function(
                self.train_step, experimental_compile=True)

    def process_per_example_grads(self, grads):
        grads_flat = tf.nest.flatten(grads)
        squared_l2_norms = [tf.reduce_sum(
            input_tensor=tf.square(g)) for g in grads_flat]
        global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
        div = tf.maximum(global_norm / self._l2_norm_clip, 1.)
        clipped_flat = [g / div for g in grads_flat]
        return tf.nest.pack_sequence_as(grads, clipped_flat)

    def reduce_per_example_grads(self, stacked_grads):
        summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
        noise_stddev = self._l2_norm_clip * self._noise_multiplier
        noise = tf.random.normal(
            tf.shape(input=summed_grads), stddev=noise_stddev)
        noised_grads = summed_grads + noise
        return noised_grads / tf.cast(stacked_grads.shape[0], tf.float32)
