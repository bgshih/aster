import tensorflow as tf
import tensorflow_addons as tfa

MODEL_PATH = "tf2_utils/tf2_weights"

class Inferer(tf.keras.Model):
    def __init__(self, combine_forward_and_backward=False):
        """
        Parameters
        ----------
        combine_forward_and_backward: uses a combination of the forward and back predictions if set to True. Only uses
        the forward prediction if set to False. The pre-trained model gives better results when
        combine_forward_and_backward=False

        """

        super(Inferer, self).__init__()
        self.combine_forward_and_backward = combine_forward_and_backward
        tfa.register_all(custom_kernels=False)
        self.model = tf.saved_model.load(MODEL_PATH, tags="serve").signatures[
            "serving_default"
        ]

    def call(self, inputs, training=False, mask=None):
        prediction = self.model(inputs)
        if self.combine_forward_and_backward:
            return self._postprocess_combine(prediction)
        else:
            return prediction["forward_logits"]


    def _postprocess_combine(self, logits: tf.float32):
        """
        Postprocess both the forward and backward logits.

        Parameters
        ----------
        logits: backward and forward logits.

        Returns
        -------
        A padded combination of backward and forward logits.

        """

        forward_logits = logits["forward_logits"]
        backward_logits = logits["backward_logits"]

        combined_logits = self._combine_logits(forward_logits, backward_logits)

        # retrieve the remaining logits of forward
        remaining_logits = forward_logits[:, combined_logits.shape[1] :, :]

        return tf.concat([combined_logits, remaining_logits], axis=1)

    def _combine_logits(self, forward_logits: tf.float32, backward_logits: tf.float32):
        """
        Combine forward and backward logits

        """
        # create masks to filter blank indexes
        forward_mask = ~tf.equal(tf.argmax(forward_logits, axis=2), 1)
        backward_mask = ~tf.equal(tf.argmax(backward_logits, axis=2), 1)

        # filter out blank indexes
        masked_forward = forward_logits[forward_mask]
        masked_backward = backward_logits[backward_mask][::-1]  # reverse it

        # ensure both tensors now have the same shape (requirement of tf.where)
        crop_masked_forward = masked_forward[: masked_backward.shape[0]]
        crop_masked_backward = masked_backward[: masked_forward.shape[0]]

        # get softmax element for each time step
        forward_max = tf.reduce_max(crop_masked_forward, axis=1)
        backward_max = tf.reduce_max(crop_masked_backward, axis=1)

        combined_logits = tf.where(
            tf.expand_dims(forward_max, 1) > tf.expand_dims(backward_max, 1),
            crop_masked_forward,
            crop_masked_backward,
        )

        return tf.expand_dims(combined_logits, 0)
