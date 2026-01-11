# Mangle Keras-Hub's TransformerDecoder to use FFNSwiGLU2...

import keras
from keras_hub.layers import TransformerDecoder

from ary_seq2seq.modeling.colmo import FFNSwiGLU2


class TransformerDecoderSwiGLU(TransformerDecoder):
	# Just overwirte the first FFN to avoid code duplication...
	# On the flipside, if the internal fields ever change names, we're screwed ;).
	def build(
		self,
		decoder_sequence_shape,
		encoder_sequence_shape=None,
		**kwargs,
	):
		super(TransformerDecoderSwiGLU, self).build(
			decoder_sequence_shape, encoder_sequence_shape=encoder_sequence_shape, **kwargs
		)

		# The graph basically looks like:
		# `_feedforward_intermediate_dense` -> `_feedforward_output_dense` -> `_feedforward_dropout`
		# With a `_feedforward_layer_norm` either at the start or at the end

		# Replace Dense w/ FFNSwiGLU2
		self._feedforward_intermediate_dense = FFNSwiGLU2(
			self.intermediate_dim,
			dtype=self.dtype_policy,
			name="feedforward_intermediate_swiglu",
		)
		self._feedforward_intermediate_dense.build(decoder_sequence_shape)

		# Replace LayerNormalization w/ RMSNormalization
		self._feedforward_layer_norm = keras.layers.RMSNormalization(
			dtype=self.dtype_policy,
			name="feedforward_layer_rmsnorm",
		)
		self._feedforward_layer_norm.build(decoder_sequence_shape)
