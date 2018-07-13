# Copyright (c) 2018 NVIDIA Corporation
"""
This package contains various decoder.
A Decoder typically takes representation and produces data.
"""
from .decoder import Decoder
from .rnn_decoders import RNNDecoderWithAttention, \
                          BeamSearchRNNDecoderWithAttention
from .transformer_decoder import TransformerDecoder
from .fc_decoders import FullyConnectedCTCDecoder, FullyConnectedDecoder
from .fc_glu_decoders import FullyConnectedCTCDecoderGLU, FullyConnectedDecoderGLU
