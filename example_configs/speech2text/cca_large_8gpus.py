# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import ListenAttendSpellEncoder
from open_seq2seq.decoders import JointCTCAttentionDecoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.decoders import Conv2LetterDecoder

from open_seq2seq.data import Speech2TextDataLayer
from open_seq2seq.losses import MultiTaskCTCEntropyLoss
from open_seq2seq.optimizers.lr_policies import poly_decay

base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": True,
    "num_epochs": 50,

    "num_gpus": 8,
    "batch_size_per_gpu": 64,
    "iter_size": 1,

    "save_summaries_steps": 1000,
    "print_loss_steps": 10,
    "print_samples_steps": 200,
    "eval_steps": 1100,
    "save_checkpoint_steps": 1100,
    "logdir": "jca_log_folder",

    "optimizer": "Adam",
    "optimizer_params": {
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 1e-3,
        "power": 2.0,
        "min_lr": 1e-6
    },

    "max_grad_norm": 1.0,

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.0001
    },

    #"dtype": "mixed",
    #"loss_scaling": "Backoff",

    "dtype": tf.float32,

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": ListenAttendSpellEncoder,
    "encoder_params": {

        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 7,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 384, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 3,
                "kernel_size": [11], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dropout_keep_prob": 0.8,
            },
            {
                "type": "conv1d", "repeat": 4,
                "kernel_size": [11], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dropout_keep_prob": 0.7,
            },
        ],

        "recurrent_layers": [],

        "dropout_keep_prob": 0.8,

        "residual_connections": False,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
    },

    "decoder": JointCTCAttentionDecoder,
    "decoder_params": {
        "attn_decoder": Conv2LetterDecoder,
        "attn_params": {
          "tgt_emb_size": 256,

          "pos_embedding": False,

          "attention_params": {
              "attention_dim": 256,
              "attention_type": "bahadanu",
              #"use_coverage": True,
          },

          "fc_params": [512, 384],

          "convnet_params": {
              "normalization": None,
              "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
              "data_format": 'channels_last',
              "convnet_layers": [
                {
                  "type": "tcn", "repeat": 1,
                  "kernel_size": [11], "stride": [1],
                  "num_channels": 512, "padding": "VALID",
                  "dropout_keep_prob": 0.8,
                },
              ]
          },

          "dropout_keep_prob": 0.8,
        },
        "ctc_decoder": FullyConnectedCTCDecoder,
        "ctc_params": {
          "initializer": tf.contrib.layers.xavier_initializer,
          "use_language_model": False,

          # params for decoding the sequence with language model
          "beam_width": 512,
          "lm_weight": 1.0,
          "word_count_weight": 1.5,
          "valid_word_count_weight": 2.5,

          "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
          "lm_binary_path": "language_model/lm.binary",
          "lm_trie_path": "language_model/trie",
          "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        },
    },

    "loss": MultiTaskCTCEntropyLoss,
    "loss_params": {

      "seq_loss_params": {
        "offset_target_by_one": False,
        "average_across_timestep": True,
        "do_mask": True
      },

      "ctc_loss_params": {
      },

      "lambda_value": 0.2,
      "lambda_params": {
        "values": [],
        "boundaries": [],
      }
    }
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv",
        ],
        "max_duration": 16.7,
        "shuffle": True,
        "autoregressive": True,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
        "autoregressive": True,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
        "autoregressive": True,
    },
}