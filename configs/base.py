from easydict import EasyDict as edict

config = edict()

# Mel
config.num_mels = 80
config.text_cleaners = ['english_cleaners']

# FastSpeech
config.vocab_size = 300
config.max_seq_len = 3000

config.encoder_dim = 256
config.encoder_n_layer = 4
config.encoder_head = 2
config.encoder_conv1d_filter_size = 1024

config.decoder_dim = 256
config.decoder_n_layer = 4
config.decoder_head = 2
config.decoder_conv1d_filter_size = 1024

config.fft_conv1d_kernel = (9, 1)
config.fft_conv1d_padding = (4, 0)

config.duration_predictor_filter_size = 256
config.duration_predictor_kernel_size = 3
config.dropout = 0.1

# Train
config.run_name = 'baseline'
config.checkpoint_path = "./models/" + config.run_name
config.logger_path = "./logger/" + config.run_name
config.mel_ground_truth = "./mels"
config.alignment_path = "./alignments"

config.batch_size = 48
config.epochs = 200
config.n_warm_up_step = 400

config.learning_rate = 1e-3
config.frozen_learning_rate = False
config.weight_decay = 1e-6
config.grad_clip_thresh = 1.0
config.decay_step = [500000, 1000000, 2000000]

config.freeze_encoder = False
config.freeze_length_regulator = False
config.freeze_decoder = False
config.freeze_mel_linear = False
config.freeze_postnet = False

config.mel_loss_scale = 1.
config.mel_postnet_loss_scale = 1.
config.duration_loss_scale = 1.

config.save_step = 3000
config.log_step = 64
config.clear_Time = config.log_step * 5

config.batch_expand_size = 32
