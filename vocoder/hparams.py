from pathlib import Path

syn_dirs = [Path('/dataset/done_HarryPotter/part136')]
voc_dirs = [Path('/dataset/done_HarryPotter/part136/voc')]

# Audio settings------------------------------------------------------------------------
# Match the values of the synthesizer
sample_rate = 24000#_syn_hp.sample_rate
n_fft = 2048 #_syn_hp.n_fft
num_mels = 80 #_syn_hp.num_mels
hop_length = 300 #_syn_hp.hop_size=sample_rate*0.0125
win_length = 1200 #_syn_hp.win_size=sr*0.05
fmin = 95 #_syn_hp.fmin
min_level_db = -100 #_syn_hp.min_level_db
ref_level_db = 20 #_syn_hp.ref_level_db
mel_max_abs_value = 4 #_syn_hp.max_abs_value
preemphasis = 0.97#_syn_hp.preemphasis
apply_preemphasis = False #_syn_hp.preemphasize

bits = 10                            # bit depth of signal
mu_law = False                       # Recommended to suppress noise if using raw bits in hp.voc_mode
                                    # below
width_rnn = False

# WAVERNN / VOCODER --------------------------------------------------------------------------------
voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from 
# mixture of logistics)
voc_upsample_factors = (5, 5, 12)    # (5, 5, 12) NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10
voc_clip_grad_norm = 1.0
# Training
voc_batch_size = 144*2
voc_lr = 1e-3
voc_gen_at_checkpoint = 3           # number of samples to generate at each checkpoint
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider 
                                    # than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 12000                   # target number of samples to be generated in each batch entry
voc_overlap = 600                   # number of samples for crossfading between batches
# milestones = [50,300,600,900]
# MultiStepLR_gamma = 0.5
milestones = [50,500,700]
MultiStepLR_gamma = 0.1