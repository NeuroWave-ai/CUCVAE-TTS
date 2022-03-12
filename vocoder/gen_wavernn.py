from vocoder.models.fatchord_version import WaveRNN
from vocoder.audio import *
import os

def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path,epoch, is_dataparallel=True):

    for i, (m, x) in enumerate(test_set, 1):
        if i > samples:
            break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL':
            x = decode_mu_law(x, 2 ** bits, from_labels=True)
        else:
            x = label_2_float(x, bits)
        os.makedirs(save_path.joinpath("wavs"), exist_ok =True)
        save_wav(x, save_path.joinpath("wavs/%depochs_%d_target.wav" % (epoch, i)))

        batch_str = "gen_batched_target%d_overlap%d" % (target, overlap) if batched else \
            "gen_not_batched"
        save_str = save_path.joinpath("wavs/%depochs_%d_%s.wav" % (epoch, i, batch_str))
        if is_dataparallel:
            wav = model.module.generate(m, batched, target, overlap, hp.mu_law)
        else:
            wav = model.generate(m, batched, target, overlap, hp.mu_law)
        save_wav(wav, save_str)

