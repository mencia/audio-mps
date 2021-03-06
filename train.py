import tensorflow as tf
import numpy as np
# import tfplot

from tensorflow.contrib.training import HParams
from model import RhoCMPS, PsiCMPS
from data import get_audio
# from utils import waveform_plot

DTYPE=tf.float32
NP_DTYPE=np.float32
tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

# Model flags
tf.flags.DEFINE_enum('mps_model', 'psi_mps',
                     ['rho_mps', 'psi_mps'],
                     'MPS model. Must be one of "rho_mps" or "psi_mps".')

# Training flags
tf.flags.DEFINE_enum('dataset', 'damped_sine',
                     ['poisson_process', 'fixed_gaussian_process', 'fixed_damped_sine_2_freq', 'damped_sine', 'fixed_damped_sine', 'damped_sine_2_freq', 'gaussian_process', 'guitar', 'organ', 'nsynth'],
                     'Dataset. Must be one of "damped_sine", "guitar", "organ", or "nsynth".')

tf.flags.DEFINE_integer("sample_duration", 2**16, "Duration of samples (as integer).")
tf.flags.DEFINE_integer("sample_rate", 16000, "Sampling rate.")
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization. Probably slow!')
tf.flags.DEFINE_integer('num_samples', 3, 'Number of samples to generate.')
tf.flags.DEFINE_string("hparams", "", 'Comma separated list of "name=value" pairs e.g. "--hparams=learning_rate=0.3"')
tf.flags.DEFINE_string("datadir", "./data", "Data directory.")
tf.flags.DEFINE_string("logdir", f"../logging/audio_mps/{FLAGS.dataset}", "Directory to write logs.")
tf.flags.DEFINE_string('filename', 'flags', 'Input file name.', short_name='f')


def main(argv):

    hparams = HParams(minibatch_size=8, bond_dim=100, delta_t=1/FLAGS.sample_rate, sigma=0.0001,
                      h_reg=200/(np.pi * FLAGS.sample_rate)**2, r_reg=0.1,
                      initial_rank=None, A=600., learning_rate=0.001)
    hparams.parse(FLAGS.hparams)

    with tf.variable_scope("data"):
        data, datalog = get_audio(datadir=FLAGS.datadir, dataset=FLAGS.dataset, hps=hparams)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        if FLAGS.mps_model == 'rho_mps':
            model = RhoCMPS(hparams=hparams, data_iterator=data)
        else:
            model = PsiCMPS(hparams=hparams, data_iterator=data)

        h_l2sqnorm = tf.reduce_sum(tf.square(model.freqs))
        r_l2sqnorm = tf.real(tf.reduce_sum(tf.conj(model.R) * model.R))

    with tf.variable_scope("total_loss"):
        total_loss = model.loss + hparams.h_reg * h_l2sqnorm \
                                    + hparams.r_reg * r_l2sqnorm

    with tf.variable_scope("summaries"):
        tf.summary.scalar("A", tf.cast(model.A, dtype=tf.float32))
        tf.summary.scalar("h_l2norm", tf.sqrt(h_l2sqnorm))
        tf.summary.scalar("r_l2norm", tf.sqrt(r_l2sqnorm))

        tf.summary.scalar("model_loss", tf.reshape(model.loss, []))
        tf.summary.scalar("total_loss", tf.reshape(total_loss, []))

        #rms(R+Rdag) summary
        rms_R_plus_Rdag = model.rms_R_plus_Rdag
        tf.summary.scalar("rms_R_plus_Rdag", tf.reshape(rms_R_plus_Rdag, []))


    step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate).minimize(total_loss, global_step=step)

    # TODO Unrolling in time?

    tf.contrib.training.train(train_op, save_checkpoint_secs=60,
                              logdir=f"{FLAGS.logdir}/"
                                     f"D{hparams.bond_dim}"
                                     f"_dt{hparams.delta_t}"
                                     f"_bs{hparams.minibatch_size}"
                                     f"_sg{hparams.sigma}"
                                     f"_sampdur{FLAGS.sample_duration}"
                                     +datalog+f"_{FLAGS.mps_model}")

if __name__ == '__main__':
    tf.app.run(main)