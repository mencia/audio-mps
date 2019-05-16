import tensorflow as tf
import numpy as np
from model import CMPS, RhoCMPS, PsiCMPS
from data import get_audio

from tensorflow.contrib.training import HParams
hparams = HParams(minibatch_size=8, bond_dim=8, delta_t=0.001,
                  sigma=1, h_reg=0, r_reg=0, initial_rank=3, A=1)

tf.flags.DEFINE_integer("sample_duration", 2**8, "Duration of samples (as integer).")


class TestCMPS(tf.test.TestCase):

    def testHIsHermitian(self):

        model = CMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H, tf.linalg.adjoint(model.H))


class TestRhoCMPS(tf.test.TestCase):


    def testRho0IsADensityMatrix(self):

        model = RhoCMPS(hparams)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.rho_0, model.rho_0 / tf.trace(model.rho_0))
            self.assertAllClose(model.rho_0, tf.transpose(model.rho_0, conjugate=True))

    def testHInitialization(self):

        test_H_diag = np.random.rand(hparams.bond_dim).astype(dtype=np.float32)
        model = RhoCMPS(hparams, H_in=test_H_diag)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(model.H, np.diag(test_H_diag))

    def testTrivialUpdateOfAncilla(self):
        """
        Update with H=R=0
        """

        test_H_diag = np.zeros([hparams.bond_dim], dtype=np.float32)
        test_R = np.zeros(2*[hparams.bond_dim], dtype=np.float32)
        signal = np.random.rand(hparams.minibatch_size).astype(dtype=np.float32)
        model = RhoCMPS(hparams, H_in=test_H_diag, Rx_in=test_R, Ry_in=test_R)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            stack_rho_0 = tf.stack(hparams.minibatch_size * [model.rho_0])
            updated_rho = model._update_ancilla_rho(stack_rho_0, signal)
            self.assertAllClose(stack_rho_0, updated_rho)



if __name__ == '__main__':
      tf.test.main()