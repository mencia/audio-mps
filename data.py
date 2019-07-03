import tensorflow as tf
import numpy as np
from scipy.linalg import block_diag
import tensorflow_probability as tfp
tfd = tfp.distributions

FLAGS = tf.flags.FLAGS

def get_audio(datadir, dataset, hps):

    if dataset == 'damped_sine':

        input_length = FLAGS.sample_duration

        freq = 261.6 # Middle C
        decay_time = 0.1

        # freq = 600.
        # decay_time = 0.003

        delay_time = input_length / 100

        delays = tf.stack(input_length * [tf.random_gamma([hps.minibatch_size], alpha=2, beta=2/delay_time)], axis=-1)

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = (input_range - delays) * hps.delta_t
        sine_wave_random_delay = 0.5 * (tf.sign(times) + 1) \
                                 * tf.sin(2 * np.pi * freq * times) * tf.exp(- times / decay_time)

        data = sine_wave_random_delay
        datalog = f"_freq{freq}_dect{decay_time}_delt{delay_time}"

    elif dataset == 'fixed_damped_sine':

        input_length = FLAGS.sample_duration
        freq = 800.
        decay_time = 0.003

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = input_range * hps.delta_t
        sine_wave_fixed = tf.sin(2 * np.pi * freq * times) * tf.exp(- times / decay_time)

        data = sine_wave_fixed
        datalog = f"_freq{freq}_dect{decay_time}_fix"

    elif dataset == 'fixed_damped_sine_2_freq':

        input_length = FLAGS.sample_duration
        freqa = 600.
        freqb = 800.
        decay_time = 0.003

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times = input_range * hps.delta_t
        sine_wave_fixed_a = tf.sin(2 * np.pi * freqa * times) * tf.exp(- times / decay_time)
        sine_wave_fixed_b = tf.sin(2 * np.pi * freqb * times) * tf.exp(- times / decay_time)

        data = tf.concat([sine_wave_fixed_a, sine_wave_fixed_b], 0)
        datalog = f"_freqa{freqa}_freqb{freqb}_dect{decay_time}_fix"

    elif dataset == 'damped_sine_2_freq':

        input_length = FLAGS.sample_duration

        # freq1 = 261.6 # Middle C
        # freq2 = 0.5 * freq1
        # decay_time = 0.1

        freq1 = 600.
        freq2 = 800.
        decay_time = 0.003

        delay_time = input_length / 100
        delays = tf.stack(input_length * [tf.random_gamma([np.int(hps.minibatch_size)], alpha=2, beta=2 / delay_time)], axis=-1)

        input_range = tf.expand_dims(tf.range(input_length, dtype=np.float32), axis=0)
        times_a = (input_range - delays[:np.int(hps.minibatch_size / 2)]) * hps.delta_t
        times_b = (input_range - delays[-np.int(hps.minibatch_size / 2):]) * hps.delta_t

        sine_wave_random_delay_1 = 0.5 * (tf.sign(times_a) + 1) \
                                   * tf.sin(2 * np.pi * freq1 * times_a) * tf.exp(- times_a / decay_time)
        sine_wave_random_delay_2 = 0.5 * (tf.sign(times_b) + 1) \
                                   * tf.sin(2 * np.pi * freq2 * times_b) * tf.exp(- times_b / decay_time)

        data = tf.concat([sine_wave_random_delay_1,sine_wave_random_delay_2],0)
        datalog = f"_freqa{freq1}_freqb{freq2}_dect{decay_time}_delt{delay_time}"

    elif dataset == 'gaussian_process':

        # D=1
        λ = [800]
        ω = [4800]
        σ = [2.]
        D_mix = len(λ)

        # D=2
        # λ = [50.*16,50.*16]
        # ω = [300.*16,500.*16]
        # σ = [1., 1.]
        # D_mix = len(λ)

        # D=3
        # λ = [50. * 16, 50. * 16, 50. * 16]
        # ω = [300. * 16, 500. * 16, 700. * 16]
        # σ = [1., 1., 1.]
        # D_mix = len(λ)

        Δt = hps.delta_t

        A_list = []
        for i in range(D_mix):
            A_list.append(np.exp(-λ[i] * Δt) * np.array([[np.cos(ω[i] * Δt), -np.sin(ω[i] * Δt)],
                                                         [np.sin(ω[i] * Δt), np.cos(ω[i] * Δt)]]))
        A_tf = tf.cast(block_diag(*A_list), dtype=tf.float32)

        def spectral_mixture(T, BATCH_SIZE, D_mix):
            """T: length of the notes
               BATCH_SIZE: number of notes"""
            # Concatenate all the q_matrix matrices in the second index, i.e. axis = 1

            α_list = []
            for i in range(D_mix):
                α_list.append(
                    tf.constant(σ[i]) * tf.sqrt(1 - tf.exp(-2 * tf.constant(λ[i] * Δt))))

            q_matrix_list = []
            for i in range(D_mix):
                q_matrix_list.append(tfd.Normal(loc=tf.constant(0.), scale=α_list[i]).sample([T, 2, BATCH_SIZE]))

            q_matrix = tf.concat(q_matrix_list, 1)  # axis = 1 concatenates in the second index

            f_initial_list = []
            for i in range(D_mix):
                f_initial_list.append(tfd.Normal(loc=tf.constant(0.), scale=σ[i]).sample([2, BATCH_SIZE]))

            f_initial = tf.concat(f_initial_list, 0)  # axis = 0 concatenates in the first index

            # Pick the odd elements of f=(f_1, f_2, ...) and sum (measurement model) them, both in the second index
            f_1 = tf.reduce_sum(tf.scan(lambda f, q: tf.matmul(A_tf, f) + q, elems=q_matrix,
                                        initializer=f_initial)[:, ::2], 1)

            return tf.transpose(f_1)


        data = spectral_mixture(FLAGS.sample_duration, hps.minibatch_size, D_mix)
        datalog = f"_Dmix{D_mix}"

    elif dataset == 'fixed_gaussian_process':

        daton = np.array([[-2.047775, -1.0920081, 0.46380842, 0.74769235, 1.3238779,
                           3.0695806, 4.2001925, 4.4789076, 4.779608, 4.3531127,
                           2.38811, 2.0234566, 0.03771305, -0.9140955, -1.3176844,
                           -1.9320265, -2.8425722, -2.8513892, -3.009201, -2.1494293,
                           -0.34900385, 1.1899669, 1.8022792, 1.3343902, 1.1101762,
                           1.3559395, 1.9228747, 0.556556, 0.24065384, 0.09251953,
                           -0.51159763, -0.8907386, -1.2283769, -1.4584271, -1.7504056,
                           -1.3575411, -1.0909736, -2.3564634, -1.3986703, -0.8463199,
                           -1.6165619, -1.054272, -1.4724426, -1.6651806, -0.7144271,
                           0.5766889, 1.2053337, 2.0506194, 1.7789936, 1.8087361,
                           1.2094569, 1.4659164, 0.75262326, 0.02318144, 0.06506933,
                           -0.12145653, 0.596879, 0.05797988, -0.814219, -1.8026068,
                           -1.6530755, -1.7706215, -2.238585, -1.3865492, -0.83658683,
                           -2.2838979, -2.1968155, -1.6094182, -0.76203614, -0.19294992,
                           0.7325621, 1.7797307, 2.8362677, 4.030409, 3.9990842,
                           4.1651373, 3.8359833, 2.5479188, 1.7521384, 0.35598382,
                           -1.3575882, -2.410719, -2.5184433, -2.0683775, -3.511917,
                           -3.001736, -2.985568, -2.4340158, -1.783841, -0.522154,
                           -0.41035572, 1.0345975, 1.2377586, 2.7098565, 2.8783133,
                           3.4934149, 3.415782, 3.4069328, 2.2665703, 2.000753]])


        data = tf.constant(daton, dtype=tf.float32)
        datalog = 'fixed_gaussian'

    elif dataset == 'poisson_process':

        Δt = hps.delta_t
        λ = 640
        τ = 0.00125
        ω = 3200
        N_st = np.int(5 * τ / Δt)  # STEADY-STATE
        N = N_st + FLAGS.sample_duration # INPUT_LENGTH = N
        T = N * Δt
        time_grid = np.arange(0, T, Δt) + Δt
        seed = None

        def Poisson_process(num_samples):

            tf.random.set_random_seed(seed)
            Nt_list = tf.random.poisson(λ * T, [num_samples], dtype=tf.int32, seed=seed)
            times_raw = tf.random.uniform(
                shape=[tf.reduce_sum(Nt_list)],
                minval=0,
                maxval=T,
                dtype=tf.float32,
                seed=seed,
                name=None)
            times_unsorted = tf.split(times_raw, Nt_list)
            jump_times = [tf.contrib.framework.sort(times_unsorted[i], direction='ASCENDING') for i in range(num_samples)]
            amplitudes_raw = 2 * tfd.Bernoulli(probs=0.5).sample(tf.reduce_sum(Nt_list), seed=seed) - 1
            amplitudes_raw = tf.cast(amplitudes_raw, tf.float32)
            amplitudes = tf.split(amplitudes_raw, Nt_list)

            process_list = []
            for k in range(num_samples):
                process_list.append(tf.reduce_sum(tf.multiply(amplitudes[k],
                            tf.transpose(tf.map_fn(lambda tk: 0.5 * (tf.sign(time_grid - (tk * tf.ones(N))) + 1.)
                                                              * tf.exp(-(time_grid - (tk * tf.ones(N))) / τ) *
                            tf.sin((time_grid - (tk * tf.ones(N))) * ω),jump_times[k]))), axis=1))
            return tf.stack(process_list)

        data = Poisson_process(hps.minibatch_size)[:, N_st:]
        datalog = '_lm'+str(λ)+'_tau'+str(τ)+'_w'+str(ω)+'_notelength'+str(N)+'_delta_t'+str(Δt)+\
                           '_Nst'+str(N_st)+'_seed'+str(seed)+"_smooth_Poisson"

    else:

        # LOAD DATA
        audio_dataset = tf.data.TFRecordDataset(f'{datadir}/{dataset}.tfrecords')

        # PARSE THE RECORD INTO TENSORS
        parse_function = lambda example_proto: \
            tf.parse_single_example(example_proto, {"audio": tf.FixedLenFeature([FLAGS.sample_duration], dtype=tf.float32)})
        #TODO change to 64000 when I drop the padding in future datasets
        audio_dataset = audio_dataset.map(parse_function)

        # CONSUMING TFRecord DATA
        audio_dataset = audio_dataset.batch(batch_size=hps.minibatch_size)
        audio_dataset = audio_dataset.shuffle(buffer_size=24)
        audio_dataset = audio_dataset.repeat()
        iterator = audio_dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        data = batch['audio']

    return data, datalog

