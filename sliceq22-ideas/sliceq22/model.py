import os
import tensorflow as tf
import numpy as np
import time
import gc
import random
from tensorflow import keras
from essentia.standard import EasyLoader
import essentia.pytools.spectral as sp
from sliceq22.audio import SNR, extract_segment_randomly
from sliceq22.overlap import nsgcq_overlap_add
from sliceq22.musdb import MUSDBDataGenerator
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, Cropping2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class SliceQ22Model:
    def __init__(
            self,
            train_dir,
            nsg_params,
            dataset,
            epochs=1000,
            batch_size=1,
            seq_len=10.0,
            sample_rate=44100,
            model_file=None,
            enable_memory_growth=True,
            inference=False
        ):
        if enable_memory_growth:
            physical_devices = tf.config.list_physical_devices("GPU")
            tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

        self.train_dir = train_dir
        self.nsg_params = nsg_params
        self.dataset = dataset
        self.seq_len = seq_len
        self.sr = sample_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.musdb_train_generator = MUSDBDataGenerator(
            self.dataset['train'],
            self.nsg_params,
            seq_len=self.seq_len,
            sr=self.sr,
            batch_size=self.batch_size
        )
        self.musdb_val_generator = MUSDBDataGenerator(
            self.dataset['val'],
            self.nsg_params,
            seq_len=self.seq_len,
            sr=self.sr,
            batch_size=self.batch_size
        )
        self.musdb_test_generator = MUSDBDataGenerator(
            self.dataset['test'],
            self.nsg_params,
            seq_len=3600, # use full track for test
            sr=self.sr,
            batch_size=1
        )

        input_example, output_example = self.get_input_output_shapes(seq_len)
        print(f'input: {input_example.shape}, output: {output_example.shape}')

        if inference and not model_file:
            raise ValueError(f'you must pass a path to a trained model to run in inference mode')

        if not model_file:
            # generate timestamp suffix
            suffix = time.strftime("%Y%m%d-%H%M%S")
            self.model_file = os.path.join(self.train_dir, f"sliceq2_{suffix}.h5")
            self.checkpoint_file = os.path.join(self.train_dir, f"sliceq2_{suffix}.ckpt")

            self.model = self.build_model(input_example, output_example)
            if not inference:
                self.model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        else:
            self.model_file = model_file
            self.model = load_model(self.model_file)

        self.model.summary()

        model_out = self.model(input_example)
        print(f'model in: {input_example.shape}, model out: {model_out.shape}, real out: {output_example.shape}')
        print(f'model in: {input_example.dtype}, model out: {model_out.dtype}, real out: {output_example.dtype}')

    def get_input_output_shapes(self, seq_len):
        misc_file = random.choice(self.dataset['val'])['mix']

        # Load an audio file
        x_full = EasyLoader(filename=misc_file, sampleRate=self.sr)()
        x = extract_segment_randomly(x_full, self.sr, self.seq_len)

        # Forward and backward transforms
        cq_frames, dc_frames, nb_frames = sp.nsgcqgram(x, **self.nsg_params)

        y_frames = sp.nsgicqgram(cq_frames, dc_frames, nb_frames, **self.nsg_params)

        cq_snr = SNR(x, y_frames[:x.size])
        print('Reconstruction SNR of sliCQ-isliCQ (no overlap): {:.3f} dB'.format(cq_snr))

        n_cq_frames = len(cq_frames)
        cq_time_coefs = cq_frames[0].shape[-1]
        total_coefs = n_cq_frames * cq_time_coefs

        cq_frames_ndarray = np.asarray(cq_frames)
        mag_cq = np.abs(cq_frames_ndarray)
        mag_cq_ola = nsgcq_overlap_add(mag_cq)

        # add fake sample and channel dimension of 1
        mag_cq = np.expand_dims(np.expand_dims(mag_cq, axis=0), axis=-1)
        mag_cq_ola = np.expand_dims(np.expand_dims(mag_cq_ola, axis=0), axis=-1)

        return (
            mag_cq_ola,
            mag_cq,
        )

    def build_model(self, input_example, output_example):
        time_bins = output_example.shape[1]

        inputs = Input(shape=input_example.shape[1:])

        x = Conv2DTranspose(32, kernel_size=(1, time_bins), strides=(1, 3), activation='relu')(inputs)
        x = BatchNormalization()(x)
        print(f'LAYER_1:\t{x.shape}')
        x = Conv2D(24, kernel_size=(1, time_bins), activation='relu')(x)
        x = BatchNormalization()(x)
        print(f'LAYER_2:\t{x.shape}')
        x = Conv2D(1, kernel_size=1, activation=None)(x)
        print(f'LAYER_3:\t{x.shape}')
        x = Cropping2D(cropping=((0, 863)))(x)
        print(f'LAYER_4:\t{x.shape}')
        outputs = Reshape(output_example.shape[1:])(x)
        print(f'LAYER_5:\t{x.shape}')

        model = keras.Model(inputs=inputs, outputs=outputs, name="sliceq22_model")
        print(model.output_shape)
        return model

    def train(self):
        monitor = EarlyStopping(monitor="loss", patience=500)

        checkpoint = ModelCheckpoint(
            self.checkpoint_file,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )

        tboard = TensorBoard()

        try:
            self.model.fit(
                x=self.musdb_train_generator,
                # no need to specify y since the generator outputs it
                validation_data=self.musdb_val_generator,
                epochs=self.epochs,
                callbacks=[monitor, checkpoint, tboard, CustomMemoryCallback()],
                verbose=1,
            )
        except KeyboardInterrupt:
            print("interrupted by ctrl-c, saving model")
            self.model.save(self.model_file)

        #train_scores = self.model.evaluate(self.musdb_train_generator)
        #print(
        #    "train scores: %s: %.2f%%" % (self.model.metrics_names[1], train_scores[1] * 100)
        #)

        #test_scores = self.model.evaluate(self.musdb_test_generator)
        #print(
        #    "test scores: %s: %.2f%%" % (self.model.metrics_names[1], test_scores[1] * 100)
        #)

        print("saving model")
        self.model.save(self.model_file)

    def inference(self, audio_file):
        # perform inference on the overlap-added slicqt x to come up with the inverse
        # Load an audio file
        x_full = EasyLoader(filename=audio_file, sampleRate=self.sr)()
        n_fr = len(x_full)

        seq_len_samples = int(np.round(self.seq_len*self.sr))

        min_chunks = int(np.floor(n_fr/seq_len_samples))
        max_chunks = int(np.ceil(n_fr/seq_len_samples))

        print(f'min chunks: {min_chunks}, max chunks: {max_chunks}')

        pad = max_chunks*seq_len_samples - n_fr

        print(f'x_full pre-pad: {x_full.shape}')
        x_full = np.pad(x_full, (0, pad), mode='constant', constant_values=0)
        print(f'x_full post-pad: {x_full.shape}')

        Xs = []
        Y_gts = []
        Y_preds = []

        for chunk in range(max_chunks):
            x = x_full[chunk*seq_len_samples:(chunk+1)*seq_len_samples]
            print(f'x_full: {x_full.shape}, x: {x.shape}')

            # Forward and backward transforms
            cq_frames, dc_frames, nb_frames = sp.nsgcqgram(x, **self.nsg_params)

            y_frames = sp.nsgicqgram(cq_frames, dc_frames, nb_frames, **self.nsg_params)
            cq_snr = SNR(x, y_frames[:x.size])
            print('Reconstruction SNR of sliCQ-isliCQ (no overlap): {:.3f} dB'.format(cq_snr))

            n_cq_frames = len(cq_frames)
            cq_time_coefs = cq_frames[0].shape[-1]
            total_coefs = n_cq_frames * cq_time_coefs

            cq_frames_ndarray = np.asarray(cq_frames)
            mag_cq = np.abs(cq_frames_ndarray)
            mag_cq_ola = nsgcq_overlap_add(mag_cq)
            
            X = np.expand_dims(np.expand_dims(mag_cq_ola, axis=0), axis=-1)
            Y_gt = np.expand_dims(np.expand_dims(mag_cq, axis=0), axis=-1)
            Y_pred = self.model.predict(X)

            print(f'X: {X.shape}, Y_gt: {Y_gt.shape}, Y_pred: {Y_pred.shape}')

            Xs.append(X)
            Y_gts.append(Y_gt)
            Y_preds.append(Y_pred)

            #y_pred_frames = sp.nsgicqgram(cq_frames, dc_frames, nb_frames, **self.nsg_params)
            #cq_snr = SNR(x, y_frames[:x.size])
            #print('Reconstruction SNR of sliCQ-isliCQ (no overlap): {:.3f} dB'.format(cq_snr))

            #print(f'inference shapes: {X.shape}, {Y_gt.shape}, {Y_pred.shape}')

        return (
            np.concatenate(Xs, axis=-2),
            np.concatenate(Y_gts, axis=-2),
            np.concatenate(Y_preds, axis=-2)
        )


class CustomMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        keras.backend.clear_session()
