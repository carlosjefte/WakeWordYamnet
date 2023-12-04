import os

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import soundfile

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

testing_wav_file_name = "data/wake_word/hey_darwin.0007.wav"

print(testing_wav_file_name)
print(type(testing_wav_file_name))

# Utility functions for loading audio files and making sure the sample rate is correct.


def transform_to_pcm16(filename):
    data, samplerate = soundfile.read(filename)
    soundfile.write(filename, data, samplerate, subtype='PCM_16')

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """

    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

testing_wav_data = load_wav_16k_mono(testing_wav_file_name)

_ = plt.plot(testing_wav_data)

# Play the audio file.
display.Audio(testing_wav_data, rate=16000)

class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names =list(pd.read_csv(class_map_path)['display_name'])

for name in class_names[:20]:
  print(name)
print('...')

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
class_scores = tf.reduce_mean(scores, axis=0)
top_class = tf.math.argmax(class_scores)
inferred_class = class_names[top_class]

print(f'The main sound is: {inferred_class}')
print(f'The embeddings shape: {embeddings.shape}')

W_AUDIO_FILE = "data/wake_word/"
NW_AUDIO_FILE = "data/not_wake_word/"
FINAL_AUDIO_FILE = "data/wake_word_data.csv"

all_data = []

data_path_dict = {
    0: [NW_AUDIO_FILE + fDir for fDir in os.listdir(NW_AUDIO_FILE)],
    1: [W_AUDIO_FILE + fDir for fDir in os.listdir(W_AUDIO_FILE)],
}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        transform_to_pcm16(single_file)
        all_data.append([single_file, class_label, 1])

df = pd.DataFrame(all_data, columns=["feature", "class_label", "fold"])
df.to_pickle(FINAL_AUDIO_FILE)

df.head()

base_data_path = './'

pd_data = df
pd_data.head()

my_classes = ['not_wake_word', 'wake_word']
my_classes_ip = [0, 1]
map_class_to_id = {'not_wake_word':0, 'wake_word':1}

filtered_pd = pd_data[pd_data.class_label.isin(my_classes_ip)]

filtered_pd.head(10)

filenames = filtered_pd['feature']
targets = filtered_pd['class_label']
folds = filtered_pd['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
main_ds.element_spec

def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec

# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec

dataset_size = main_ds.cardinality().numpy()

train_size = int(0.7 * dataset_size) 
valid_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - valid_size

train_ds = main_ds.take(train_size).cache()
val_ds = main_ds.take(valid_size).cache().cache()
test_ds = main_ds.take(test_size).cache().cache()

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))
], name='wake_word_detection')

my_model.summary()

my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=3,
                                            restore_best_weights=True)

history = my_model.fit(train_ds,
                       epochs=20,
                       validation_data=val_ds,
                       callbacks=callback)

loss, accuracy = my_model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
result = my_model(embeddings).numpy()

inferred_class = my_classes[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}')

class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)

saved_model_path = 'saved_model/wake_word_yamnet'

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

tf.keras.utils.plot_model(serving_model)

reloaded_model = tf.saved_model.load(saved_model_path)

reloaded_results = reloaded_model(testing_wav_data)
wake_or_not = my_classes[tf.math.argmax(reloaded_results)]
print(f'The main sound is: {wake_or_not}')

serving_results = reloaded_model.signatures['serving_default'](testing_wav_data)
cat_or_dog = my_classes[tf.math.argmax(serving_results['classifier'])]
print(f'The main sound is: {cat_or_dog}')