import numpy as np
from tensorflow.saved_model import load
import time
from tensorflow.python import keras
import tensorflow as tf
import sounddevice as sd
import threading

fs = 22050
seconds = 1
step = 0.05
M = 50
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 16000
num_mfcc = 16
len_mfcc = 26

#tensorflow.compat.v1.disable_eager_execution()
tf.config.run_functions_eagerly(True)

model_path = r"saved_model/wake_word_yamnet"
my_classes = ['not_wake_word', 'wake_word']

class Wake_Word_listener:
    queue = []
    keras.backend.clear_session()
    model = None
    cancel_wake = False

    def listener(self):
        self.wake = False
        self.model = load(model_path)
        self.prediction_M()
        return self.wake

    def sound_capture(self):
        try:
            self.cancel_wake = False
            print('start talking')
            while self.wake == False:
                try:
                    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    self.queue.append(audio)
                    if self.wake == True:
                        break
                except Exception as e:
                    print(e)
            return
        except:
            self.cancel_wake = True
            return

    def load_wav_16k_mono(self, audio):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        return audio.flatten()

    def prediction_M(self):
        capt_audio = threading.Thread(target=self.sound_capture, daemon=True)
        capt_audio.start()
        while True:
            if len(self.queue) > 2:
                deff = len(self.queue) - 15
                for i in range(deff):
                    self.queue.pop(0)
            if len(self.queue) > 0:
                audio = self.load_wav_16k_mono(self.queue[len(self.queue) - 1])
                print(np.asarray(audio).shape)

                results = self.model(audio)
                wake_or_not = my_classes[tf.math.argmax(results)]

                print(results)

                if results.numpy()[1] > 0.7 and results.numpy()[0] < -0.7:
                    print("wake word detected!")
                if self.cancel_wake:
                    self.queue.clear()
                    return
                time.sleep(1)

if (__name__ == "__main__"):
    k = Wake_Word_listener()
    k.listener()
