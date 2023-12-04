import sounddevice as sd
from scipy.io.wavfile import write

AUDIO_FILE = r"data/wake_word/"
NOT_WAKE_FILE = r"data/not_wake_word/"

def record_and_save_audio(save_path, n_times = 2100):
    confiramtion = input("do you want to record wake word sounds?(y/n) ")
    if confiramtion.lower() == "y":
        start_mark = input("type the index you want to start at: ")
        input("Press Enter to start recording: ")
        for i in range(n_times):
            if i >= int(start_mark):
                fs = 44100
                seconds = 1
                myRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()
                write(save_path + str(i) + ".wav", fs, myRecording)
                print(f"Currently on: {i + 1} / {n_times}")
                input("Press Enter to start recording: ")
                

def record_background(save_path, n_times = 150):
    confiramtion = input("do you want to record backgroun sounds?(y/n) ")
    if confiramtion.lower() == "y":
        start_mark = input("type the index you want to start at: ")
        input("Press Enter to start recording background: ")
        for i in range(n_times):
            if i >= int(start_mark):
                fs = 44100
                seconds = 1
                myRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()
                write(save_path + str(i) + ".wav", fs, myRecording)
                print(f"Currently on: {i + 1} / {n_times}")

record_and_save_audio(AUDIO_FILE)
record_background(NOT_WAKE_FILE)
