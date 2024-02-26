import os
from pydub import AudioSegment, silence
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
from time import time

class VAD:
    def __init__(self, target_folder='vad_audio'):
        self.target_folder = target_folder
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

    def detect_silence(self, path, min_len=100, thresh=1.6):
        signal = AudioSegment.from_wav(path)
        print(f'VAD detecting on {path} .....')
        s = silence.detect_silence(signal, min_silence_len=min_len, silence_thresh=signal.dBFS * thresh)
        return np.array(s) / 1000

    @staticmethod
    def dump_silence(silence_list, l=0.3):
        new_list = []
        for s in silence_list:
            if s[1] - s[0] >= l:
                new_list.append(s)
        return np.array(new_list)

    def remove_silence(self, signal, silence, sr):
        st = time()
        t = silence[:, 1] - silence[:, 0]
        signal_obj = np.empty(len(silence) + 2, dtype=object)
        print(f'removing silence...total {t.sum()} seconds')
        last = s_len = 0
        silence_counter = 0
        with tqdm(total=len(silence) - 1) as pbar:
            i = 0
            for s in silence:
                cur, nex = (s * sr).astype(int)
                if cur == 0:
                    last = nex
                    silence_counter += (nex-cur)/sr
                    continue
                else:
                    signal_obj[i] = (signal[last:cur], silence_counter+s_len/sr)
                    s_len += signal_obj[i][0].shape[0]
                i += 1
                last = nex
                silence_counter += (nex-cur)/sr
                pbar.update(1)
        signal_obj[i] = (signal[last:],silence_counter+s_len/sr)
        s_len += signal_obj[i][0].shape[0]
        signal_obj = [x for x in signal_obj if x is not None]
        print(f'removed {(signal.shape[0] - s_len) / sr} seconds,finished in {round(time() - st, 2)} seconds')
        return signal_obj

    def process_audio(self, audio_path):
        sr, signal = wavfile.read(audio_path)
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        silence = self.detect_silence(audio_path)
        silence = self.dump_silence(silence)
        signal_obj = self.remove_silence(signal, silence, sr)
        print(f'saving audio chunks into {self.target_folder} folder ..')
        for idx, (chunk,chunk_start) in tqdm(enumerate(signal_obj)):
            chunk_duration = len(chunk) / sr
            while chunk_duration > 0:
                chunk_part_duration = min(chunk_duration, 5)  # Maximum 5 seconds per chunk_part
                chunk_part = chunk[: int(chunk_part_duration * sr)]
                output_path = os.path.join(
                    self.target_folder, f'{audio_filename}_{chunk_start}.wav'
                )
                wavfile.write(output_path, sr, chunk_part)
                chunk = chunk[int(chunk_part_duration * sr):]  # Remove the saved chunk_part
                chunk_duration -= chunk_part_duration
                chunk_start += chunk_part_duration


