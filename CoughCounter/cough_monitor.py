import argparse
import os
from vad import VAD
from dataloaders import AudioSeg5s, CoughSeg5s
from pl_interface import Classification_Interface,Detection_Interface
from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


def cough_monitoring(target_path):
    vad = VAD()
    audio_files = [file for file in os.listdir(target_path) if file.endswith('.wav')]
    for audio_file in audio_files:
        audio_path = os.path.join(target_path, audio_file)
        vad.process_audio(audio_path)

    HUBERTC_interface = Classification_Interface()
    trainer = Trainer()
    audioseg = AudioSeg5s()
    HUBERTC_checkpoint = 'checkpoints/Classification.ckpt'
    HUBERTC_interface.eval()
    with torch.no_grad():
        print('Classifying on Audio Segments..')
        predictions = trainer.predict(HUBERTC_interface,audioseg,ckpt_path=HUBERTC_checkpoint )
        audioseg.update_dataframe(predictions)

    HUBERTRF_interface = Detection_Interface()
    trainer = Trainer()
    coughseg = CoughSeg5s()
    HUBERTRF_checkpoint = 'checkpoints/Detection.ckpt'
    HUBERTRF_checkpoint = torch.load(HUBERTRF_checkpoint)
    HUBERTRF_interface.load_state_dict(HUBERTRF_checkpoint["state_dict"])
    HUBERTRF_interface.eval()
    with torch.no_grad():
        print('Detecting Coughs on Cough Segments..')
        predictions = trainer.predict(HUBERTRF_interface,coughseg)
        coughseg.update_dataframe(predictions)
        result_path = coughseg.generate_result_txt(coughseg.labels)

    return result_path

def plot_cough_frequency(result_path, audio_path):

    with open(result_path, 'r') as f:
        lines = f.readlines()

    current_audio = None
    cough_timestamps = []

    for line in lines:
        if line.startswith("Total Coughs for"):
            # Extract audio file name from the summary line
            current_audio = line.split(":")[0].split("Total Coughs for ")[1] + '.wav'
            # If previous audio file ended, generate plot for that file
            if current_audio and cough_timestamps:
                plot_histogram(cough_timestamps, current_audio, audio_path)
                cough_timestamps = []
        elif line.startswith("cough"):
            # Collect cough timestamps for the current audio file
            cough_timestamps.append(float(line.split()[1]))

    # Generate plot for the last audio file
    if current_audio and cough_timestamps:
        plot_histogram(cough_timestamps, current_audio, audio_path)

def plot_histogram(cough_timestamps, audio_file, audio_path):
    # Read the audio file to determine its duration
    sr, signal = wavfile.read(audio_path + audio_file)
    audio_duration = len(signal) / sr

    # Determine the unit duration for the histogram
    if audio_duration <= 60:  # Less than or equal to a minute
        unit_duration = 10  # Coughs per 10 seconds
        time = 'seconds'
    elif 60 < audio_duration < 3600:  # Between 1 minute and an hour
        unit_duration = 60  # Coughs per minute
        time = 'minutes'
    else:
        unit_duration = 3600  # Coughs per hour
        time = 'hours'

    # Create histogram bins
    bins = [i * unit_duration for i in range(int(np.ceil(audio_duration / unit_duration)) + 1)]
    
    bins[-1] = audio_duration
    # Plot the histogram
    plt.hist(cough_timestamps, bins=bins, edgecolor='black')
    plt.xlabel(time)

    plt.ylabel('Frequency')
    plt.yticks(np.arange(0,len(cough_timestamps)+1))
    plt.title(f'Cough Frequency Histogram of {audio_file}')
    plt.savefig(f'cough_monitor_results/{audio_file}.png')
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='perform cough monitoring on audio files.')
    parser.add_argument('target_path', type=str, help='Path to the directory containing audio files')
    args = parser.parse_args()
    result_path = cough_monitoring(args.target_path)
    plot_cough_frequency(result_path, args.target_path)
    print('cough monitoring complete, see results in cough_monitor_results directory.')