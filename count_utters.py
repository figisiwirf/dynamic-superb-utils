import os 
import glob
import json
from tqdm import tqdm

import librosa
from typing import List

def read_wavs(p_wav):
    wavs = [p_wav]

    index = 2
    while True:
        key = '{}_pair{}.wav'.format(p_wav.replace('.wav', ''), index)
        p = os.path.join(os.path.dirname(p_wav), key)
        if os.path.exists(p):
            wavs.append(p)
            index += 1
        else:
            break
    return wavs

def get_duration(p_wav):
    wav, sr = librosa.load(p_wav)
    return wav.shape[0] / sr

def plot_histogram(
    array: List,
    title: str,
    filename: str
):
    # Plot the histogram of the durations
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    plt.hist(array, bins=100, range=(0, 50))
    ax = plt.gca()
    text = 'Max: {:.3f}s Min: {:.3f}s\nMean: {:.3f}s Std: {:.3f}s'.format(
                            np.max(array), 
                            np.min(array), 
                            np.mean(array), 
                            np.std(array)
                        )
    plt.text(0.9, 0.9, text, 
        horizontalalignment='right',            
        verticalalignment='top',
        transform = ax.transAxes
    )
    plt.vlines(30, 0, 5000, colors='r', linestyles='dashed')
    text = 'Duration > 30s:\n {:.5f}%'.format(sum(np.array(array) > 30) / len(array) * 100)
    plt.text(0.9, 0.5, text,
        horizontalalignment='right',            
        verticalalignment='top',
        transform = ax.transAxes
    )
    plt.title(title)
    plt.xlabel('Duration (s)')
    plt.savefig(filename)

def main():
    p_dir = '/livingrooms/wcchen/DynamicSUPERB_Tasks/DynamicSUPERB_TextGeneration_Tasks/'
    existing_wavs = glob.glob(os.path.join(p_dir, '*/*.wav'))

    all_wavs = []

    all_duration = []
    all_example_duration = []
    for fmeta in tqdm(glob.glob(os.path.join(p_dir, '*/metadata.json'))):
        metadata = json.load(open(fmeta, 'r'))

        for pwav in metadata:
            pwav = os.path.join( os.path.dirname(fmeta), pwav )
            if pwav not in existing_wavs:
                break
            
            p_wavs = read_wavs(pwav)
            all_wavs += p_wavs

            durations = [ get_duration(wav) for wav in p_wavs ]
            all_duration += durations
            all_example_duration.append( sum(durations) )
    
    print('Num of existing wavs: {}'.format(len(existing_wavs)))
    print('After deduplicate: {}\n'.format(len(set(existing_wavs))))

    print('Num of necessary utters: {}'.format(len(all_wavs)))
    print('After deduplicate: {}\n'.format(len(set(all_wavs))))

    print('Total duration: {}\n'.format(sum(all_duration)))

    print('Average duration per utterance: {}'.format(sum(all_duration) / len(all_duration)))
    print('Average duration per example: {}'.format(sum(all_example_duration) / len(all_example_duration)))

    plot_histogram(all_duration, title='Histogram of All Utterances', filename='./stat_images/histogram_utter_individual.png')
    plot_histogram(all_example_duration, title='Histogram of All Examples', filename='./stat_images/histogram_utter_example.png')
    
    # print('Neccessary utters not in existing wavs: \n{}\n'.format(
    #     [ p for p in all_wavs if p not in existing_wavs]
    # ))
        

    # print('Existing utters not in necessary wavs: \n{}\n'.format(
    #     [ p for p in existing_wavs if p not in all_wavs]
    # ))


if __name__ == "__main__":
    main()