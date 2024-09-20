from transformers import WhisperFeatureExtractor
import torch
import numpy as np
import random

import os
import json
import argparse

import librosa
import soundfile as sf

from tqdm import tqdm
from pathlib import Path

from config import Config
from models.salmonn import SALMONN

def set_random_seed(seed):
    """
    Set the random seed for all random number generators.

    Args:
        seed: The random seed to use.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    """
    Parse command line arguments for inference.

    Args:
        p_dataset (Path): Path to the dataset folder. Defaults to /livingrooms/wcchen/Dynamic_Datasets/.
        p_results (Path): Path to the results folder. Required.
        p_save (Path): Path to the save folder. Required.
        seed (int): The random seed. Defaults to 33.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_dataset', type=Path, default='/livingrooms/wcchen/Dynamic_Datasets/')
    # parser.add_argument('--p_results', type=Path, required=True)
    parser.add_argument('--p_save', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=33)
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()

def get_pretrained_model(config):
    """
    Get a pre-trained model and tokenizer.

    Args:
        model_name: The model name to load. Defaults to 'Qwen/Qwen-Audio-Chat'.

    Returns:
        tuple[SALMONN, WhisperFeatureExtractor]: The pre-trained model and processor.
    """
    model = SALMONN.from_config(config.config.model).eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(config.config.model.whisper_path)
    return model, wav_processor

def read_wavs(p_wav):
    """
    Read all the wave files in the same directory as `p_wav`, including itself.
    
    The function assumes that the wave files are named as follows:
    
    - The first wave file is named as `p_wav`.
    - The second wave file is named as `p_wav.stem + "_pair2." + p_wav.suffix`.
    - The third wave file is named as `p_wav.stem + "_pair3." + p_wav.suffix`.
    - And so on.
    
    The function will return a list of strings, where each string is the path to a wave file.
    """
    wavs = [str(p_wav)]
    index = 2
    while True:
        pair_wav_key = '{}_pair{}.{}'.format(p_wav.stem, index, 'wav')
        pair_wav_path = p_wav.parent / pair_wav_key
        if pair_wav_path.exists():
            wavs.append(str(pair_wav_path))
            index += 1
        else:
            break
    return wavs

def prepare_one_sample(p_wavs, wav_processor, device='cuda:0'):
    """
    Prepare a single sample from a list of wave files.

    Args:
        p_wavs (List[Path]): A list of paths to wave files.
        wav_processor (WavProcessor): A WavProcessor instance.
        device (str, optional): The device to put the tensors on. Defaults to 'cuda:0'.

    Returns:
        dict: A dict containing the spectrogram, raw audio, and padding mask.
    """
    # wavs = [sf.read(p_wav) for p_wav in p_wavs]
    # sr_set = set([sr for _, sr in wavs])
    # sr = wavs[0][1] # initial sr
    # if len(sr_set) > 1:
    #     sr = min(sr_set)
    #     wavs = [sf.read(p_wav, samplerate=sr) for p_wav in p_wavs]
    SAMPLING_RATE = 16000
    wavs = []
    for p_wav in p_wavs:
        wav, sr = sf.read(p_wav)
        if sr != SAMPLING_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLING_RATE)
        wavs.append(wav)

    # Reduce stereo to mono
    wavs = [wav[:, 0] if len(wav.shape) > 2 else wav for wav in wavs]   # Does our audio have 2 channels?
    
    # Concat all wavs and insert silence (0.5s) in between
    new_wavs = []
    for wav in wavs:
        new_wavs.append(wav)
        new_wavs.append(np.zeros(int(0.5 * SAMPLING_RATE)))

    wav = np.concatenate(new_wavs[:-1], axis=0)

    # Pad to at least 1s and truncate to at most 30s
    if len(wav) < SAMPLING_RATE:
        sil = np.zeros(SAMPLING_RATE - len(wav), dtype=float)
        wav = np.concatenate((wav, sil), axis=0)
    wav = wav[ : SAMPLING_RATE * 30]

    spectrogram = wav_processor(wav, sampling_rate=SAMPLING_RATE, return_tensors="pt")['input_features']
    samples = {
        'spectrogram': spectrogram.to(device),
        'raw_wav': torch.from_numpy(wav).unsqueeze(0).to(device),
        'padding_mask': torch.zeros(len(wav), dtype=torch.bool).unsqueeze(0).to(device),
    }
    return samples


def main():
    args = parse_args()
    cfg = Config(args)
    set_random_seed(args.seed)

    model_name = 'SALMONN/SALMONN-7B' if '7B' in args.cfg_path else 'SALMONN/SALMONN-13B'
    model, wav_processor = get_pretrained_model(cfg)
    model = model.to(args.device)
    print(model_name)

    for metafile in tqdm(args.p_dataset.glob('*/metadata.json')):
        taskname = metafile.parent.name
        print('Processing {}'.format(taskname))
        metadata = json.load(metafile.open('r'))
        savefile = args.p_save / model_name / '{}.json'.format(taskname)
        savefile.parent.mkdir(parents=True, exist_ok=True)

        if savefile.exists():
            print('Skip {}...'.format(taskname))
            continue

        if taskname in [
            'Emergency_traffic_detection_ETD',
            # 'PhonologicalFeatureClassification_VoxAngeles-Phone'
        ]:
            print('Skip {}'.format(taskname))
            continue

        for pwav, example in tqdm(metadata.items()):
            p_wav = args.p_dataset / metafile.parent / pwav

            # Read all the wave files
            wavs = read_wavs(p_wav)

            # Create the query
            if p_wav.exists():
                samples = prepare_one_sample(wavs, wav_processor, device=args.device)
                prompt = [
                    cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + example['instruction'].strip())
                ]
            else:
                print('Skip {} due to missing wave files...'.format(taskname))
                break


            with torch.cuda.amp.autocast(dtype=torch.float16):
                response = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
                
            def remove_special_tokens(text):
                input_ids = model.llama_tokenizer(text, add_special_tokens=False).input_ids
                return model.llama_tokenizer.decode(input_ids, skip_special_tokens=True).strip()

            response = remove_special_tokens(response)

            # Record response
            metadata[pwav]['sllm_name'] = model_name
            metadata[pwav]['sllm_response'] = response

        # Save the results
        json.dump(metadata, savefile.open('w'), indent=4, ensure_ascii=False)
        print('Done {}'.format(taskname))

if __name__ == '__main__':
    main()
