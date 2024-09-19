from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import torch
import numpy as np

import os
import json
import random
import librosa
import argparse
from tqdm import tqdm
from pathlib import Path

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
    parser.add_argument('--p_save', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=33)
    return parser.parse_args()

def get_pretrained_model(model_name='Qwen/Qwen2-Audio-7B-Instruct'):
    """
    Get a pre-trained model and tokenizer.

    Args:
        model_name: The model name to load. Defaults to 'Qwen/Qwen2-Audio-7B-Instruct'.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: The pre-trained model and tokenizer.
    """

    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto").eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    return model, processor

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

def create_query(processor, wavs, texts):
    """
    Create a query for QWen model from a list of wave files and a list of text prompts.

    Args:
        wavs (List[str]): A list of paths to wave files.
        texts (List[str]): A list of text prompts.

    Returns:
        query (dict): A query for QWen model. The query is a dict with two keys, 'audio' and 'text'.
    """
    wavs_query = [
        {'type': 'audio', 'audio_url': str(p_wav)} for p_wav in wavs if os.path.exists(p_wav)
    ]

    text_query = [
        {'type': 'text', 'text': text} for text in texts
    ]

    query = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': wavs_query + text_query}
    ]

    text = processor.apply_chat_template(query, add_generation_prompt=True, tokenize=False)
    audios = [
        librosa.load(wav, sr=processor.feature_extractor.sampling_rate)[0] for wav in wavs
    ]

    return text, audios

def main():
    args = parse_args()
    set_random_seed(args.seed)

    model_name = 'Qwen/Qwen2-Audio-7B-Instruct'
    model, processor = get_pretrained_model(model_name)

    for metafile in tqdm(args.p_dataset.glob('*/metadata.json')):
        metadata = json.load(metafile.open('r'))
        taskname = metafile.parent.name
        print('Processing {}'.format(taskname))
        
        savefile = args.p_save / model_name / '{}.json'.format(taskname)
        savefile.parent.mkdir(parents=True, exist_ok=True)

        if savefile.exists():
            continue

        if taskname in [
            # 'PhonologicalFeatureClassification_VoxAngeles-Phone',
            'Emergency_traffic_detection_ETD',
        ]:
            print('Skip {}'.format(taskname))
            continue

        for pwav, example in tqdm(metadata.items()):
            p_wav = args.p_dataset / taskname / pwav

            # Read all the wave files
            wavs = read_wavs(p_wav)

            # Create the query
            text, audios = create_query(processor, wavs, [example['instruction']])
            
            # Generate the response
            inputs = processor(text=text, audios=audios, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True).to('cuda')
            # inputs.input_ids = inputs.input_ids.to('cuda')

            outputs = model.generate(**inputs, max_new_tokens=256)
            outputs = outputs[:, inputs.input_ids.size(1):]

            response = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Record response
            metadata[pwav]['sllm_name'] = model_name
            metadata[pwav]['sllm_response'] = response

        # Save the results
        json.dump(metadata, savefile.open('w'), indent=4, ensure_ascii=False)
        print('Done {}'.format(taskname))


    # for p_result in tqdm(args.p_results.glob('Qwen-Audio-Chat/*.json')):
    #     result = json.load(p_result.open('r'))
    #     metadata = json.load((args.p_dataset / p_result.stem / 'metadata.json').open('r'))
        
    #     new_result = {}
    #     for k, v in metadata.items():
    #         if k in result:
    #             new_result[k] = result[k]
    #             continue
            
    #         p_wav1 = args.p_dataset / p_result.stem / v['file']
    #         p_wav2 = args.p_dataset / p_result.stem / v['file2'] if 'file2' in v else None
            
    #         if p_wav2 is None:
    #             query = tokenizer.from_list_format([
    #                 {'audio': str(p_wav1)}, # Either a local path or an url
    #                 {'text': metadata['instruction']},
    #             ])
    #             response, history = model.chat(tokenizer, query=query, history=None)
    #             new_result[k] = {
    #                 "instruction": metadata['instruction'],
    #                 "response": response,
    #                 "model": "Qwen-Audio-Chat",
    #                 "label": metadata['label']
    #             }
    #         else:
    #             query = tokenizer.from_list_format([
    #                 {'audio': str(p_wav1)}, # Either a local path or an url
    #                 {'audio': str(p_wav2)},
    #                 {'text': metadata['instruction']},
    #             ])
    #             response, history = model.chat(tokenizer, query=query, history=None)
    #             new_result[k] = {
    #                 "instruction": metadata['instruction'],
    #                 "response": response,
    #                 "model": "Qwen-Audio-Chat",
    #                 "label": metadata['label']
    #             }

    #     p_save_result = args.p_save / p_result.relative_to(args.p_results)
    #     p_save_result.parent.mkdir(parents=True, exist_ok=True)
    #     json.dump(new_result, p_save_result.open('w'), indent=4)

if __name__ == '__main__':
    main()
