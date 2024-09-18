import os
import argparse
import json
import soundfile as sf
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

def is_audio(example):
    if isinstance(example, dict):
        if 'path' in example and 'array' in example and 'sampling_rate' in example:
            return True
    return False

def preprocess(
    hf_path: str,
    task_name: str,
    save_dir: Path
) -> None:
    task_dir = save_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    p_meta = task_dir / 'metadata.json'
    if p_meta.exists():
        print("Skip {}...".format(task_name))
        return

    dataset = load_dataset(hf_path, split='test') # Specify cache_dir if you want

    metadata = defaultdict(dict)
    for example in tqdm(dataset):
        file_path = Path(str(example["file"])) if 'file' in example else Path(str(example["file1"]))
        save_path = task_dir / f"{task_name}_{file_path.stem}.wav"

        if hf_path == 'DynamicSuperb/SuperbSV_VoxCeleb1-Test':
            audio = example['enrollment']['array']
            sr = example['enrollment']['sampling_rate']
            sf.write(save_path, audio, sr)

            extra_audio = example['testing']['array']
            extra_sr = example['testing']['sampling_rate']
            extra_save_path = task_dir / f"{task_name}_{file_path.stem}_pair2.wav"
            sf.write(extra_save_path, extra_audio, extra_sr)

        if hf_path == 'DynamicSuperb/SuperbQbE_Quesst14-Eval':
            audio = example['query']['array']
            sr = example['query']['sampling_rate']
            sf.write(save_path, audio, sr)

            extra_audio = example['doc']['array']
            extra_sr = example['doc']['sampling_rate']
            extra_save_path = task_dir / f"{task_name}_{file_path.stem}_pair2.wav"
            sf.write(extra_save_path, extra_audio, extra_sr)
            

        if 'audio' in example:
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            sf.write(save_path, audio, sr)

        # For multiple utterances as input (e.g. SV)
        index = 2
        key = "audio{}".format(index)
        while key in example:
            extra_audio = example[key]["array"]
            extra_sr = example[key]["sampling_rate"]
            extra_save_path = task_dir / f"{task_name}_{file_path.stem}_pair{index}.wav"
            sf.write(extra_save_path, extra_audio, extra_sr)

            index = index + 1
            key = "audio{}".format(index)

        example_key = save_path.name
        assert example_key not in metadata, f"Found duplicated key: {example_key}."

        for key in example.keys():
            if "audio" in key or is_audio(example[key]):
                continue
            metadata[example_key][key] = example[key]

    err_msg = (
        f"Length mismatch: metadata ({len(metadata)}) != dataset ({len(dataset)})."
    )
    assert len(metadata) == len(dataset), err_msg

    with (task_dir / "metadata.json").open(mode="w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

def main(task_csv: Path, save_dir: Path) -> None:    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(task_csv)
    for i in range(len(df)):
        hf_path = df.iloc[i, 1]
        task_name = os.path.basename(hf_path)
        print("Processing {}...".format(task_name))

        if task_name in [
            'EmotionalVoiceConversion_ESD',
            'SingingVoiceSynthesis_ACEOpencpop-Test700',
            'SongSeparation_SingSet',
            'SuperbSE_VoiceBankDEMAND-Test',
            'SuperbSS_Libri2Mix-Test',
        ]:
            task_dir = save_dir / 'DynamicSUPERB_AudioGeneration_Tasks'
        else:
            task_dir = save_dir / 'DynamicSUPERB_TextGeneration_Tasks'

        preprocess(
            hf_path=hf_path,
            task_name=task_name,
            save_dir=task_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_csv", type=Path, default='./DynamicSUPERB_ConfirmedTasks.csv', help='Tasks to be processed.')
    parser.add_argument("--save_dir", type=Path, help='Save all tasks to this directory.')
    main(**vars(parser.parse_args()))

