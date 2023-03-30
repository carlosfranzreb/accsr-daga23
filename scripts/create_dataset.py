""" Create the dataset for the experiments. First you have to download the
German CV10 from Mozilla's website. This script extracts the data of the
desired accents, storing each subset in a different file. The desired accents
are presented as a dictionary with the name of the accent in the dataset
as key and the value is the name of the file where the data will be stored."""


import json
import random
import os
import logging
from time import time as now

import pandas as pd
import torchaudio
import swifter


def get_duration(audio_filepath):
    """Return the duration of the given audio file."""
    audio, sr = torchaudio.load(audio_filepath)
    return audio.shape[1] / sr


def process_text(text, alphabet):
    """
    Lower-case the transcripts and remove all punctuation by checking if each character
    is in the alphabet.
    """
    text = text.lower()
    text = "".join([c for c in text if c in alphabet])
    return text


def get_accents(cv_loc, mapping, acc_folder, alphabet, root_folder):
    """Extract the data of the desired accents, storing each subset in a different file,
    as defined in the given mapping (acc -> dump file). Also, compute the duration of
    the audio clips."""
    clips_dir = os.path.join(cv_loc, "clips")
    df = pd.read_csv(os.path.join(cv_loc, "validated.tsv"), sep="\t")
    logging.info(f"CV contains {df.shape[0]} validated samples")
    df = df[df["sentence"].notnull()]  # remove samples without a transcript
    logging.info(f"{df.shape[0]} samples left after removing those without text")
    df.drop(
        columns=[
            "client_id",
            "up_votes",
            "down_votes",
            "age",
            "gender",
            "locale",
            "segment",
        ],
        inplace=True,
    )
    df.rename(
        columns={"path": "audio_filepath", "sentence": "text"},
        inplace=True,
    )
    logging.info(f"Accents found: {list(df['accents'].unique())}")
    for acc, f in mapping.items():
        data = df[df["accents"] == acc]
        logging.info(f"Found {data.shape[0]} samples for accent {acc}")
        data.drop(columns=["accents"], inplace=True)
        data["audio_filepath"] = data["audio_filepath"].swifter.apply(
            lambda x: os.path.join(clips_dir, x)
        )
        logging.info(f"Done adding directory to filepaths")
        data["duration"] = data["audio_filepath"].swifter.apply(get_duration)
        data["audio_filepath"] = data["audio_filepath"].swifter.apply(
            lambda x: x.replace(root_folder, "{root}")
        )
        logging.info(f"Done calculating durations")
        data["text"] = data["text"].swifter.apply(process_text, args=(alphabet,))
        logging.info(f"Done processing the texts")
        json_dict = json.loads(data.to_json(orient="records"))
        logging.info(f"Dumping {len(json_dict)} samples for accent {acc}")
        json.dump(json_dict, open(f"{acc_folder}/{f}.json", "w", encoding="utf-8"))


def split_data(
    pretrain_lang,
    seen,
    unseen,
    ratio,
    pretrain_hours,
    acc_folder,
    dump_folder,
    shuffle=True,
):
    """Split the data into pretrain, train and test files. The pretrain file comprises
    aprox. "pretrain_hours" hours of one language (pretrain_lang). The remaining objects
    of that dataset are treated as a seen dataset. Seen datasets are split into train
    and test files according to the given ratio. Unseen files are test files."""
    logging.info(f"Pre-train accent: {pretrain_lang} ({pretrain_hours} h)")
    logging.info(f"Seen accents: {seen} with train/test ratio {ratio}/{1-ratio}")
    logging.info(f"Unseen accents: {unseen}")
    logging.info(f"Dumping data from CV10 for these accents to folder {dump_folder}.")
    logging.info(f"Shuffling? {shuffle}")
    files = [f"pretrain_{pretrain_lang}"]
    files += [f"train_{lang}" for lang in [pretrain_lang] + seen]
    files += [f"test_{lang}" for lang in [pretrain_lang] + seen + unseen]
    writers = get_writers(dump_folder, files)
    data = json.load(open(f"{acc_folder}/{pretrain_lang}.json"))
    if shuffle is True:
        random.shuffle(data)
    dur = 0
    while dur / 3600 < pretrain_hours:
        obj = data.pop()
        dur += obj["duration"]
        write_objs([obj], writers[f"pretrain_{pretrain_lang}"])
    split(pretrain_lang, data, ratio, writers, False)
    for lang in seen:  # split seen data into train and test dicts
        split(
            lang,
            json.load(open(f"{acc_folder}/{lang}.json")),
            ratio,
            writers,
            shuffle,
        )
    logging.info(f"Creating test files for unseen accents: {unseen}")
    for lang in unseen:
        data = json.load(open(f"{acc_folder}/{lang}.json"))
        logging.info(f"Test set of lang {lang} comprises {len(data)} samples")
        write_objs(data, writers[f"test_{lang}"])
    for writer in writers.values():
        writer.close()


def get_writers(dump_folder, files):
    """Return file writers for each of the given files and test languages."""
    return {x: open(f"{dump_folder}/{x}.txt", "w") for x in files}


def split(lang, data, ratio, writers, shuffle):
    """Split the given data into train and test sets, according to the given ratio.
    Write directly the objects to their corresponding files. If shuffle is True,
    shuffle the data before splitting it. Return the number of data objects
    added to each dataset as a tuple (train, test)."""
    logging.info(f"Splitting data of accent {lang} into train and test sets")
    if shuffle is True:
        random.shuffle(data)
    split = round(len(data) * ratio)
    logging.info(f"Train set comprises {len(data[:split])} samples")
    logging.info(f"Test set comprises {len(data[split:])} samples")
    write_objs(data[:split], writers[f"train_{lang}"])
    write_objs(data[split:], writers[f"test_{lang}"])


def write_objs(objs, writer):
    """Write objects in the array as lines in the file. Return the number of samples written."""
    for obj in objs:
        writer.write(json.dumps(obj) + "\n")


def create_german_cv(cv_loc, root_folder):
    alphabet = "abcdefghijklmnopqrstuvwxyzäöü "
    acc_folder = "data/de/acc_split"
    train_folder = "data/de/train_split"
    mapping = {
        "Russisch Deutsch": "ru",
        "Französisch Deutsch": "fr",
        "Italienisch Deutsch": "it",
        "Österreichisches Deutsch": "at",
        "Schweizerdeutsch": "ch",
        "Deutschland Deutsch": "de",
        "Amerikanisches Deutsch": "us",
        "Britisches Deutsch": "gb",
        "Kanadisches Deutsch": "ca",
        "Griechisch Deutsch": "gr",
        "Polnisch Deutsch": "pl",
        "Deutschland Deutsch,Alemanischer Akzent": "de_al",
        "Deutschland Deutsch,Niederrhein": "de_ni",
    }
    get_accents(os.path.join(cv_loc, "de"), mapping, acc_folder, alphabet, root_folder)
    pretrain_hours = 350
    ratio = 0.8
    pretrain_lang = "de"
    seen = ["at", "ch"]
    unseen = ["ru", "fr", "it", "us", "gb", "ca", "de_al", "de_ni"]
    split_data(
        pretrain_lang, seen, unseen, ratio, pretrain_hours, acc_folder, train_folder
    )


def create_english_cv(cv_loc, root_folder):
    alphabet = "abcdefghijklmnopqrstuvwxyz '"
    acc_folder = "data/en/acc_split"
    train_folder = "data/en/train_split"
    mapping = {
        "United States English": "us",
        "England English": "uk",
        "India and South Asia (India, Pakistan, Sri Lanka)": "in",
        "Canadian English": "ca",
        "Australian English": "au",
        "German English,Non native speaker": "de",
        "Scottish English": "sc",
        "New Zealand English": "nz",
        "Irish English": "ie",
        "Southern African (South Africa, Zimbabwe, Namibia)": "za",
        "Northern Irish": "ni",
        "Filipino": "ph",
        "Hong Kong English": "hk",
        "Singaporean English": "sg",
    }
    # get_accents(os.path.join(cv_loc, "en"), mapping, acc_folder, alphabet, root_folder)
    pretrain_hours = 300
    ratio = 0.8
    pretrain_lang = "us"
    seen = ["uk", "ca", "in", "de"]
    unseen = ["hk", "sg", "au", "sc", "nz", "ie", "za", "ni", "ph"]
    split_data(
        pretrain_lang, seen, unseen, ratio, pretrain_hours, acc_folder, train_folder
    )


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"logs/create_dataset/{int(now())}.log", level=logging.INFO
    )
    cv_loc = "/ds/audio/cv-corpus-10.0-2022-07-04/"
    root_folder = "/ds/audio"
    create_english_cv(cv_loc, root_folder)
