import json
import editdistance
import string
import os


def evaluate(model, config, logger):
    """Evaluate the model on the test datasets with the WER. Dump the transcriptions
    and the WER of each sample."""
    wers, results = dict(), dict()
    folder = os.path.join(logger.log_dir, config.data.folder)
    for file in os.listdir(folder):
        results[file] = dict()
        audiofiles, transcripts = [], []
        for line in open(f"{folder}/{file}"):
            obj = json.loads(line)
            audiofiles.append(obj["audio_filepath"])
            transcripts.append(process_text(obj["text"]))
        predicted = model.transcribe(
            audiofiles,
            batch_size=config.data.config.batch_size,
            num_workers=config.data.config.num_workers,
        )[0]
        transcript_words = 0  # total no. of words in all transcripts
        n_edits = 0  # sum of the edits of all samples
        for i in range(len(audiofiles)):
            transcript_words += len(transcripts[i])
            edits = editdistance.eval(transcripts[i], predicted[i].split())
            n_edits += edits
            results[file][audiofiles[i]] = {
                "reference": " ".join(transcripts[i]),
                "hypothesis": predicted[i],
                "wer": edits / len(transcripts[i]),
            }
        wers[file] = {
            "avg_wer": n_edits / transcript_words,
            "n_words": transcript_words,
        }
    json.dump(wers, open(f"{logger.log_dir}/avg_wers.json", "w"))
    json.dump(results, open(f"{logger.log_dir}/results.json", "w"))
    logger.save()


def process_text(text):
    """Lowercase, remove punctuation. Return the string as a list of words (separated
    by white spaces). Numbers are already in written form."""
    no_punc = text.translate(str.maketrans("", "", string.punctuation))
    return no_punc.lower().split()
