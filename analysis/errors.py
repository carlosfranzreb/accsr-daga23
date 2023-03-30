"""
Script that goes through the predicted and reference texts, and evaluates which kind of
error (deletion, addition, replacement) was made for which word. In the end, the script
should output a text file with three columns: word, kind of error and frequency. The
predicted and reference texts come from the results of an evaluation run (see
src/evaluate.py).
"""


import json
import os

from Levenshtein import editops


class ErrorCounter:
    def __init__(self, config, trainer):
        self.config = config
        self.params = config.components.ErrorCounter
        self.dump_dir = trainer.logger.log_dir

    def run(self):
        # Read the JSON file
        with open(self.params.path) as f:
            results = json.load(f)

        # Count the frequency of each type of error
        error_counts = {}
        replacements = {}
        for filename, samples in results.items():
            error_counts[filename] = dict()
            replacements[filename] = dict()
            # Count the errors for each sample of the file
            for sample_data in samples.values():
                ref_text = sample_data["reference"]
                pred_text = sample_data["hypothesis"]
                errors = compare_texts(ref_text, pred_text)
                for error in errors:
                    word, error_type = error[0], error[-1]
                    if word not in error_counts[filename]:
                        error_counts[filename][word] = {
                            "occurrences": count_occurrences(samples, word),
                            error_type: 1,
                        }
                    elif error_type not in error_counts[filename][word]:
                        error_counts[filename][word][error_type] = 1
                    else:
                        error_counts[filename][word][error_type] += 1
                    if error_type == "replace":
                        if word not in replacements[filename]:
                            replacements[filename][word] = {error[1]: 1}
                        elif error[1] not in replacements[filename][word]:
                            replacements[filename][word][error[1]] = 1
                        else:
                            replacements[filename][word][error[1]] += 1

            # Sort the error counts by frequency
            for word, error_types in error_counts[filename].items():
                error_counts[filename][word] = dict(
                    sorted(error_types.items(), key=lambda item: item[1], reverse=True)
                )

        # Dump all error counts
        json.dump(
            error_counts,
            open(os.path.join(self.dump_dir, "errors.json"), "w"),
            indent=4,
        )
        json.dump(
            replacements,
            open(os.path.join(self.dump_dir, "replacements.json"), "w"),
            indent=4,
        )


def compare_texts(ref_text, pred_text):
    """
    Compute the edit operations necessary to convert the reference text into the
    predicted text. Return a list of tuples, where each tuple contains the word and the
    type of error (deletion, insertion, substitution).
    """
    ref_words = ref_text.split()
    pred_words = pred_text.split()
    edits = list()
    for edit in editops(ref_words, pred_words):
        op = edit[0]
        if op == "insert":
            edits.append([pred_words[edit[2]], op])
        elif op == "replace":
            edits.append([ref_words[edit[1]], pred_words[edit[2]], op])
        else:
            edits.append([ref_words[edit[1]], op])
    return edits


def count_occurrences(samples, word):
    """
    Count the number of occurrences of the word in the reference texts of the samples.
    """
    count = 0
    for sample in samples.values():
        for sample_word in sample["reference"].split():
            if sample_word == word:
                count += 1
    return count
