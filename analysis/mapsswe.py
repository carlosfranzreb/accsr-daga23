"""
Implementation of the Matched-Pair Sentence Segment Word Error (MAPSSWE) test, which
determines whether two ASR models differ significantly.
"""


import os
import json
import logging  # TODO: implement logging
import numpy as np
from scipy.stats import norm


class MAPSSWE:
    def __init__(self, config, trainer):
        """
        The component config must contain two paths: `path_a` and `path_b` are the paths to
        the predictions of the two models for all test samples, as computed by the
        evaluation script (see `src/evaluate.py`).
        """
        self.config = config
        self.params = config.components.MAPSSWE
        # create dump file and add headers
        self.dump_file = os.path.join(trainer.logger.log_dir, "mapsswe.txt")
        with open(self.dump_file, "w") as f:
            f.write("Test set P-value\n")

    def run(self):
        """
        Compute the MAPSSWE test statistic for two ASR models. We compute the
        difference in the number of errors for each sentence in the test set, and then
        compute the mean and variance of this difference. The test statistic is the
        mean divided by the standard deviation. The test statistic is then compared to
        the normal distribution; if the test statistic is greater than an arbitrary
        threshold, then we reject the null hypothesis that the two models are the same.
        """
        preds_a = json.load(open(self.params.path_a))
        preds_b = json.load(open(self.params.path_b))
        all_diffs = np.array([])
        # iterate over the test sets
        for test_set in preds_a:
            if test_set not in preds_b:
                print(f"`{test_set}` not in {self.params.path_a}; skipping")
            diffs = list()
            # iterate over the samples in the test set
            for sample_id in preds_a[test_set]:
                # compute the number of edits for each prediction with the WERs
                ref_len = len(preds_a[test_set][sample_id]["reference"].split())
                errors_a = preds_a[test_set][sample_id]["wer"] * ref_len
                errors_b = preds_b[test_set][sample_id]["wer"] * ref_len
                # compute the difference in the number of errors
                diffs.append(errors_a - errors_b)
            # add the diffs to the total diffs
            diffs = np.array(diffs)
            all_diffs = np.concatenate((all_diffs, diffs))
            # compute the p-value and dump it
            with open(self.dump_file, "a") as f:
                f.write(f"{test_set} {compute_p(diffs):.4f}\n")
        # compute the p-value for all diffs and dump it
        with open(self.dump_file, "a") as f:
            f.write(f"all\t{compute_p(all_diffs):.4f}\n")


def compute_p(diffs):
    """
    Compute the test statistic for the given differences in the number of errors.
    """
    w = np.sqrt(diffs.shape[0]) * np.mean(diffs) / np.std(diffs)
    return 1 - norm.cdf(w)
