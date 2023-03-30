# Multi-task and adversarial learning for accented speech recognition

This framework can be used to an ASR model together with an accent classifier. The classifier is placed on top of an encoder block, receiving its output as input. The models are either trained jointly in a multi-task learning fashion, or adversarially, where a gradient reversal layer is prepended to the classifier. Our experiments are performed on the [Conformer Transducer](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_transducer_large), but there are other models that use the same code base; they can also be used here. The purpose of this framework is explained in more detail in the accompanying publication, cited at the end of this document.

## Results from the DAGA paper

The experiments behind the results discussed in the aforementioned paper can be found in the following notebooks:

- The optimization experiments (Figure 2) can be found in `scripts/results_analysis/wers/weights_vs_lrs.ipynb`.
- The WERs achieved for the different branching points can be found in `scripts/results_analysis/wers/branches.ipynb`.
- The WERs achieved with the different techniques (Table 2) can be found in `scripts/results_analysis/wers/noac_vs_al.ipynb`.
- The EMD distances from Table 3 come from `scripts/results_analysis/encoder_dists/distance_analysis.ipynb`.
- The statistical significance of adversarial training against normal training according to the MAPSSWE test can be found in `logs/de/analysis/version_12`.

## Installation

We use the [NeMo toolkit](https://github.com/NVIDIA/NeMo) from NVIDIA. NVIDIA provides [Docker images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) where all the required dependencies are already installed. These images contain all the dependencies we require except for two cases:

1. The `scripts/create_dataset.py` script, which is used to create the data file required by NeMo, uses [swifter](https://github.com/jmcarpenter2/swifter) for efficiency purposes.
2. The analysis script `analysis/errors.py` uses [Levenhstein](https://github.com/maxbachmann/python-Levenshtein) to compute the edit operations on the transcripts.

These two scripts have nothing to do with NeMo and have few dependencies. They can be run locally in virtual environments. When training an ASR model, the NeMo image should be used.

## Creating the data files

NeMo requires the data to be defined in files with a certain structure. `scripts/create_dataset.py` does that for Common Voice, splitting the data into seen and unseen accents. Seen accents have both train and test files, whereas unseen accents only have test files, meaning that they are not seen during training. The standard accent may also have a pre-train file, which is used to fine-tune the ASR model before performing the joint learning with the classifier.

## Types of jobs

The framework can run two types of jobs:

1. **Experiment:** a training or evaluation job.
2. **Analysis:** run various kinds of analysis scripts on an ASR model.

### Experiments

Experiments are run either to train or evaluate a model. We support four kinds of experiments. Which one to perform is defined in the `ensemble.action` parameter of the configuration file.

1. `train_asr`: train the ASR model, without an accent classifier.
2. `train_ac`: the classifier is appended to the encoder of the ASR model, but only the classifier is trained. The ASR model is frozen and only the forward pass of its encoder is run.
3. `train`: both models are trained jointly, either in a multi-task fashion or adversarially.
4. `evaluate_asr`: evaluating an existing ASR model. The predicted transcriptions and their corresponding word error rates (WERs) are computed and stored for all test files.

### Experiment configuration

Regardless of what experiment we want to perform, the configuration is similar. We first need to define the job type and the language, which are used to decide where to store the experiment logs.

```yaml
job: experiment
language: en
```

Then, we configure the training and test data, the PyTorch Lightning trainer and the optimizer by referencing the files where this information is stored. You can look at the files mentioned below to see what they include.

```yaml
data_file: config/data/default.yaml
trainer_file: config/trainer/default.yaml
```

We then define the ensemble model, which combines the ASR model and the classifier. It defines on which encoder block to place the classifier, the weight of each model when computing the final loss, the mode in which they are connected (whether a gradient reversal layer is added, and which kind), and what kind of experiment should be performed (the action).

```yaml
ensemble:
  branch: 10
  ac_weight: .1
  asr_weight: .9
  mode: DAT
  action: train
```

We then define the ASR model with the class name, pre-trained checkpoint provided by NVIDIA and optionally our own checkpoint.

```yaml
asr:
  classname: EncDecRNNTBPEModel
  pretrained: stt_de_conformer_transducer_large
  ckpt: null
```

Finally, we define the accent classifier. It requires a class name where the PyTorch neural network is defined (within the `src.accent_classifiers` directory), and optionally a checkpoint to initialize its weights. We also need to define how many accents we are considering, and whether the classifier is binary. Binary classifiers group all different accents into one category. A dropout rate can be optinally defined here.

```yaml
ac:
  classname: AC
  ckpt: null
  n_accents: 2
  binary: true # whether the classifier is binary or not
  dropout: null
```

### Analysis

Analysis jobs are meant to analyse the behavior of an ASR model. To run an analysis job, we define `job: analysis` in the configuration file. The language is also required, as well as the trainer and data files. The analysis components are defined under the `components` variable.

```yaml
job: analysis
language: en
data_file: config/data/default.yaml
trainer_file: config/trainer/default.yaml
components:
    <insert component configs>
```

We have currently implemented four analysis scripts, which we now describe.

#### Visualize encoder representations

`analysis/encodings.py` computes the outputs of all ASR encoder blocks for a number of samples of each accent, reduces their dimensionality with t-SNE, computes the distances among accent representations and, if the representations have been reduced to two dimensions, draws and stores the plots for all encoder blocks. Its configuration looks as follows:

```yaml
EncoderViz:
  n_samples: 100
  tsne:
    learning_rate: auto
    init: pca
```

`n_samples` defines how many samples of each accent are used. The first samples are used without shuffling, so that results of different analysis runs can be compared. `tsne` may comprise any argument from Scikit-learn's [t-SNE class](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). We compute three distances among the accent representations:

1. Cosine and Euclidean distances between all pairs of mean accent representations.
2. The earth mover's distance between all pairs of accent distributions.

#### Computing gradient sizes at the branching point

The optimization of multi-task learning settings can be tricky. One important aspect to consider is how large are the gradients that arrive at the branching point, where the two models diverge. `analysis/gradients.py` computes the ratio between the gradient sizes of the two models for all combinations of the given values for the branching point, model checkpoints and loss weights. Its configuration looks like this:

```yaml
GradChecker:
  branches: [6, 10, 16]
  ac_weights: [.1]
  asr_weights: [.9]
  trained_asr: [true]
  trained_ac: [true, false]
```

The checkpoints for the ASR model and the accent classifiers must be defined outside the components config, on the upper-most level (outside the `components` subconfig). If `trained_asr` is true, an ASR checkpoint must be provided; if `trained_ac` is true, a classifier checkpoint must be provided for all the defined branches. For the configuration above, we would add the following config to the upper-most level:

```yaml
ckpt_asr: logs/ensemble/train/binary/b15/DAT/version_3
ckpt_ac:
  b6: logs/ensemble/train/binary/b6/DAT/version_0/checkpoints/last.ckpt
  b10: logs/ensemble/train/binary/b10/DAT/version_0/checkpoints/last.ckpt
  b16: logs/ensemble/train/binary/b16/DAT/version_2/checkpoints/last.ckpt
```

#### Error analysis

`analysis/errors.py` goes through the predicted and reference texts output by an evaluation experiment and dumps how often each kind of error (deletion, addition or replacement) was made for each word, grouped by accent. As configuration, it only requires the path to a `results.json` file, which is where evaluation experiments store the true and predicted transcripts of all samples:

```yaml
  ErrorCounter:
    path: logs/de/asr/evaluate/version_0/results.json
```

#### Statistical significance (MAPSSWE)

`analysis/mapsswe.py` computes the statistical significance of the difference between two ASR models. It implements the Matched-Pair Sentence Segment Word Error (MAPSSWE), which consists of computing the difference in the number of errors made by each model for each sample, and then computing the mean and variance of these differences. The test statistic is the mean divided by the standard deviation. The test statistic is then compared to the normal distribution; if the test statistic is greater than an arbitrary threshold, then we reject the null hypothesis that the two models are the same. The threshold is usually a very small number, around 0.001. This script requires the paths to the `results.json` files produced by the evaluation experiments run on the two ASR models:

```yaml
MAPSSWE:
  path_a: logs/de/asr/evaluate/version_1/results.json
  path_b: logs/de/asr/evaluate/version_2/results.json
```

## Running an experiment

Experiments are defined with a config file and run with `python run.py`. This script receives four arguments:

1. `--accelerator`: accelerator used (GPU, CPU, etc.). Defaults to `gpu`.
2. `--config`: location of the config file. This argument is required.
3. `--devices`: total number of devices across nodes. This argument is used to adjust the batch size; the config parameter defines the batch size per device.
4. `--debug`: whether we are debugging; it ensures that the distributed training is not initialized.

Analysis jobs are run with `python analyse.py`, which receives the same arguments.

## Ensemble modes

There are several kinds of ensemble modes available, which are defined in the config file with `ensemble.mode`

- `MTL`: the gradients of the AC is backpropagated normally to the ASR encoder.
- `DAT`: the gradients of the AC is negated before being backpropagated to the ASR encoder.
- `OneWayDAT`: only the gradients of accented samples are negated; gradients of standard samples are backpropagated normally.
- `SwitchDAT`: there are as many ACs as there are accents; Each batch is passed to all ACs, but only the gradients of the AC assigned to the batch's accent are backpropagated to the ASR model.

### Additional configurations for SwitchDAT

SwitchDAT requires additional configuration, other than the config that is required by all experiments. The classifier configuration `ac` requires two additional parameters:

1. `mode`: the mode of the individual classifiers: may be `MTL`, `DAT` or `OneWay`.
2. `optim`: optimizer configuration to train the classifiers

## Citation

TODO