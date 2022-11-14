# Mutual Information Alleviates Hallucinations in Abstractive Summarization
Authors: Liam van der Poel, [Ryan Cotterell](https://rycolab.io/),  [Clara Meister](https://cimeister.github.io/)

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [To install fairseq on Euler server](#to-install-fairseq-on-euler-server)
- [Experiments with the batch system](#experiments-with-the-batch-system)
- [Preparing XSUM dataset](#preparing-xsum-dataset)
	- [Download raw XSUM dataset](#download-raw-xsum-dataset)
	- [Download test/validation/train split JSON](#download-testvalidationtrain-split-json)
	- [Installing Moses](#installing-moses)
	- [Install BPE encoder](#install-bpe-encoder)
	- [To install files2rouge](#to-install-files2rouge)
	- [Perform preprocessing](#perform-preprocessing)
	- [Move data to the server](#move-data-to-the-server)
- [Train a Summarization model](#train-a-summarization-model)
	- [Training a model from scratch](#training-a-model-from-scratch)
	- [Using a pretrained model](#using-a-pretrained-model)
- [Evaluate a Summarization model](#evaluate-a-summarization-model)
	- [To generate summaries on the test set](#to-generate-summaries-on-the-test-set)
	- [Computing ROUGE scores](#computing-rouge-scores)
- [Train a language model](#train-a-language-model)
- [Evaluate a language model](#evaluate-a-language-model)
- [Hyperparameter selection](#hyperparameter-selection)
	- [MMI Decoding](#mmi-decoding)
		- [Theoretical approach](#theoretical-approach)
		- [Implementation](#implementation)
	- [Generating on validation vs. test sets](#generating-on-validation-vs-test-sets)
- [Modifying fairseq](#modifying-fairseq)
	- [Displaying probabilities for language and summarizer models separately](#displaying-probabilities-for-language-and-summarizer-models-separately)
	- [Displaying entropy](#displaying-entropy)
		- [For reference summaries](#for-reference-summaries)
		- [For generated summaries](#for-generated-summaries)
- [XSum Hallucination Annotations](#xsum-hallucination-annotations)
	- [Post-processing labels](#post-processing-labels)
		- [BART specific instructions](#bart-specific-instructions)
	- [Computing token level entropy](#computing-token-level-entropy)
	- [Statistics for entropy of hallucinated tokens](#statistics-for-entropy-of-hallucinated-tokens)
	- [Statistics for probability of hallucinated tokens](#statistics-for-probability-of-hallucinated-tokens)
	- [Selecting optimal lambda value](#selecting-optimal-lambda-value)
- [Entropy threshold MMI decoding](#entropy-threshold-mmi-decoding)

<!-- /TOC -->
## To install fairseq on Euler server

Ensure any previously installed versions of fairseq are removed with `python -m pip uninstall fairseq`. Then fork the repository. Instructions are based on [general fairseq instructions](https://github.com/pytorch/fairseq#requirements-and-installation) and [euler specific instructions](https://github.com/jasonwei20/fairseq/blob/master/jason-lm-wt103/run-wikitext.MD).

```
git clone https://github.com/VanderpoelLiam/fairseq
module load gcc/6.3.0 python_gpu/3.8.5 hdf5 eth_proxy
python -m venv fair_env
source fair_env/bin/activate
cd fairseq
PYTHONPATH=$(which python)
$PYTHONPATH -m pip install --upgrade pip
$PYTHONPATH -m pip install --editable ./
```

Each time login to server need to run:
```
module load gcc/6.3.0 python_gpu/3.8.5 hdf5 eth_proxy
source fair_env/bin/activate
PYTHONPATH=$(which python)
```

I also tried to install apex (but it failed due to Cuda versions not matching):
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

## Experiments with the batch system

To run Euler in interactive mode with sufficient specs for most tasks:
`bsub -I -n 2 -R "rusage[mem=2048]" -R "rusage[ngpus_excl_p=1]"`

## Preparing XSUM dataset
### Download raw XSUM dataset

Download dataset of 237018 articles.
`wget http://bollin.inf.ed.ac.uk/public/direct/XSUM-EMNLP18-Summary-Data-Original.tar.gz`.

Then extract the `bbc-summary-data` folder.
`tar -xzf XSUM-EMNLP18-Summary-Data-Original.tar.gz`

See [here](https://github.com/EdinburghNLP/XSum/issues/9) and [here](https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/bart/README.summarization.md) for more information.

### Download test/validation/train split JSON

`wget https://github.com/EdinburghNLP/XSum/blob/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json`

### Installing Moses
Ensure that the Moses tokeniser is installed with:
`git clone https://github.com/moses-smt/mosesdecoder.git`

See [here](http://www.statmt.org/moses/?n=Development.GetStarted) for dependencies.

### Install BPE encoder
Follow instructions [here](https://github.com/glample/fastBPE) to install fastBPE.

### To install files2rouge
Based on [files2rouge](https://github.com/pltrdy/files2rouge), do the following:
```
$PYTHONPATH -m pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git     
cd files2rouge
$PYTHONPATH setup_rouge.py
$PYTHONPATH setup.py install
```

### Perform preprocessing
Run `python src/preprocess.py` with the desired arguments to preprocess the data. See `python src/preprocess.py -h` for more information on the possible datasets. Sample usage is `python src/preprocess.py --all` to run the preprocessing for all the datatasets. The resulting data is stored in the `data` directory.

The sample dataset consists of `4/2/2` examples in train/valid/test respectively. It is useful for debugging purposes to not work with the full XSUM dataset.

### Move data to the server
Log into server with `sftp`. Ensure local and remote directories are the same then run `put -r .` to move over all files. Do this for the following folder in the `data` directory: `xsum-summarizer`,  `xsum-lang`, `xsum-lang-full`.

## Train a Summarization model
### Training a model from scratch
Based on [this](https://github.com/pytorch/fairseq/tree/main/examples/translation#training-a-new-model).

Use the `src/train_command.py` script to generate the `fairseq-train` commands and copy them to the clipboard. Paste the result on Euler to run. Basic usage is:
```
python src/train_command.py 1
```
to run experiment `1` and log to `logs/experiments/train_1.log`.

Run `python src/train_command.py -h` for more information on the other parameters.

### Using a pretrained model
An alternate approach using a BART pre-trained model is explained [here](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md) and [here](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)



To download the pretrained model:
```
cd checkpoints/summarization_model/
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz
tar -xzvf bart.large.xsum.tar.gz
rm bart.large.xsum.tar.gz
cd ../..
```

Ensure the BART model preprocessing was run:
```
python src/preprocessing/preprocess.py --bart --full --sample
```

To generate with the pretrained model:
```
bsub -J generate_bart \
-o logs/experiments/generate_bart.log \
-W 60 -n 4 -R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
fairseq-generate \
data/xsum-summarizer-bart \
--path checkpoints/summarization_model/bart.large.xsum/model.pt \
--batch-size 8 --beam 5 \
--truncate-source \
--skip-invalid-size-inputs-valid-test \
--bpe gpt2 \
```

## Evaluate a Summarization model
### To generate summaries on the test set
Similarly,

Use the `src/generate_command.py` script to generate the `fairseq-generate` commands. Basic usage is:
```
python src/generate_command.py 1 --wait_train
```
to run experiment `1` and log to `logs/experiments/generate_1.log`.

Run `python src/generate_command.py -h` for more information on the other parameters

### Computing ROUGE scores
Running `src/score_generate.sh 1` extracts the target/hypothesis sentences from `logs/experiments/generate_1.log` to the `logs/rouge/generate_1` directory and removes tokenization and BPE. The full ROUGE scores are saved to the `logs/rouge/generate_1/score` file and the `F_1` scores output to the terminal.

## Train a language model
Based on [this](https://github.com/pytorch/fairseq/blob/main/examples/language_model/README.md).

Use the `src/train_language_command.py` script to generate the `fairseq-train` commands and copy them to the clipboard. Paste the result on Euler to run. Sample usage is:
```
python src/train_language_command.py \
--dataset standard \
--full \
--update_freq 32 \
--lr 0.0002
--restore  
```

See `src/train_language_command.py -h` for more details on the parameters.

## Evaluate a language model
Similarly,

Use the `src/generate_command.py` script to generate the `fairseq-generate` commands. Sample usage is:
```
python src/eval_language_command.py.py \
--dataset standard \
--full \
--update_freq 32 \
--lr 0.0002
--wait  
```
See `src/eval_language_command.py -h` for more details on the parameters.

## Hyperparameter selection
### MMI Decoding
#### Theoretical approach
`\lambda` should take a value in `[0, 1]`. I found the model would generate nonsense close to 1. My prior is therefore shifted towards zero. So I pick 10 values on a log scale in `(0, 1)` (i.e. `2^-1, 2^-2, ..., 2^-9, 2^-10`).

`\gamma` should take a value in `[0, N_t]` where `N_t` is the length of hypothesis `t`.  But this would lead to a dependence on `t`. Instead I estimate `M`, the average value of `N_t` for all hypothesis generated by our summarizer (we count tokens not words). I then pick 10 values evenly in `(0, M)`. `M` was computed to be `26.90` so we select gamma from `[2, 4, 7, 9, 12, 14, 17, 19, 22, 24]`.

`\mu` scales the influence of `log(N_t)` which is on average `log(26.90) = 3.292`. Our prior is that we want all terms to be around the same order of magnitude. The average value of `log(y|x)` is `-1.102`, so picking `\mu` to be one of (`2^-1, 2^-2, ..., 2^-9, 2^-10`) feels appropriate.

ROUGE-1 RL scores are used as the validation metric.

#### Implementation
Use the `src/lambda_gridsearch_command.py` script to generate the batch command as for both train and generate. For example:
```
python src/lambda_gridsearch_command.py 8 --lang_full
```
runs model number `8` with the full language model. See `python src/lambda_gridsearch_command.py -h` for help.

The logs are saved to `logs/hyperparameters/lang_full/8`. In the log directory, we generate `batch.log` which is the batch command logs. Then, for each lambda of value `x`, we create two files: `x.log`, the results of `fairseq-generate` and `score_x` the corresponding rouge scores.

### Generating on validation vs. test sets
By default we generate on the test set with `fairseq-generate`. However for hyperparameter selection we need to generate on the validation set. This is done by adding the parameter `--gen-subset "valid"` to `fairseq-generate`.

## Modifying fairseq
I created a new branch called `liam` to make modifications to fairseq. I use the `lstm` and `lstm_lm` architectures as well as the `xsum-summarizer-samples` and `xsum-lang-samples` datasets. The architectures and datasets are chosen to be fast to train and generate with. We also create the `test-project` directory with the following structure:
```
test-project
├── checkpoints
│   ├── summarization_dummy/
│   └── lang_dummy/
└── data
    ├── xsum-summarizer-samples/[...]
    └── xsum-lang-samples/[...]
```

Where the `xsum-*` directories contain all the data copied over from `master-thesis/`.

Then to initialize the models, we train them for one iteration. For the summarization model run:
```
bsub -I -W 10 -n 4 -R "rusage[mem=4096]" \
fairseq-train data/xsum-summarizer-samples \
  --arch lstm \
  --save-dir checkpoints/summarization_dummy \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 4096 \
  --max-update 1 \
  --max-epoch 1
```

and for the language model run:
```
bsub -I -W 10 -n 4 -R "rusage[mem=4096]" \
fairseq-train data/xsum-lang-samples \
  --task language_modeling \
  --arch lstm_lm \
  --save-dir checkpoints/lang_dummy \
  --optimizer adam --lr 0.005 --lr-shrink 0.5 \
  --max-tokens 4096 \
  --max-update 1 \
  --max-epoch 1
```

Then to generate with `\lambda = 1` run:
```
bsub -I -W 10 -n 4 -R "rusage[mem=4096]" -R "rusage[ngpus_excl_p=1]" \
fairseq-generate \
data/xsum-summarizer-samples \
--gen-subset "valid" \
--path checkpoints/summarization_dummy/checkpoint_best.pt \
--batch-size 16 --beam 5 --truncate-source \
--skip-invalid-size-inputs-valid-test \
--lm-path checkpoints/lang_dummy/checkpoint_best.pt \
--lm-weight -1
```

With an unmodified `fairseq-generate` this produces for a single article and output of the form (I add `[...]` for brevity):
```
S-1	Transport Minister Juan Mol@@ in@@ ar said [...]
T-1	One of Mexico &apos;s biggest airlines , Mex@@ ic@@ ana de [...]
H-1	5.0183587074279785	roadside Venezuel@@ released Venezuel@@ [...]
D-1	5.0183587074279785	roadside Venezuel@@ released Venezuel@@ [...]
P-1	2.4488 2.6601 2.9603 3.0198 3.1490 3.3097 3.4029 3.4311 [...]
```


### Displaying probabilities for language and summarizer models separately
The MMI decoding objective has the form: `$\log p(y | x) - \lambda \log p(y)$`.

`P-1` is an array where the `i`'th  entry `P-1[i]` corresponds to: `$\log p(y_i | x, y_{<i}) - \lambda \log p(y_i | y_{<i})$`. The sum of these two probability distributions is then normalized. Hence it is not the case that `P-1 = P_SM-1 + $\lambda$ P_LM-1`

Our modifications produce two additional arrays, `P_SM` and `P_LM` which correspond to `$\log p(y | x)$` and `$\log p(y)$` respectively. Therefore, `P_SM-1[i]` corresponds to `$\log p(y_i | x, y_{<i})$` and `P_LM-1[i]` corresponds to `$\log p(y_i | y_{<i})$`.

This looks like:
```
[...]
P-1	2.4488 2.6601 2.9603 3.0198 3.1490 3.3097 3.4029 3.4311 [...]
P_SM-1	-8.3416 -8.3461 -7.9928 -7.9528 -7.8088 -7.7310 -7.6555 [...]
P_LM-1	-10.7904 -11.0063 -10.9531 -10.9726 -10.9579 -11.0407 [...]
```

As a sanity check we can see that for `i=0`, we have `2.4488 = P-1[0] = P_SM-1[0] - 1 * P_LM-1[0] = -8.3416 - 1 * -10.7904 = 2.4488`.

### Displaying entropy
#### For reference summaries
The `sequence_scorer.py` and `generate.py` files in fairseq were modified to compute token level entropy values on the reference summaries. Running:
```
fairseq-generate \
data/xsum-summarizer-samples \
--gen-subset "valid" \
--path checkpoints/summarization_model/11/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--score-reference \
--truncate-source \
--lm-path checkpoints/lang_full/checkpoint_best.pt \
--lm-weight -1
```

should result in an output of the form:
```
[...]
T-1	One of Mexico &apos;s biggest airlines , Mex@@ ic@@ ana de Avi@@ ac@@ [...]
H-1	-13.054492950439453	One of Mexico &apos;s biggest airlines , Mex@@ [...]
P-1	-16.4506 -1.5589 -8.1095 -1.8594 -8.5545 -13.3892 -7.5062 -17.5240 [...]
P_SM-1	-6.4942 -0.1495 -0.7367 -0.0593 -1.9511 -1.6154 -1.9322 -4.4989 [...]
P_LM-1	-11.6483 -19.5079 -18.5996 -14.2160 -12.6121 -10.2308 -10.8821 [...]
ENT_LANG-1	3.3440 1.9895 0.0965 5.3574 6.1091 2.9450 7.1573 0.2616 [...]
ENT-1	5.9865 1.0240 2.1275 0.4982 3.9551 3.9361 2.8885 7.6781 2.3818 [...]
```

#### For generated summaries
For the generated summaries run:
```
fairseq-generate \
data/xsum-summarizer-samples \
--gen-subset "valid" \
--path checkpoints/summarization_model/11/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--truncate-source \
--lm-path checkpoints/lang_full/checkpoint_best.pt \
--lm-weight -1
```
should result in an output of the form:
```
[...]
H-1	16.198698043823242	an@@ L@@ El@@ P@@ de ana airline Mex@@ W@@ Dor@@[...]
P-1	-42.1291 -14.4789 -19.7052 -12.0393 -25.7978 -12.9992 -19.6963 [...]
P_SM-1	-12.4465 -10.1111 -8.1167 -9.6915 -11.2029 -8.1613 -9.7413 [...]
P_LM-1	-2.0259 -19.8412 -19.6330 -20.9502 -4.7277 -13.5809 -12.2417 [...]
ENT_LANG-1	6.1471 8.4263 7.7026 8.7397 4.8474 4.7859 4.6337 0.8963 [...]
ENT-1	8.0262 7.6571 6.7409 7.1623 5.3124 3.4494 7.7405 6.1100 6.6523 [...]
```

### Displaying ranking
At at each position in the sequence we select a particular token to output, not necessarily the token with highest probability. The ranking of a token is the ranking of its probability with respect to the distribution at our current position. E.g. if we pick the 3rd most probable token then it has rank 3.

We modify the `sequence_scorer.py` and `generate.py` files in fairseq to compute token level rankings:
```
fairseq-generate \
data/xsum-summarizer-samples \
--gen-subset "valid" \
--path checkpoints/summarization_model/standard/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--score-reference \
--truncate-source
```
The output should now contain a line of the form:
```
[...]
T-1	One of Mexico &apos;s biggest airlines , Mex@@ ic@@ ana de Avi@@ ac@@ [...]
H-1	-3.9946937561035156	One of Mexico &apos;s biggest airlines , Mex@@ [...]
P-1	-6.4942 -0.1495 -0.7367 -0.0593 -1.9511 -1.6154 -1.9322 -4.4989 [...]
RANK-1	7882 3780 13751 111 3154 6732 38 9051 13328 2859 10583 6880 17216 [...]
```

As a sanity check, we shift the tokens by 1 in either direction to see that this leads to lower probabilites/worse rankings. For example, with the sequence `[BOS, I, went, home, EOS]` we calculate the probability/ranking of `went` using the `1st` distribution over tokens. This sanity check looks at what happens when we use the `0th` distribution and the `2nd` distribution.

Shifting tokens by `+1`:
```
[...]
P-1	-21.2402 -16.3707 -11.4401 -16.5040 -18.6815 -8.1419 -13.4884 [...]
RANK-1	25465 807 21394 4734 210 15289 3960 7574 30588 3537 170 47495 [...]
```

Shifting tokens by `-1`:
```

P-1	-15.8094 -15.9375 -20.9515 -9.2937 -9.1373 -12.4901 -12.4808 -6.1468 [...]
RANK-1	307 578 30343 472 8405 42258 12578 29894 5758 26202 34352 26376 [...]
```

If we consider the token `Mexico`:

token shift  | `-1` | `0` | `+1` |
--- | --- | --- | --- |
Log probability     | -20.9515 | -0.7367 | -11.4401 |
Ranking             | 30343    | 13751   | 21394    |

### Entropy threshold MMI decoding
Based on the statistics for the first hallucinated token in a sequence, we can see that high entropy is correlated with the start of a hallucination. So the idea is modify fairseq to have entropy above a threshold trigger MMI decoding.

For actual generation we need to modify the test set to remove the 500 test examples that were used to pick the threshold value to avoid information leakage. This is the dataset `xsum-summarizer-no-500`.

The below code snippet is to check that the thresholding is behaving as expected, only triggering when the token entropy is higher than the threshold:
```
fairseq-generate \
data/xsum-summarizer-samples \
--path checkpoints/summarization_model/standard/checkpoint_best.pt \
--batch-size 16 --beam 5 \
--gen-subset "valid" \
--truncate-source \
--log-format none \
--lm-path checkpoints/language_model/standard/checkpoint_best.pt \
--lm-weight -100 \
--ent-threshold 4
```

## XSum Hallucination Annotations
Data from Maynez et al. [here](https://github.com/google-research-datasets/xsum_hallucination_annotations).

Zhou et al. already postprocessed the data from Maynez et al. [here](https://github.com/violet-zct/fairseq-detect-hallucination/tree/master/eval_data/xsum/Gold). This data is stored under `data/xsum-hallucination-raw/`:
```
data
└── xsum-hallucination
    ├── Gold.docid
    ├── Gold.label
    ├── Gold.ref
    ├── Gold.source
    └── Gold.target
```

### Post-processing labels
The labels in `Gold.label` are for the summaries in `Gold.target`. However we want labels for the result of the `Gold.ref` summaries after applying tokenisation and BPE--log-interval 10000000000 \. This is done as follows (base directory for scripts is `src/hallucination_labelling/`):

1. Create `data/Xsum-hallucination-split.json` containing all the id's from `data/xsum-hallucination-raw/Gold.docid` in the test split and nothing in the train/val splits.
2. Run the preprocessing on this split to get all the `test.*` files and the two `dict.*` files in `data/xsum-hallucination/`. We want labels for the sentences in `test.bpe.target`.
3. Next, run `align_labels.py` to get the labels for most of `test.bpe.target` and save to `test.label`. These labels are extracted from `Gold.label` and aligned with `Gold.target`.
4. Missing labels are indicated by a `?` and these cases are processed manually with a helper in the same script.
5. I found that cases where `1 0 1` appeared where often mistakes, so I also process these cases manually

#### BART specific instructions
For the BART dataset in `data/xsum-hallucination-bart`, the issue is that `test.bpe.target` is encoded hence looks like:
```
3198 2776 837 838 11514 3869 28057 764 [...]
```
So we need to decode these numbers using the encoding in `data/bart.encoder.json`. This is done by running the following in python:
```
import json
import os

with open('data/bpe/bart/encoder.json') as f:
    encoder = json.load(f)

encoder_keys = list(encoder.keys())
encoder_values = list(encoder.values())
base_dir = "data/xsum-hallucination-bart/"
filename = base_dir + "test.bpe.target"
with open(filename) as f:
    lines = [line.rstrip() for line in f]

new_filename = base_dir + "test.bpe.target.encoded"
os.rename(filename, new_filename)

new_lines = []
for line in lines:
		tokens = list(map(int, line.split()))
		inds = list(map(encoder_values.index, tokens))
		decoded_toks = [encoder_keys[i] for i in inds]
		cleaned_toks = []
		for tok in decoded_toks:
				if (tok == "\u0120"):
						cleaned_toks.append(tok)
				else:
						clean_tok = tok.replace("\u0120","")
						cleaned_toks.append(clean_tok)
		new_line = " ".join(cleaned_toks)
		new_lines.append(new_line)
		with open(filename, 'a') as f:
				print(new_line, file=f)
```
Then follow the same instructions as above for alignment.

### Computing token level entropy
For all the following scripts we need to specify which dataset we are working with `[standard, bart]` using the `--dataset` parameter. by default we use `standard`

Run:
```
python src/hallucination_labelling/compute_token_level_entropy_command.py
```
and paste the results on Euler to compute the token level entropy values and log them to `logs/experiments/token_level_entropy_standard`

Then run:
```
python src/hallucination_labelling/align_data.py
```

in order to extract the entropy scores to `test.entropy.sm` and `test.entropy.lm` and the probability scores to `test.prob.sm` and `test.prob.lm` in the directory `data/xsum-hallucination/`. This data is now aligned with the `test.label` hallucination labels in the same directory.

### Statistics for entropy of hallucinated tokens
Given the data in `test.label`, `test.entropy.sm` and `test.entropy.lm` in `data/xsum-hallucination/` we want to generate statistics on this data.

Run `python src/hallucination_labelling/entropy_stats.py` in order to get statistics and distribution information comparing the entropy values for various token labellings.

### Statistics for probability of hallucinated tokens
Likewise we want statistics on `test.prob.sm` and `test.prob.lm`.

Run `src/hallucination_labelling/probability_stats.py` in order to get statistics and distribution information.

### Token for initial hallucinated tokens
Run `src/hallucination_labelling/token_stats.py` to get information on the distribution of tokens with label `initial hallucinated`

### Selecting optimal hyperparameters
We assume that our trained models in `checkpoints` have the following structure:
```--log-interval 10000000000 \
checkpoints
├── language_model
│   ├── standard
│   │   └── checkpoint_best.pt
│   └── bart
│       └── checkpoint_best.pt
└── summarization_model
    ├── standard
    │   └── checkpoint_best.pt
    └── bart
        └── checkpoint_best.pt
```
The two hyperparemeters we want to select are $\lambda$ and `ent_threshold`, the influence of the language model and the entropy threshold for the summarization model at which we trigger MMI decoding respectively. The goal is to pick the optimal parameter combination such that we minimizs the average log probability of Initial Hallucinated tokens (i.e. P = P_SM + $\lambda$ P_LM) and maximizes the ROUGE score of the 500 generated sentences.

The source directory for the following scripts is `src/hallucination_labelling/`. The $\lambda$ candidates are chosen by `generate_lambdas.py`, and the `ent_threshold` candidates by `generate_thresholds.py`. For each parameter combination we do the following:
* Run `fairseq-generate` as seen in the [token level entropy](#computing-token-level-entropy) section to score the references and get the average log probability of Initial Hallucinated tokens
* Run `fairseq-generate` but without the `--score-reference` parameter to generate hypothesis sentences, and compute the ROUGE score
* This gives 2 values, a log probability and a ROUGE score. Add this point to a plot. Our goal is to pick parameters that maximizes ROUGE and minimizes log probability.

The script `param_search_command.py` with and without the `--score_ref` parameter generates the commands to run the above `fairseq-generate` experiments on the server. Then due to limitations of the server, we have to run separate postprocessing on the raw results on both server and client side:
* Run `process_param_search.py` on the server
* Run `process_param_search.py --local` locally.
* Run `plot_param_search.py` to get the resulting plots.
(Additional parameters can be provided to `param_search_command.py`, the same parameters should also be passed to `process_param_search.py` and `plot_param_search.py`)

### Decoding with optimal hyperparameters
The test set has the 500 labelled hallucination test examples removed. For a given lambda/threshold parameter pair:
* Run `src/generate_no_500_command.py lambda threshold`
* Run `src/generate_no_500_command.py lambda threshold --score_ref`
(with appropriate additional parameters) to both generate summaries and score the reference summaries.

## Automatic factuality detection
### Setup
We use the automatic factuality detection as described in the [paper](https://arxiv.org/abs/2011.02593) which is implemented in the associated [repository](https://github.com/violet-zct/fairseq-detect-hallucination). To install the package, I first fork the repository then run the following:
```
git clone https://github.com/VanderpoelLiam/fairseq-detect-hallucination.git
python -m venv detect_hall
source detect_hall/bin/activate
python -m pip install --upgrade pip
cd fairseq-detect-hallucination/
pip install wheel
pip install -U git+https://github.com/ddkang/loss_dropper.git
pip install --editable ./
```

Then to check everything is correctly installed, we run a hallucination evaluation script using the trained XSum model:
```
mkdir models
cd models/
<!-- This takes a while -->
wget https://dl.fbaipublicfiles.com/detect-hallucination/xsum.roberta.tar.gz
```

Then we need to modify lines 12-13 of `util_scripts/eval_predict_hallucination_xsum.py` to:
```
models = ["models/xsum.roberta.tar.gz"]
datapath = "models/xsum.roberta.tar.gz/data"
```

and can then run the evaluation with:
```
python util_scripts/eval_predict_hallucination_xsum.py
```

This should produce the following output:
```--log-interval 10000000000 \
models/xsum.roberta.tar.gz
Loaded the model!
use ref = 0
TranS2S
Processed 100 lines!
Processed 200 lines!
Processed 300 lines!
Processed 400 lines!
Processed 500 lines!
Percentage of hallucination tokens = 0.5571630588491243, gold = 0.4674208637006418
Sentence-level F1: 0.9594172736732571
Sentence-level hallucination percentage (gold) = 0.924
Spearman-corr by token: 0.32839376258434905
Spearman-corr by probs: 0.327952862600306
0.5642327215931277 0.6725622527344659 0.6136532540609407 0.922 0.32839376258434905 0.327952862600306 0.6041553355814206
use ref = 1
TranS2S
Processed 100 lines!
Processed 200 lines!
Processed 300 lines!
Processed 400 lines!
Processed 500 lines!
Percentage of hallucination tokens = 0.5867507886435331, gold = 0.4674208637006418
Sentence-level F1: 0.9604989604989606
Sentence-level hallucination percentage (gold) = 0.924
Spearman-corr by token: 0.3156002998185347
Spearman-corr by probs: 0.32043934981785943
0.5587690025954765 0.7014195950663253 0.6220204313280364 0.924 0.3156002998185347 0.32043934981785943 0.6028499945610791
```

### Hallucination prediction
We want to obtain hallucination labels for the reference summaries for a particular preprocessing of the `xsum-no-500` dataset (e.g. standard or bart). The reference labelling is the same for any `lambda/ent_threshold` pair, we just need to specify one of generated `standard_ref_no_500_*` log files.

Take the example of the `bart` preprocessed dataset, where we generated `bart_ref_no_500_6.5602E-02_3.5987E+00` (i.e. `lambda = 6.5602E-02` and `ent_threshold = 3.5987E+00`). To get the desired labelling `label_processed_bart_ref`:
* Run `src/preprocessing/preprocess --detect_hall`. To get the source file `data/xsum-detect-hall/source`. It is the same for both `standard` and `bart` preprocessed datasets.
* Run `src/detect_hallucination/detect_hypothesis_hallucinations_command.py 6.5602E-02 3.5987E+00 --score_ref --dataset bart` to generate the batch command. Then paste this on the server. It is not always possible to process the labels successfully. However the failure percentage is quite low `TODO/%` for this particular example, and failed results are indicated by `FAIL` at that line in the `label_processed_bart_ref` file.

#### Internals of hallucination prediction
The data directory is `data/xsum-detect-hall/`. The labeling requires 3 files:
* `source`: The raw source text.
* `hypo_bart_6.5602E-02_3.5987E+00`: The hypothesis sentences with tokenization and bpe removed.
* `hypo_processed_bart_6.5602E-02_3.5987E+00`: The hypothesis sentences as produced by the model.

and the resulting file is:
* `label_processed_bart_ref`: The predicted hallucination labeling of `hypo_processed_bart_6.5602E-02_3.5987E+00` (but this would be the same for any choice of `lambda/ent_threshold` as the reference targets are the same for any parameter pair)


## Evaluation of our decoding method
Assume we are working with the `standard` preprocessed dataset with `lambda = 1.3120E-01` and `ent_threshold = 3.5618E+00`. In previous steps we did the following:
* Determined optimal hyperparameters (e.g. `standard`, `lambda = 1.3120E-01` and `ent_threshold = 3.5618E+00`).
* Run `src/generate_no_500_command.py 1.3120E-01 3.5618E+00` and `src/generate_no_500_command.py 1.3120E-01 3.5618E+00 --score_ref` (with appropriate additional parameters) in order to generate `logs/experiments/standard_no_500_1.3120E-01_3.5618E+00` and `logs/experiments/standard_no_500_ref_1.3120E-01_3.5618E+00`. These log files contain information about token level entropy, probability and ranking under both the summarization and language models for generated text and reference summaries respectively
* Run hallucination prediction scripts as described in the previous section to generate `label_processed_standard_ref` that is the token level hallucination labels for the reference summaries.

Remaining is to evaluate the performance of our decoding method compared to the default (`lambda = ent_threshold = 0`). The metrics we extract are:
1. ROUGE scores, [BERTScores](https://github.com/Tiiiger/bert_score#readme)
2. Average log probability by token label
3. Average ranking by token label

We are looking to see that our decoding method does not substantially decrease ROUGE scores for the generated text, and both average log probability and ranking are lower for initial hallucinated tokens from the reference summaries.

Run `python src/evaluate_decoding.py 1.3120E-01 3.5618E+00` (with appropriate additional parameters) to generate the various metrics.

# Repeatiing the analysis for CNN-Dailymail dataset
## Preparing CNN-Dailymail dataset
The general instructions for preparing the datasets are given [here](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md)

Useful links:
- https://github.com/facebookresearch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/bart/README.summarization.md
- https://github.com/abisee/cnn-dailymail

### Download raw CNN and DailyMail datasets
Download the `stories` files from [here](https://cs.nyu.edu/~kcho/DMQA/) to the `data/` directory under `data/cnn` and `data/dailymail`.

### Extract the datasets
We download [make_datafiles.py](https://github.com/artmatsak/cnn-dailymail/blob/master/make_datafiles.py) and the [url_lists](https://github.com/artmatsak/cnn-dailymail/tree/master/url_lists)to preprocess this raw data for summarization tasks.

`make_datafiles.py` was then modified to work with our existing preprocessing code. Then run:
```
python src/preprocessing/make_datafiles.py data/cnn/stories data/dailymail/stories
```

We create a smaller version of the same dataset consisting of only 10 samples using:
```
for SPLIT in train test valid; do   
	for LANG in source target; do
	head -10 data/cnn-dm-summarizer/$SPLIT.$LANG.txt >> data/cnn-dm-summarizer-samples/$SPLIT.$LANG.txt;
	done;
done
```

This gives us the full dataset in `data/cnn-dm-summarizer/` and the 10 samples in `data/cnn-dm-summarizer-samples/`.

### Preprocess the datasets
As before, run `python src/preprocess.py --cnn` with the desired additional arguments to preprocess the data. See `python src/preprocess.py -h` for more information on the possible datasets.

### Using a pretrained model
We download the associated BART model finetuned on CNN-DM to `checkpoints/summarization_model/bart.large.cnn` from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)
