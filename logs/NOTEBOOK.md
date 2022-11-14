# Summarization modelling
## Train
### `train_1`
SIGBUS error. 2 GPUs, `4096` memory. Interpretation is that it ran out of memory. `--max-tokens 4096 --update-freq 2`.

### `train_2`
Trained successfully until time limit. 2 GPUs, `4096` memory. `--max-tokens 2048 --update-freq 2`
```
epoch 034
valid | loss 8.128
train | loss 7.314
```

### `train_3`
Trained successfully until time limit. 2 GPUs, `4096` memory. `--max-tokens 4096 --update-freq 1`
```
epoch 032
valid | loss 8.388
train | loss 7.491
```

### `train_4`
Trained successfully until time limit. 1 GPU, `4096` memory. `--max-tokens 4096 --update-freq 8`. Much better performance because of higher `update-freq`. Doubling GPUs is equivalent to doubling the `update-freq`.

Next time:
  Max memory used was `4707` total while we had `16384` total allocated, should increase `update-freq` to use more of the memory or decrease memory requested. Also try continuing training from previous checkpoint.

Best validation score:
```
epoch 031
valid | loss 7.059
train | loss 5.968
```

Last epoch score:
```
epoch 034
valid | loss 7.078
train | loss 5.931
```

### `train_5`
I increased `update-freq` from `8` to `32` and decreased `dropout` from `0.3` to `0.2`. Started training from best model from `train_4`.

Trained successfully until time limit, 1 GPU. Still allocating too much memory. Decrease allocated memory to `2048` per CPU core. Did not improve on `valid` loss from `model_4`, but `train` loss went much lower, suggests overfitting. Trained for `30` epochs compared to `34` in  `train_4`, so increasing `update-freq` further should be okay. Could also use 2 GPUs.

Best validation score: Same as `train_4`.

Last epoch score:
```
epoch 054
valid | loss 7.426
train | loss 4.789
```

### `train_6`
Increased `update-freq` from `8` to `16` and changed the architecture and associated parameters to the larger `transformer_wmt_en_de` model. Best validation score occurs by epoch `5` so we are overtraining by a lot.

Best validation score:
```
epoch 005
valid | loss 6.871
train | loss 6.213
```

Last epoch score:
```
epoch 027
valid | loss 7.725
train | loss 4.339
```

### `train_7`
Increased `update-freq` from `16` to `64`. Used same`transformer_wmt_en_de` model as `train_6`. Increased dropout from `0.1` to `0.3` to fight overfitting. Still training too fast as reach best validation score in epoch `8`. Next time follow Clara suggestion to decrease learning rate and include patience so don't waste epochs `8` to `20` overfitting. Happy with memory usage and `update-freq`.

Best validation score:
```
epoch 008
valid | loss 6.718
train | loss 6.611
```

Last epoch score:
```
epoch 020
valid | loss 7.182
train | loss 5.104
```

### `train_8`
Decreased learning rate by factor of `10` from `7e-4` to `7e-5`. Added `--patience 5` to stop training if validation score does not improve after `5` epochs. Used dropout of `0.1`. As kept improving till last epoch should continue training for more time.

Best validation score = Last epoch score:
```
epoch 021
valid | loss 6.515
train | loss 6.228
```

### `train_9`
Continued training `train_8` for `20` more hours with `patience 5`.

From now on, I only report the best validation score as the `patience` parameter ensures I train for maximum `5` additional epochs after the validation score stops improving.

In total this model trained for `40` hours. We trained for `2` more epochs without improving validation score. I do not think this is worth training further. Rather I think we need to run `train_10` for an additional `20` hours.

Best validation score:
```
epoch 039
valid | loss 6.297
train | loss 5.407
```

### `train_10`
Same parameters as `train_8` but with dropout increased from `0.1` to `0.3` and time increased from `20` hours to `24` hours.

As expected from `train_9` this did not finish training so I ran it for more time.

Best validation score:
```
epoch 023
valid | loss 6.693
train | loss 6.663
```

### `train_11`
Continued training `train_10` for `24` more hours with `patience 5`.

This still did not finish training but I am happy with the performance and based on the improvement from `train_8` to `train_9` I suspect that any improvement will be marginal.

Best validation score:
```
epoch 050
valid | loss 6.225
train | loss 5.832
```

## Generate
### `generate_1`
Failed due to bug in my modification of fairseq.

### `generate_2`
```
R1: 18.68
R2: 2.514
RL: 14.70
```

### `generate_3`
```
R1: 17.81
R2: 2.533
RL: 14.24
```
### `generate_4`
Like in training we only use `4569` memory of the `16384` total allocated, so we should massively increase the batch size in the future.
```
R1: 19.06
R2: 3.166
RL: 15.08
```

### `generate_5`
Increased `batch-size` from `8` to `32`.

Could not run as training did not improve on `model_4`.

### `generate_6`
Used larger `batch-size` of `32` and the larger `transformer_wmt_en_de` model. Still only used `4667` of memory, so decreased CPU memory from `4096` to `2048` for next time. Also increased default `batch-size` to `64`.

Interesting how ROUGE scores are worse than `generate_4` even though `valid` scores are better.
```
R1: 18.8
R2: 3.112
RL: 14.92
```

### `generate_7`
Best ROUGE scores so far by a lot.
```
R1: 26.17
R2: 7.785
RL: 21.17
```

### `generate_8`
Another big improvement in ROUGE scores and training did not finish.
```
R1: 30.34
R2: 10.02
RL: 24.32
```

### `generate_9`
Slight improvement ROUGE scores with an additional `20` hours of training.
```
R1: 32.55
R2: 11.48
RL: 26.04
```

### `generate_10`
Performed worse than `generate_8` suggests increased dropout too much. But also did not finish training, need to wait on `generate_11`
```
R1: 29.55
R2: 9.55
RL: 23.89
```

### `generate_11`
This is the most performant model so far. It does better than `generate_9` on R2 and RL metrics but worse on R1.
```
R1: 32.35
R2: 11.53
RL: 26.16
```

# Language Modelling
## Base runs
### `train_lang_1`
Trained successfully until time limit.

### `train_lang_full_1`
Trained successfully until hit `--max-update 50000`

### `eval_lang_1`
For evaluations, a lower perplexity and loss indicates a better model.
Loss (base 2): 5.4836, Perplexity: 44.74

### `eval_lang_full_1`
Loss (base 2): 5.1522, Perplexity: 35.56


## Update freq exploration
I looked at how increasing the update frequency from `2` in the previous runs to larger values impacts the model.

The unchanged parameters were: `dataset = standard, lr = 5.0000E-04`
The parameters we changed were: `full, update_freq`

### `LM` Results
There was a `Bus error` for `update_freq = 32`. So the model did not train, I did not bother rerunning the results as the `update_freq = 64` model did run.

`update_freq`  | 16 | 32 | 64 |
--- | --- | --- | --- |
epoch                  | 024     | NA | 068      |
train loss             | 4.761   | NA | 4.798    |
valid loss             | 5.556   | NA | 5.66     |
Successfully completed | Yes     | NA | Yes      |
CPU time (sec)         | 6077.44 | NA | 14147.59 |
Average Memory (MB)    | 3461.19 | NA | 2762.71  |
eval loss              | 5.5857  | NA | 5.6758   |
eval perplexity        | 48.02   | NA | 51.12    |

Based on these results I should decrease the learning rate for the `update_freq = 64` model.

### `LM_FULL` Results
`update_freq`  | 16 | 32 | 64 |
--- | --- | --- | --- |
epoch                  | 020      | 017      | 012 |
train loss             | 4.741    | 4.775    | 5.032 |
valid loss             | 4.798    | 4.841    | 5.022 |
Successfully completed | No       | No       | No |
CPU time (sec)         | 74108.00 | 65377.00 | 67736.00 |
Average Memory (MB)    | 2882.01  | 2809.54  | 2162.84 |
eval loss              | 4.7386   | 4.7831   | 4.9695 |
eval perplexity        | 26.70    | 27.53    | 31.33 |

Based on these results I should increase the training time for all the `LM_FULL` models so they can finish training. I cannot yet tell if I should decrease the learning rate. After additional training time:

`update_freq`  | 16 | 32 | 64 |
--- | --- | --- | --- |
epoch                  | 036      | 051      | 033      |
train loss             | 4.641    | 4.548    | 4.649    |
valid loss             | 4.742    | 4.721    | 4.789    |
Successfully completed | No       | No       | No       |
CPU time (sec)         | 72516.00 | 73798.00 | 68982.00 |
Average Memory (MB)    | 2108.48  | 2846.70  | 2859.11  |
eval loss              | 4.6793   | 4.6572   | 4.7317   |
eval perplexity        | 25.62    | 25.23    | 26.57    |

Something went wrong when writing logs for this additional training time, so the CPU time is not correct. I found that I had trained the models for additional epochs, but had not written the results. Nevertheless, the model only saves the best model, so the train/valid/eval loss values are still correct.

I had to run all the models `3` times. I am satisfied with the `update_freq = 32` model as it somehow ran the most. I think the total runtime was `3600` mins.

## Learning rate exploration
I looked at how decreasing the learning rate from `5.0E-04` impacts the model.

The unchanged parameters were: `dataset = standard, update_freq = 64, data_source = only targets`
The parameters we changed were: `lr`

`lr = 2.5E-04` failed with weird error: `FileNotFoundError: [Errno 2] No such file or directory: 'checkpoints/language_model/standard/dummy'`

`lr`  | `1.0E-04` | `2.5E-04` | `5.0E-04` |
--- | --- | --- | --- |
epoch                  | 161      | NA | 068      |
train loss             | 4.653    | NA | 4.798    |
valid loss             | 5.631    | NA | 5.66     |
Successfully completed | Yes      | NA | Yes      |
CPU time (sec)         | 32426.29 | NA | 14147.59 |
Average Memory (MB)    | 3037.04  | NA | 2762.71  |
eval loss              | 5.6404   | NA | 5.6758   |
eval perplexity        | 49.88    | NA | 51.12    |

Decreasng the learning rate does seem to improve performance, but it takes around `2` times as long to train. The `LM_FULL` model does so much better, it is not worth training the `LM` model further.

## BART language model
### `train_lang_bart_32_5.0000E-04_full`
The model didn't finish training, but I was sick of the training getting `Bus error`.

epoch                  | 045    
train loss             | 4.479
valid loss             | 4.604
Successfully completed | No  
CPU time (sec)         | 175509.98
eval loss              | 4.5453
eval perplexity        | 23.35  


# MMI Decoding
## `train_3` model
For LM, best performance is at `2^{-03}`, but pretty good in the `2^{-03}-2^{-05}` range. Big improvement from `2^{-02}` to `2^{-04}`. Results from `2^{-05}-2^{-10}` are pretty much identical. Interpretation is that language model gives an improvement but need to try more values in the `2^{-02}-2^{-05}` range.

For LM_FULL, same conclusions as for language model. Interesting that full model does not perform better even though it has better loss/perplexity.

### Results
Lamda | LM | LM_FULL
--- | --- | ---
2^{-01} | 3.007 | 4.75
2^{-02} | 14.06 | 14.13
2^{-03} | **14.5** | 14.44
2^{-04} | 14.48 | **14.45**
2^{-05} | 14.42 | 14.41
2^{-06} | 14.39 | 14.38
2^{-07} | 14.38 | 14.38
2^{-08} | 14.39 | 14.38
2^{-09} | 14.38 | 14.38
2^{-10} | 14.38 | 14.37
0       | 14.24 | 14.24


## `train_9` model
Got the following error:
`RuntimeError: CUDA out of memory. Tried to allocate 1.06 GiB (GPU 0; 10.76 GiB total capacity; 4.13 GiB already allocated; 1.02 GiB free; 8.59 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
`
This is likely because the batch size is too large. So I decreased the `batch_size` from `64` to `32` and increased the training time from `360` to `1000` mins.

Another issue is that I will not log to `lambda_search.log` until after the whole batch job is completed. So I need to have one to do the generating and another to process the generated output.

Weird thing is that it seems that the `fairseq-generate` did work. I am getting reasonably numbers when running the ROUGE scoring locally. It is not clear at what point the `out of memory` error occured. Only way to be certain is to rerun the script.

### Results
Bug in lambda generation meant I only had `9` unique lambda values

Lamda | LM | LM_FULL
--- | --- | ---
5.0000E-01 | 5.913 | 11.63
2.5000E-01 | 23.05 | 23.37
2.0318E-01 | 23.89 | 24.25
1.2500E-01 | 24.86 | 24.96
1.5850E-01 | 24.46 | 24.66
1.3116E-01 | 24.76 | 24.89
1.2137E-01 | 24.89 | 24.98
1.1525E-01 | 24.87 | 24.97
8.7584E-02 | 24.96 | 25.01
6.2500E-02 | 25.12 | 25.12
4.3715E-02 | 25.19 | 25.20
4.0373E-02 | 25.20 | 25.19
3.5432E-02 | 25.22 | 25.19
3.1250E-02 | 25.22 | 25.19
1.5625E-02 | 25.2 | **25.23**
7.8125E-03 | **25.23** | **25.23**
3.9062E-03 | 25.22 | 25.19
1.9531E-03 | 25.21 | 25.21
9.7656E-04 | 25.21 | 25.2
0       | **26.04** | **26.04**


## `train_11` model
Still getting `RuntimeError: CUDA out of memory.` error. I suspect batch size is still to large, so I decrease it from `32` to `16` in `MMI_decode.sh`. I left training time at `1000` mins as we only run for around `135` mins total with the `32` batch size.

New issue is that csplit overwrites files when I run two different gridsearch simultaneosly. Fix is to change the directory csplit writes to to the log directory e.g. `logs/hyperparameters/lang/11`.  

Also new is `Can't locate XML/Parser.pm in @INC` error. Emailed service desk as couldnt find a fix myself. But still could run all the ROUGE scoring locally.

On second attempt noticed that lambda is chosen to be `1.1525E-01` twice. Upon checking, the `1.1525E-01.log` file is correct, suggesting that the file was just overwritten. Fix is to do uniqueness check after rounding.

### Results

Lamda | LM | LM_FULL
--- | --- | ---
5.0000E-01 | 5.024 | 12.53
2.5000E-01 | 23.04 | 23.79
2.0318E-01 | 23.96 | 24.44
1.5850E-01 | 24.46 | 24.87
1.3116E-01 | 24.80 | 25.04
1.2500E-01 | 24.83 | 25.06
1.2137E-01 | 24.84 | 25.07
1.1525E-01 | 24.92 | 25.11
8.7584E-02 | 25.06 | 25.18
6.2500E-02 | 25.16 | 25.28
4.3715E-02 | 25.23 | 25.26
4.0373E-02 | 25.23 | 25.25
3.5432E-02 | 25.24 | 25.26
3.1250E-02 | 25.28 | 25.3
1.5625E-02 | **25.29** | **25.34**
7.8125E-03 | **25.29**| 25.29
3.9062E-03 | 25.28 | 25.29
1.9531E-03 | 25.26 | 25.27
9.7656E-04 | 25.27 | 25.27
0       | **26.16** | **26.16**

# Detecting Hallucinated Content
## Entropy by label
### Best Standard Summarization Model


Label | Average Entropy | Standard Error
--- | --- | ---
Hallucinated | 3.8111 | 0.0292
Non-Hallucinated | 3.6893 | 0.0209
Initial Hallucinate |: 4.1972 | 0.0648
Subsequent Hallucinated | 3.7405 | 0.0323

The T-test tests for the null hypothesis that two independent samples have identical average. For a p-value of 1% rejecting null-hypothesis means the averages are different. If we cannot reject, then the averages are the same. The data in the table is (whether we reject the null-hypothesis, t-statistic, p-value).

T-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '3.4241E+00', '6.1899E-04')
Initial Hallucinated    | ('Reject', '6.9612E+00', '3.6305E-12')
Subsequent Hallucinated | ('Cannot reject', '1.3707E+00', '1.7049E-01')

T-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '-5.5894E+00', '2.4133E-08')

The null hypothesis for the Kolmogorov–Smirnov test is that the two distributions are identical. The python implementation `ks_2samp` takes two PDFs as input. We choose the same p-value cutoff and table entry format as for the T-test.

KS-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '4.9225E-02', '2.0785E-06')
Initial Hallucinated      | ('Reject', '1.2742E-01', '1.6456E-09')
Subsequent Hallucinated | ('Reject', '4.2153E-02', '2.3055E-04')

KS-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '1.3291E-01', '1.5049E-09')

### Best Standard Language Model
Label | Average Entropy | Standard Deviation
--- | --- | ---
Hallucinated | 2.9969 | 0.0268
Non-Hallucinated | 3.3244 | 0.0203
Initial Hallucinated | 3.2458 | 0.0807
Subsequent Hallucinated | 2.9514 | 0.0279

T-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '-9.8227E+00', '1.0924E-22')
Initial Hallucinated    | ('Cannot reject', '-1.1183E+00', '2.6347E-01')
Subsequent Hallucinated | ('Reject', '-1.0741E+01', '8.7883E-27')


T-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '-3.9853E+00', '6.8469E-05')

KS-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '9.0718E-02', '9.7214E-21')
Initial Hallucinated    | ('Reject', '8.8501E-02', '8.2722E-05')
Subsequent Hallucinated | ('Reject', '9.3361E-02', '9.7876E-20')

KS-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '9.1125E-02', '1.0251E-04')

## Entropy by label
### Best BART Summarization Model
Label | Average Entropy | Standard Error
--- | --- | ---
Hallucinated | 2.5405 | 0.0200
Non-Hallucinated | 2.3898 | 0.0131
Initial Hallucinated | 3.1147 | 0.0514
Subsequent Hallucinated | 2.4490 | 0.0214


T-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '6.5731E+00', '5.1122E-11')
Initial Hallucinated    | ('Reject', '1.5365E+01', '1.3283E-52')
Subsequent Hallucinated | ('Cannot reject', '2.4882E+00', '1.2851E-02')

T-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '-1.1607E+01', '9.1677E-31')

KS-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '6.5855E-02', '2.5614E-12')
Initial Hallucinated    | ('Reject', '2.4589E-01', '5.7428E-35')
Subsequent Hallucinated | ('Reject', '4.8446E-02', '2.7251E-06')

KS-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '2.3520E-01', '5.3953E-30')

### Best BART Language Model
Label | Average Entropy | Standard Error
--- | --- | ---
Hallucinated | 2.7610 | 0.0254
Non-Hallucinated | 3.1549 | 0.0198
Initial Hallucinated | 3.0634 | 0.0801
Subsequent Hallucinated | 2.7128 | 0.0265

T-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '-1.2276E+01', '1.8805E-34')
Initial Hallucinated    | ('Cannot reject', '-1.2812E+00', '2.0016E-01')
Subsequent Hallucinated | ('Reject', '-1.3307E+01', '3.9254E-40')


T-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '-4.7632E+00', '1.9581E-06')

KS-test  | Non-Hallucinated
--- | --- |
Hallucinated            | ('Reject', '1.0749E-01', '4.0337E-32')
Initial Hallucinated    | ('Reject', '9.4579E-02', '1.6902E-05')
Subsequent Hallucinated | ('Reject', '1.1327E-01', '1.7174E-32')

KS-test  | Initial Hallucinated |
--- | --- |
Subsequent Hallucinated | ('Reject', '8.3294E-02', '4.0821E-04')

## Standard Log probabilities by label
The file `generate_hallucination.log` in `logs/experiments/` contains the log probabilities at the token level for both the summarization and language models. We wish to implement an entropy threshold such that when the entropy is above a certain value, we begin performing MMI decoding. Recall this means that our token probability is now: P_SM + $\lambda$ P_LM. The goal is to pick $\lambda$ to maximize the MMI objective given above.

As a first experiment, we can look at the average values for P_SM, P_LM by hallucination label (Hallucinated, Non-Hallucinated, Initial Hallucinated, Subsequent Hallucinated). This should indicate the sign of $\lambda$.

Label | Average P_SM | Average P_LM
--- | --- | ---
Hallucinated            | $-5.3715 \pm 0.0722$ | $-12.6874 \pm 0.0487$
Non-Hallucinated        | $-4.5491 \pm 0.0490$ | $-12.4756 \pm 0.0325$
Initial Hallucinated    | $-6.9432 \pm 0.1861$ | $-11.6208 \pm 0.1324$
Subsequent Hallucinated | $-5.0842 \pm 0.0774$ | $-12.8823 \pm 0.0516$

To maximize P_SM + $\lambda$ P_LM (the MMI objective) we should pick $\lambda$ to be negative.

## BART Log probabilities by label
Label | Average P_SM | Average P_LM
--- | --- | ---
Hallucinated |            $-2.8745 \pm 0.0510$ | $-13.5638 \pm 0.0553$
Non-Hallucinated |        $-2.2042 \pm 0.0328$ | $-13.1707 \pm 0.0394$
Initial Hallucinated |    $-4.5451 \pm 0.1574$ | $-12.8141 \pm 0.1598$
Subsequent Hallucinated | $-2.6083 \pm 0.0525$ | $-13.6833 \pm 0.0587$

# Entropy threshold decoding
As a first experiment I ran the entropy threshold implementation with $\lambda = -0.015625$ and the summarization entropy threshold of $4.2$. So when the token entropy is greater than this value, we instead run MMI decoding for this token.

The ROUGE scores for this approach on the validation set are:
```
R1: 31.61
R2: 11.13
RL: 25.3
```
These values cannot be directly compared to our previous results as they were computed on the full test set. But it is a good sanity check that the scores are not wildly different. Future experiments will use the `xsum-summarizer-no-500` dataset that has the `500` hallucinated test examples removed. This is to avoid information leakage as we use these `500` examples to pick the threshold value.

# Parameter search over lambda and entropy threshold
The results in `logs/hallucination_labelling/` contain our experiments in picking the optimal lambda/threshold combination that results in maximal ROUGE score and minimal log probability of the initial hallucinated tokens.

The initial parameter search was to pick `5` threshold values uniformly in `[mean - std, mean + std]`, with the mean/std taken over entropy values of tokens with label `initial hallucinated`. The lambda values were then picked to be the `10` values in `[2^{-1}, 2^{-2}, ..., 2^{-10}]`.

This initial search gives us ranges `[lamb_min, lamb_max]` and `[thres_min, thres_max]` that look promising. The second parameter search takes `10` lambda/threshold values uniformly from their respective ranges.

We then plot our results. We have three contour plots of lambda and threshold values vs
 1. ROUGE-L score
 2. Average token log probability for the intial hallucinated tokens
 3. A linear combination of the two: w * RL - AVG_P_SM

 The weight `w < 1` is picked so that we have around a 3:1 weighting for the ROUGE score as this is more important to us than the log probabilities being small.

## Standard
Based on the results in `logs/hallucination_labelling/standard` when plotted. We determine the optimal values to be:
```
lambda    = 1.3120E-01
threshold = 3.5618E+00
```
Another discovery was that is was pointless to have a `lambda=0` value in our search as when `lambda=0` entropy thresholding has no effect. So `lambda=0,threshold=...` is the same for any threshold.

## BART
The optimal values are:
```
lambda    = 6.5602E-02
threshold = 3.5987E+00
```

# Effect of new decoding algorithm
Our new decoding algorithm consists of applying Pointwise Mutual Information Decoding when the token entropy is above a certain threshold.

Our experiments in `logs/experiments/` consist of running the decoding with beam search and compare it to this new algorithm with respect to:
1. Various factuality metrics
2. Average log probability by token label
3. Average ranking by token label

Factuality metrics like ROUGE scores are computed on the generated summaries. Token label metrics are computed on reference summaries.

## Standard
We compare our experiments with various parameter to the case of `lambda = ent_threshold = 0`:
```
PROBABILITY STATISTICS
Average, Standard Error
Hallucinated:            -5.09 \pm 0.01
Non-Hallucinated:        -3.58 \pm 0.02
Initial Hallucinated:    -5.62 \pm 0.03
Subsequent Hallucinated: -5.01 \pm 0.01

RANKING STATISTICS
Average, Standard Error
Hallucinated:            10458 \pm 29
Non-Hallucinated:        8020 \pm 62
Initial Hallucinated:    13115 \pm 87
Subsequent Hallucinated: 10091 \pm 30

R1: 31.35
R2: 10.94
RL: 25.17

BERTS P: 0.901
BERTS R: 0.886
BERTS F1: 0.893

FactScore: 0.155
```

Parameters are `lambda = 1.3120E-01, threshold = 3.5618E+00`:
```

PROBABILITY STATISTICS
Average, Standard Error
Hallucinated:            -5.59 \pm 0.01
Non-Hallucinated:        -3.93 \pm 0.02
Initial Hallucinated:    -6.17 \pm 0.03
Subsequent Hallucinated: -5.51 \pm 0.01

-5.59 , -3.93 , -6.17 , -5.51 \pm

0.01, 0.02, 0.03, 0.01


RANKING STATISTICS
Average, Standard Error
Hallucinated:            11686 \pm 31
Non-Hallucinated:        9007 \pm 67
Initial Hallucinated:    14387 \pm 91
Subsequent Hallucinated: 11312 \pm 32

R1: 31.05
R2: 10.58
RL: 24.92

BERTS P: 0.897
BERTS R: 0.885
BERTS F1: 0.891

FactScore: 0.167
```

## BART
Parameters are `lambda = threshold = 0`:
```
PROBABILITY STATISTICS
Average, Standard Error
Hallucinated:            -2.48 \pm 0.01
Non-Hallucinated:        -2.15 \pm 0.01
Initial Hallucinated:    -3.48 \pm 0.02
Subsequent Hallucinated: -2.35 \pm 0.01

RANKING STATISTICS
Average, Standard Error
Hallucinated:            14219 \pm 31
Non-Hallucinated:        13403 \pm 80
Initial Hallucinated:    16029 \pm 94
Subsequent Hallucinated: 13994 \pm 33

R1: 45.33
R2: 22.31
RL: 37.2

BERTS P: 0.926
BERTS R: 0.917
BERTS F1: 0.922

FactScore: 0.126
```

Parameters are `lambda = 6.5602E-02, threshold = 3.5987E+00`:
```
PROBABILITY STATISTICS
Average, Standard Error
Hallucinated:            -2.55 \pm 0.01
Non-Hallucinated:        -2.22 \pm 0.01
Initial Hallucinated:    -3.61 \pm 0.02
Subsequent Hallucinated: -2.42 \pm 0.01

RANKING STATISTICS
Average, Standard Error
Hallucinated:            14614 \pm 32
Non-Hallucinated:        13678 \pm 81
Initial Hallucinated:    16581 \pm 96
Subsequent Hallucinated: 14370 \pm 34

R1: 45.27
R2: 22.24
RL: 37.19

BERTS P: 0.926
BERTS R: 0.917
BERTS F1: 0.922

FactScore: 0.128
```

Change in log probabilities:
Label | Standard | Bart
--- | --- | ---
Hallucinated|            -0.5 \pm 0.01  | -0.07 \pm 0.01
Non-Hallucinated|        -0.35 \pm 0.03 | -0.07 \pm 0.01
Initial Hallucinated|    -0.55 \pm 0.04 | -0.13 \pm 0.03
Subsequent Hallucinated| -0.5 \pm 0.01  | -0.07 \pm 0.01

Change in rankings:
Label | Standard | Bart
--- | --- | ---
Hallucinated|            1228 \pm 42  | 395 \pm 45
Non-Hallucinated|        987 \pm 91   | 275 \pm 114
Initial Hallucinated|    1272 \pm 126 | 552 \pm 134
Subsequent Hallucinated| 1221 \pm 44  | 376 \pm 47
