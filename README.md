# ReClor_challenge


## Experimental Results

### ReClor

|   base-model |       Pre-train       | bs/gpu * gpu_num | lr   | epoch | Val Acc. | Test Acc. | path |
| :----------: | --------------------- | ---------------- | ---- | ----- | -------- | --------- | ---- |
| RoBERTa-base |  ---                  | 24 * 1           | 2e-5 |   10  |  53.0    |  48.5*    | roberta.base.2.0 |
|              |  ERICA                | 24 * 1           | 1e-5 |   10  |  54.8    |  49.3     | roberta.base.erica.ep.rp.1.0 |
|              |  wiki-path-v4.0 + MLM | 24 * 1           | 2e-5 |   10  |  56.2    |  51.0     | roberta.base.wiki_erica_path_v4_0.2.0_cp500.1.0.2080Ti |
| RoBERTa-large|  ---                  | 24 * 1           | 1e-5 |   10  |  64.0    |  55.6     | roberta.large.2.0 |
|              |  wiki-path-v8.1 + MLM | 6 * 4            | 1e-5 |   10  |**67.8**  |  61.5     | roberta.large.wiki_erica_path_v8.1.1.2080ti-cp500.2.0.w4.2080Ti |
|              |                       | 6 * 4            | 1e-5 |   10  |  67.4    |  61.2     | roberta.large.wiki_erica_path_v8.1.1.2080ti-cp500.4.0.w4.2080Ti (Re-use the head of pre-training) |
|              |                       | 6 * 4            | 1e-5 |   10  |  67.6    |**61.7**   | roberta.large.wiki_erica_path_v8.1.1.2080ti-cp500.5.0.w4.2080Ti |
| DAGN         |  RoBERTa-large        | 16 * 1           | 5e-6 |   10  |  64.8    |           |   |
|              |  wiki-path-v8.1 + MLM | 16 * 1           | 5e-6 |   10  |          |           |   |

Average: 67.4 60.5
path-pt-v8-1-cp500-w4-2.0-p-tuning-tk10-s42: 67.40 58.60 76.59 44.46  
path-pt-v8-1-cp500-w4-2.0-p-tuning-tk10-s43: 67.00 60.80 78.18 47.14  
path-pt-v8-1-cp500-w4-2.0-p-tuning-tk10-s44: 67.80 60.50 76.59 47.86  
path-pt-v8-1-cp500-w4-2.0-p-tuning-tk10-s45: 68.60 61.90 79.09 48.39  
path-pt-v8-1-cp500-w4-2.0-p-tuning-tk10-s4321: 66.20 60.80 77.27 47.86  

Average: 67.08 59.3 76.00 46.18
path-pt-v8-1-no_aug-cp500-w4-2.0-s42: 66.40 58.90 76.36 45.18
path-pt-v8-1-no_aug-cp500-w4-2.0-s43: 66.80 60.00 75.45 47.86
path-pt-v8-1-no_aug-cp500-w4-2.0-s44: 65.40 58.90 77.05 44.64
path-pt-v8-1-no_aug-cp500-w4-2.0-s45: 68.20 56.70 74.55 42.68
path-pt-v8-1-no_aug-cp500-w4-2.0-s4321: 68.60 62.00 76.59 50.54

Average: 68.32 60.34 78.04 46.43
path-pt-v8-1-no_aug-cp500-w4-5.0-s42: 67.80 61.30 78.86 47.5
path-pt-v8-1-no_aug-cp500-w4-5.0-s43: 69.20 59.40 77.27 45.36
path-pt-v8-1-no_aug-cp500-w4-5.0-s44: 67.20 61.30 77.95 48.21
path-pt-v8-1-no_aug-cp500-w4-5.0-s45: 69.00 60.60 78.41 46.61
path-pt-v8-1-no_aug-cp500-w4-5.0-s4321: 67.80 59.10 77.73 44.46

path-pt-v8-1-random-cp500-w4-2.0-s42: 63.6
path-pt-v8-1-random-cp500-w4-2.0-s43: 65.2
path-pt-v8-1-random-cp500-w4-2.0-s44: 65.0 
path-pt-v8-1-random-cp500-w4-2.0-s45: 65.6
path-pt-v8-1-random-cp500-w4-2.0-s4321: 63.2

Average: 66.92 60.5 
path-pt-v8-1-random-cp500-w4-5.0-s42: 66.6 59.10 75.45 46.25
path-pt-v8-1-random-cp500-w4-5.0-s43: 65.2 59.70 77.73 45.54
path-pt-v8-1-random-cp500-w4-5.0-s44: 68.0 60.50 76.82 47.68
path-pt-v8-1-random-cp500-w4-5.0-s45: 67.0 61.60 77.50 49.11
path-pt-v8-1-random-cp500-w4-5.0-s4321: 67.8 61.6 78.64 48.21

Average: 60.1
path-pt-v8-1-random-ht-cp500-w4-5.0-s42: 63.2 76.36 52.86
path-pt-v8-1-random-ht-cp500-w4-5.0-s43: 58.3 76.14 44.28
path-pt-v8-1-random-ht-cp500-w4-5.0-s44: 61.8 77.05 49.82
path-pt-v8-1-random-ht-cp500-w4-5.0-s45: 59.4 78.18 44.648
path-pt-v8-1-random-ht-cp500-w4-5.0-s4322: 57.60 79.09 40.71

Average: 57.9
path-pt-v8-1-random-ht-no-aug-cp500-w4-5.0-s42: 55.7 74.09 41.25
path-pt-v8-1-random-ht-no-aug-cp500-w4-5.0-s43: 59.1 76.14 45.71
path-pt-v8-1-random-ht-no-aug-cp500-w4-5.0-s44: 58.4 76.82 43.93
path-pt-v8-1-random-ht-no-aug-cp500-w4-5.0-s45: 58.6 75.91 45.00
path-pt-v8-1-random-ht-no-aug-cp500-w4-5.0-s4322: 57.50 74.77 43.93

Average: 66.28 58.92
path-pt-v8-1-no_shuffling-cp500-w4-2.0-s42: 65.60 58.20 75.68 44.46
path-pt-v8-1-no_shuffling-cp500-w4-2.0-s43: 65.40 59.50 76.14 46.43
path-pt-v8-1-no_shuffling-cp500-w4-2.0-s44: 68.00 58.40 76.59 44.11
path-pt-v8-1-no_shuffling-cp500-w4-2.0-s45: 67.60 59.60 79.32 44.11
path-pt-v8-1-no_shuffling-cp500-w4-2.0-s4321: 64.80 59.10 78.41 43.93

Average: 67.16 58.4
path-pt-v8-1-no_shuffling-cp500-w4-5.0-s42: 66.20 59.70 76.14 46.79
path-pt-v8-1-no_shuffling-cp500-w4-5.0-s43: 68.00 56.90 74.32 43.21
path-pt-v8-1-no_shuffling-cp500-w4-5.0-s44: 68.20 57.50 74.77 43.93
path-pt-v8-1-no_shuffling-cp500-w4-5.0-s45: 68.00 59.10 78.64 43.75
path-pt-v8-1-no_shuffling-cp500-w4-5.0-s4321: 65.40 58.8 77.73 43.93

------

Average: 66.8 59.6
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.0-s42: 66.80 60.30 77.50 46.79 s1800  
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.0-s43: 65.60 59.70 78.41 45.00 s1500  
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.0-s44: 66.00 58.00 77.05 43.04 s1000   
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.0-s45: 67.60 59.20 77.73 44.64 s1900  
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.0-s4321: 67.80 60.70 79.55 45.89 s1300

Average 66.5
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.1-s42: 65.60  s1700  
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.1-s43: 65.00  s1100  
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.1-s44: 67.60  s1600   
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.1-s45: 67.80  s700  
path-pt-v8-2-2-1aug-ctx-cp500-w4-2.1-s4321: 66.60  s1500

Average: 67.9 60.3
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.0-s42: 66.20 60.70 78.41 46.79 s1300
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.0-s43: 66.60 58.20 74.55 45.36 s800
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.0-s44: 67.20 60.80 77.50 47.68 s1200
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.0-s45: 67.80 60.80 79.09 46.43 s1000
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.0-s4321: 71.60 60.80 78.41 46.96 s1500

Average: 68.2 61.2
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.1-s42: 66.80 61.00 79.77 46.25 s1000
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.1-s43: 69.80 60.10 78.64 45.54 s1000
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.1-s44: 68.80 62.60 79.77 49.11 s900
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.1-s45: 68.40 60.40 77.95 46.61 s1200
path-pt-v8-2-2-1aug-ctx-cp500-w4-5.1-s4321: 67.4 62.10 79.32 48.57 s1500


### LogiQA

|   base-model |       Pre-train       | bs/gpu * gpu_num | lr   | epoch | Val Acc. | Test Acc. | path |
| :----------: | --------------------- | ---------------- | ---- | ----- | -------- | --------- | ---- |
| RoBERTa-large|  ---                  | 24 * 1           | 5e-6 |   10  |  34.25   |  34.25    | logiqa.roberta.large.2080ti.w3.v2.1 |
|              | wiki-path-v8.1 + MLM  | 24 * 1           | 5e-6 |   10  |  41.01   |  36.25    | logiqa.roberta.large.wiki_erica_path_v8.1.1.2080ti-cp500.w3.v2.1 |

## Progress

### 2021/3/12

1. 对于部分问题，为了判断答案是否正确需要一些额外的常识的辅助，是类似A事件反映了B特征，B特征能够推出C这种类似的情况，其中A来自题干，C是需要判断的选项，而B是一个未知的决策因子或者桥。
2. 对于另一部分问题，需要正确处理的是逻辑顺序，即如果能够按照一定顺序重新组合题目和选项中的叙事，整体的逻辑“看上去”就是正确的。
    1. 从这个角度出发，需要解决的问题是如何建模叙述的线索。sentence的粒度太粗，token的粒度太细，希望能使用短语的粒度。
    2. 关于顺序如何建模？GNN是肯定要用的了，问题在于怎么把顺序考虑进去，以提供解释性或者方便实验验证。
