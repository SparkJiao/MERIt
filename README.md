# ReClor_challenge


## Experimental Results

### ReClor

|   base-model |       Pre-train       | bs/gpu * gpu_num | lr   | epoch | Val Acc. | Test Acc. | path |
| :----------: | --------------------- | ---------------- | ---- | ----- | -------- | --------- | ---- |
| RoBERTa-base |  ---                  | 24 * 1           | 2e-5 |   10  |  53.0    |  48.5*    | roberta.base.2.0 |
|              |  ERICA                | 24 * 1           | 1e-5 |   10  |  54.8    |  49.3     | roberta.base.erica.ep.rp.1.0 |
|              |  wiki-path-v4.0 + MLM | 24 * 1           | 2e-5 |   10  |  56.2    |  51.0     | roberta.base.wiki_erica_path_v4_0.2.0_cp500.1.0.2080Ti |
| RoBERTa-large|  ---                  | 24 * 1           | 1e-5 |   10  |  64.0    |  55.6     | roberta.large.2.0 |
|              |  wiki-path-v8.1 + MLM | 6 * 4            | 1e-5 |   10  |**67.8**  |**61.5**   | roberta.large.wiki_erica_path_v8.1.1.2080ti-cp500.2.0.w4.2080Ti |
|              |                       | 6 * 4            | 1e-5 |   10  |  67.4    |  61.2     | roberta.large.wiki_erica_path_v8.1.1.2080ti-cp500.4.0.w4.2080Ti (Re-use the head of pre-training) |
| DAGN         |  RoBERTa-large        | 16 * 1           | 5e-6 |   10  |  64.8    |           |   |
|              |  wiki-path-v8.1 + MLM | 16 * 1           | 5e-6 |   10  |          |           |   |

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
