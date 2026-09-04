# Qwen/Qwen2.5-0.5B: WikiText-2 test-prefix perplexity

16383 scored tokens; context=512, stride=256.

| Format | Perplexity | Top-1 accuracy (95% CI) | Quality retention vs FP16 (95% paired bootstrap CI) |
|---|---:|---:|---:|
| fp16 | 12.764 (+0.00%) | 48.70% (47.94%–49.47%) | 100.00% (100.00%–100.00%) |
| int8 | 12.776 (+0.09%) | 48.61% (47.85%–49.38%) | 99.81% (99.57%–100.04%) |
| int4 | 15.061 (+18.00%) | 46.13% (45.37%–46.90%) | 94.72% (93.83%–95.63%) |

Quality retention здесь означает отношение next-token top-1 accuracy к FP16.
Это узкая языковая метрика на WikiText-2, а не процент всех способностей модели.
PPL change is not a percentage of quality retained.
This fixed prefix is a pilot, not a full-corpus published benchmark score.
Use validation data for tuning; do not tune configurations on this test result.
Possible model pretraining overlap with WikiText is not ruled out.
Dataset: https://huggingface.co/datasets/Salesforce/wikitext
