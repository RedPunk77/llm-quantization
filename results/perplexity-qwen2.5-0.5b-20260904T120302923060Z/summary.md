# Qwen/Qwen2.5-0.5B: WikiText-2 test-prefix perplexity

16383 scored tokens; context=512, stride=256.

| Format | Mean NLL | Perplexity (lower is better) | PPL change vs FP16 |
|---|---:|---:|---:|
| fp16 | 2.5466 | 12.764 | +0.00% |
| int8 | 2.5475 | 12.776 | +0.09% |
| int4 | 2.7121 | 15.061 | +18.00% |

PPL change is not a percentage of quality retained. No task accuracy measured.
This fixed prefix is a pilot, not a full-corpus published benchmark score.
Use validation data for tuning; do not tune configurations on this test result.
Possible model pretraining overlap with WikiText is not ruled out.
Dataset: https://huggingface.co/datasets/Salesforce/wikitext
