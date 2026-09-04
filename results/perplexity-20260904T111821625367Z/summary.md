# WikiText-2 test-prefix perplexity

16383 scored tokens; context=512, stride=256.

| Format | Mean NLL | Perplexity (lower is better) | PPL change vs FP16 |
|---|---:|---:|---:|
| fp16 | 2.7397 | 15.482 | +0.00% |
| int8 | 2.7414 | 15.509 | +0.17% |
| int4 | 3.1257 | 22.775 | +47.11% |

PPL change is not a percentage of quality retained. No task accuracy measured.
This fixed prefix is a pilot, not a full-corpus published benchmark score.
Use validation data for tuning; do not tune configurations on this test result.
Possible model pretraining overlap with WikiText is not ruled out.
Dataset: https://huggingface.co/datasets/Salesforce/wikitext
