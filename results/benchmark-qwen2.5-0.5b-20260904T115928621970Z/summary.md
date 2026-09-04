# Qwen/Qwen2.5-0.5B: synchronized greedy benchmark

Medians across measured runs; warmups excluded. Decimal MB. IQR is Q1–Q3.

| Format | Decode tokens/s (IQR) | TTFT ms | Peak MLX MB | Decode speedup | Memory reduction |
|---|---:|---:|---:|---:|---:|
| fp16 | 26.4 (25.1–28.8) | 255.7 | 1161.0 | 1.00x | 0.0% |
| int8 | 37.6 (35.8–38.3) | 289.4 | 679.1 | 1.43x | 41.5% |
| int4 | 43.6 (41.0–45.6) | 206.8 | 439.9 | 1.65x | 62.1% |

This runner synchronizes every token, so results describe this execution path,
not maximum MLX-LM streaming/serving throughput. TTFT includes prefill and
first-token selection; decode excludes that first token. Memory excludes
loading peak and is not whole-process RAM. EOS is ignored for fixed work.
No quality retention has been measured. One synthetic prompt length and
one process per format do not establish performance across workloads.
Thermal/background-load drift can remain despite randomized format order.
Do not compare these timings directly to the earlier streaming smoke runs.
