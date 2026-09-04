# SmolLM2-135M: synchronized greedy benchmark

Medians across measured runs; warmups excluded. Decimal MB. IQR is Q1–Q3.

| Format | Decode tokens/s (IQR) | TTFT ms | Peak MLX MB | Decode speedup | Memory reduction |
|---|---:|---:|---:|---:|---:|
| fp16 | 56.0 (41.6–57.3) | 46.4 | 363.0 | 1.00x | 0.0% |
| int8 | 61.5 (57.3–78.2) | 48.3 | 248.5 | 1.10x | 31.5% |
| int4 | 87.5 (79.3–91.4) | 47.0 | 181.7 | 1.56x | 49.9% |

This runner synchronizes every token, so results describe this execution path,
not maximum MLX-LM streaming/serving throughput. TTFT includes prefill and
first-token selection; decode excludes that first token. Memory excludes
loading peak and is not whole-process RAM. EOS is ignored for fixed work.
No quality retention has been measured. One synthetic prompt length and
one process per format do not establish performance across workloads.
Thermal/background-load drift can remain despite randomized format order.
Do not compare these timings directly to the earlier streaming smoke runs.
