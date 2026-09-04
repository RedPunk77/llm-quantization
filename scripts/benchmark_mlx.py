"""Измеряем batch=1 с синхронизацией GPU после каждого greedy-токена."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter

if __package__:
    from .model_config import MODELS, add_model_argument, artifact_path
else:
    from model_config import MODELS, add_model_argument, artifact_path

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ('fp16', 'int8', 'int4')


def summarize(rows):
    result = {}
    for key in ('ttft_seconds', 'decode_tokens_per_second', 'total_seconds', 'peak_mlx_bytes'):
        values = sorted(row[key] for row in rows)
        if not values:
            raise ValueError('No measurements')
        quartiles = statistics.quantiles(values, n=4, method='inclusive') if len(values) > 1 else values * 3
        result[key] = {'median': statistics.median(values), 'min': values[0],
                       'max': values[-1], 'q1': quartiles[0], 'q3': quartiles[2]}
    return result


def measure(model, prompt, output_tokens, mx, make_cache):
    # У каждого запуска свой KV-cache: префикс не переиспользуется между измерениями.
    cache = make_cache(model)
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    logits = model(prompt[None], cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    mx.synchronize()
    first = perf_counter()
    tokens = [token]
    del logits
    for _ in range(output_tokens - 1):
        logits = model(token[:, None], cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        mx.synchronize()
        tokens.append(token)
        del logits
    end = perf_counter()
    peak = mx.get_peak_memory()
    # Перенос в Python не входит во время. EOS не останавливает цикл: длина фиксирована.
    ids = [int(t.item()) for t in tokens]
    return {'ttft_seconds': first - start,
            'decode_tokens_per_second': (output_tokens - 1) / (end - first),
            'decode_seconds': end - first, 'total_seconds': end - start,
            'peak_mlx_bytes': peak, 'generated_tokens': len(ids),
            'generated_token_ids': ids}


def worker(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    suffix = args.format if args.format == 'fp16' else args.format + '-g64'
    artifact = artifact_path(args.model, args.format)
    source = json.loads((artifact / 'source_revision.json').read_text())
    config = json.loads((artifact / 'config.json').read_text())
    model, tokenizer = load(str(artifact))
    mx.eval(model.parameters())
    params = tree_flatten(model.parameters())
    dtypes = sorted({str(v.dtype) for _, v in params})
    quant = config.get('quantization_config', config.get('quantization'))
    if args.format == 'fp16':
        if quant or dtypes != ['mlx.core.float16']:
            raise ValueError('Baseline must have FP16 parameters and no quantization')
    elif not quant or quant.get('bits') != int(args.format[3:]) or quant.get('group_size') != 64 or quant.get('mode', 'affine') != 'affine':
        raise ValueError('Unexpected quantization configuration')
    del params
    # Повторяем текст ради фиксированной длины. Для оценки качества этот вход не используется.
    text = 'A computer stores information and performs calculations. '
    ids = tokenizer.encode(text * (args.input_tokens + 1), add_special_tokens=False)[:args.input_tokens]
    if len(ids) != args.input_tokens:
        raise ValueError('Prompt too short')
    prompt = mx.array(ids)
    mx.eval(prompt)
    mx.random.seed(42)
    data = {
        'kind': 'synchronized_greedy_benchmark_v1', 'status': 'running',
        'model': args.model, 'format': args.format, 'source': source, 'quantization_config': quant,
        'parameter_dtypes': dtypes, 'platform': platform.platform(),
        'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'power_state': subprocess.check_output(['pmset', '-g', 'batt'], text=True).strip(),
        'python': platform.python_version(),
        'versions': {p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm', 'transformers', 'huggingface-hub')},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_token_ids': ids, 'input_tokens': args.input_tokens,
        'output_tokens': args.output_tokens, 'batch_size': 1,
        'warmup_runs': args.warmup, 'repetitions': args.repetitions,
        'seed': 42, 'kv_cache': 'fresh per run, unquantized',
        'eos_policy': 'ignored to enforce fixed output length',
        'timing_policy': 'synchronize after every greedy token; exclude tokenization/load/host decoding',
        'memory_metric': 'peak MLX allocation during generation, includes resident weights; not process RSS',
        'weight_files_bytes': sum(p.stat().st_size for p in artifact.glob('*.safetensors')),
        'artifact_files_bytes': sum(p.stat().st_size for p in artifact.rglob('*') if p.is_file()),
        'warmup_observations': [], 'observations': [],
    }
    def save():
        args.output.write_text(json.dumps(data, indent=2) + '\n')
    save()
    try:
        for phase, count in [('warmup_observations', args.warmup), ('observations', args.repetitions)]:
            for i in range(count):
                row = measure(model, prompt, args.output_tokens, mx, make_prompt_cache)
                data[phase].append(row)
                save()
                print(f'{args.format} {phase} {i+1}/{count}: {row["decode_tokens_per_second"]:.1f} tokens/s', flush=True)
        data['summary'] = summarize(data['observations'])
        data['status'] = 'complete'
        save()
    except Exception as exc:
        data['status'] = 'failed'
        data['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


def report(directory):
    runs = [json.loads((directory / (fmt + '.json')).read_text()) for fmt in FORMATS]
    baseline = runs[0]
    for run in runs:
        if run['status'] != 'complete':
            raise ValueError('Cannot summarize incomplete run')
        for key in ('source', 'input_token_ids', 'output_tokens', 'versions', 'timing_policy', 'script_sha256'):
            if run[key] != baseline[key]:
                raise ValueError(f'Incomparable runs: {key}')
    lines = [f'# {baseline["source"]["repo"]}: synchronized greedy benchmark', '',
             'Medians across measured runs; warmups excluded. Decimal MB. IQR is Q1–Q3.', '',
             '| Format | Decode tokens/s (IQR) | TTFT ms | Peak MLX MB | Decode speedup | Memory reduction |',
             '|---|---:|---:|---:|---:|---:|']
    for run in runs:
        s = run['summary']
        speed = s['decode_tokens_per_second']
        peak = s['peak_mlx_bytes']['median']
        ratio = speed['median'] / baseline['summary']['decode_tokens_per_second']['median']
        saving = 1 - peak / baseline['summary']['peak_mlx_bytes']['median']
        lines.append(f'| {run["format"]} | {speed["median"]:.1f} ({speed["q1"]:.1f}–{speed["q3"]:.1f}) | {s["ttft_seconds"]["median"]*1000:.1f} | {peak/1e6:.1f} | {ratio:.2f}x | {saving:.1%} |')
    lines += ['', 'This runner synchronizes every token, so results describe this execution path,',
              'not maximum MLX-LM streaming/serving throughput. TTFT includes prefill and',
              'first-token selection; decode excludes that first token. Memory excludes',
              'loading peak and is not whole-process RAM. EOS is ignored for fixed work.',
              'No quality retention has been measured. One synthetic prompt length and',
              'one process per format do not establish performance across workloads.',
              'Thermal/background-load drift can remain despite randomized format order.',
              'Do not compare these timings directly to the earlier streaming smoke runs.']
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--format', choices=FORMATS, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    add_model_argument(parser)
    args = parser.parse_args()
    if args.input_tokens < 1 or args.output_tokens < 2 or args.warmup < 3 or args.repetitions < 10:
        parser.error('Require input>=1, output>=2, warmup>=3, repetitions>=10')
    if args.format:
        if args.output is None:
            parser.error('Worker requires --output')
        worker(args)
        return
    directory = ROOT / 'results' / ('benchmark-' + args.model + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True)
    order = list(FORMATS)
    random.Random(42).shuffle(order)
    (directory / 'manifest.json').write_text(json.dumps({'order': order, 'arguments': vars(args)}, default=str, indent=2) + '\n')
    for fmt in order:
        output = directory / (fmt + '.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--model', args.model, '--format', fmt, '--output', str(output),
                   '--input-tokens', str(args.input_tokens), '--output-tokens', str(args.output_tokens),
                   '--warmup', str(args.warmup), '--repetitions', str(args.repetitions)]
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            (directory / (fmt + '-failure.json')).write_text(json.dumps({'format': fmt, 'returncode': completed.returncode}) + '\n')
            raise SystemExit(f'{fmt} failed; partial results preserved in {directory}')
    report(directory)
    print(f'Saved report: {directory / "summary.md"}')


if __name__ == '__main__':
    main()
