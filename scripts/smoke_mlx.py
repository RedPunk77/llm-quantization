"""Один проверочный запуск FP16 или квантованной модели без оценки устойчивой скорости."""
import os
from pathlib import Path

if __package__:
    from .model_config import MODELS, add_model_argument, artifact_path
else:
    from model_config import MODELS, add_model_argument, artifact_path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

import argparse
import importlib.metadata
import json
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from time import perf_counter



def prepare_artifact(artifact, repo, revision, convert_fn, snapshot_fn):
    """Конвертируем локальную ревизию; готовой считаем только полностью сохранённую модель."""
    provenance = artifact / 'source_revision.json'
    if provenance.is_file():
        source = json.loads(provenance.read_text())
        if source['repo'] != repo or (revision and revision != source['revision']):
            raise ValueError('Local artifact differs from requested source')
        return source

    # Передаём локальный путь: при сохранении MLX не должен повторно искать ветку main.
    snapshot = Path(snapshot_fn(repo_id=repo, revision=revision, allow_patterns=[
        '*.safetensors', '*.json', '*.model', '*.tiktoken', '*.txt', '*.jinja',
    ]))
    source = {'repo': repo, 'revision': snapshot.name}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        backup = artifact.with_name(artifact.name + '.incomplete-' +
                                   datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        artifact.rename(backup)
        print(f'Preserved incomplete artifact: {backup}')
    with tempfile.TemporaryDirectory(prefix='.conversion-', dir=artifact.parent) as staging:
        destination = Path(staging) / 'model'
        convert_fn(str(snapshot), mlx_path=str(destination), dtype='float16')
        (destination / 'source_revision.json').write_text(json.dumps(source, indent=2) + '\n')
        destination.rename(artifact)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', help='Immutable Hub revision to reproduce a run')
    parser.add_argument('--format', choices=['fp16', 'int8', 'int4'], default='fp16')
    parser.add_argument('--group-size', type=int, choices=[32, 64, 128], default=64)
    add_model_argument(parser)
    args = parser.parse_args()
    try:
        import mlx.core as mx
    except ImportError as exc:
        if 'No Metal device' in str(exc):
            raise SystemExit(
                'MLX cannot access Metal in this session. Run this script in '
                'a normal VS Code/macOS terminal. No measurements were taken.'
            ) from exc
        raise
    from huggingface_hub import snapshot_download
    from mlx.utils import tree_flatten
    from mlx_lm import convert, load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    repo = MODELS[args.model]
    name = args.model + '-' + args.format
    if args.format != 'fp16':
        name += f'-g{args.group_size}'
    artifact = ROOT / 'models' / name
    if args.format == 'fp16':
        source = prepare_artifact(artifact, repo, args.revision, convert, snapshot_download)
        quantization = None
    else:
        spec = json.loads((artifact / 'quantization_run.json').read_text())
        source = spec['source']
        quantization = json.loads((artifact / 'config.json').read_text()).get('quantization_config')
        if spec['bits'] != int(args.format[3:]) or spec['group_size'] != args.group_size:
            raise ValueError('Quantization settings mismatch')
        if args.revision and source['revision'] != args.revision:
            raise ValueError('Source revision mismatch')
    model, tokenizer = load(str(artifact))
    mx.eval(model.parameters())
    dtypes = sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())})
    if args.format == 'fp16' and dtypes != ['mlx.core.float16']:
        raise RuntimeError(f'Expected FP16 parameters, found {dtypes}')
    if args.format != 'fp16' and 'mlx.core.uint32' not in dtypes:
        raise RuntimeError('Expected packed integer weights in quantized model')
    prompt = 'The main purpose of a computer is'
    prompt_ids = tokenizer.encode(prompt)
    mx.synchronize()
    mx.reset_peak_memory()
    start = perf_counter()
    first_token_seconds = None
    text = ''
    last = None
    for response in stream_generate(model, tokenizer, prompt=prompt_ids,
                                    max_tokens=64, sampler=make_sampler(temp=0.0)):
        if first_token_seconds is None:
            mx.synchronize()
            first_token_seconds = perf_counter() - start
        text += response.text
        last = response
    mx.synchronize()
    elapsed = perf_counter() - start
    if last is None:
        raise RuntimeError('No generation response')
    result = {
        'run_kind': 'smoke_test_not_benchmark',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'source': source,
        'weight_dtype': 'float16' if args.format == 'fp16' else args.format + '_affine_packed',
        'quantization_config': quantization,
        'floating_parameter_dtypes': [d for d in dtypes if 'float' in d],
        'parameter_dtypes': dtypes,
        'platform': platform.platform(),
        'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'python': platform.python_version(),
        'versions': {p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm', 'transformers', 'huggingface-hub')},
        'prompt': prompt,
        'prompt_token_ids': prompt_ids,
        'max_new_tokens': 64,
        'temperature': 0.0,
        'batch_size': 1,
        'warmup_runs': 0,
        'measured_runs': 1,
        'generated_text': text,
        'generation_tokens_backend': last.generation_tokens,
        'prompt_tokens_backend': last.prompt_tokens,
        'backend_prompt_tokens_per_second': last.prompt_tps,
        'backend_generation_tokens_per_second': last.generation_tps,
        'first_stream_response_seconds': first_token_seconds,
        'generation_wall_seconds': elapsed,
        'mlx_peak_allocated_bytes_after_load': mx.get_peak_memory(),
        'weight_files_bytes': sum(p.stat().st_size for p in artifact.glob('*.safetensors')),
        'limitations': [
            'Single cold run; includes first-use overhead; no speedup claim.',
            'Memory is MLX allocator peak during generation, including resident weights; excludes loading peak and is not process RSS.',
            'First stream response timing is not a validated serving TTFT.',
            'Throughput fields use installed MLX-LM definitions.',
            'No quality evaluation; base model receives raw completion prompt.',
        ],
    }
    destination = ROOT / 'results' / (name + '-smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'Saved: {destination}')


if __name__ == '__main__':
    main()
