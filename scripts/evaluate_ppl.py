"""Сравниваем perplexity на фрагменте WikiText-2 с перекрывающимися окнами"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .model_config import MODELS, add_model_argument, artifact_path
else:
    from model_config import MODELS, add_model_argument, artifact_path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))


def windows(length, context, stride):
    """Возвращаем границы окна и первую цель; токены 1..length-1 учитываем один раз"""
    if length < 2 or context < 2 or not 1 <= stride < context:
        raise ValueError('Require length/context >=2 and 1 <= stride < context')
    previous_end = 1
    for start in range(0, length - 1, stride):
        end = min(start + context, length)
        first_target = max(previous_end, start + 1)
        if first_target < end:
            yield start, end, first_target
        previous_end = end
        if end == length:
            break


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def wilson_interval(correct, total, z=1.96):
    """95% Wilson interval для доли правильных предсказаний"""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError('Некорректные correct/total')
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def paired_retention_interval(baseline, candidate, samples=10000, seed=42):
    """Парный bootstrap CI для отношения accuracy candidate / accuracy FP16"""
    if len(baseline) != len(candidate) or not baseline or sum(baseline) == 0:
        raise ValueError('Нужны парные результаты и ненулевая accuracy FP16')
    import numpy as np
    # Для пары бинарных исходов достаточно четырёх совместных частот
    counts = [0, 0, 0, 0]
    for base, current in zip(baseline, candidate):
        counts[2 * int(bool(base)) + int(bool(current))] += 1
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(baseline), np.asarray(counts) / len(baseline), size=samples)
    base_correct = draws[:, 2] + draws[:, 3]
    current_correct = draws[:, 1] + draws[:, 3]
    ratios = current_correct[base_correct > 0] / base_correct[base_correct > 0]
    low, high = np.quantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def prepare(directory, limit, model='smollm2-135m'):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    repo = 'Salesforce/wikitext'
    revision = 'b08601e04326c79dfdd32d625aee71d232d685c3'
    filename = 'wikitext-2-raw-v1/test-00000-of-00001.parquet'
    path = hf_hub_download(repo, filename, repo_type='dataset', revision=revision)
    text = '\n\n'.join(pq.read_table(path, columns=['text'])['text'].to_pylist())
    baseline = artifact_path(model)
    tokenizer = AutoTokenizer.from_pretrained(str(baseline), local_files_only=True)
    all_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    ids = all_ids[:limit] if limit else all_ids
    data = {'dataset': repo, 'revision': revision, 'file': filename,
            'parquet_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'text_join': 'two newlines, retain all rows',
            'selection': f'first {len(ids)} tokens of test split',
            'full_token_count': len(all_ids), 'token_ids': ids, 'tokens_sha256': digest(ids),
            'model_source': json.loads((baseline / 'source_revision.json').read_text()),
            'add_special_tokens': False}
    (directory / 'corpus.json').write_text(json.dumps(data) + '\n')
    return data


def worker(args):
    import mlx.core as mx
    from mlx_lm import load
    corpus = json.loads((args.directory / 'corpus.json').read_text())
    suffix = args.format if args.format == 'fp16' else args.format + '-g64'
    path = artifact_path(args.model, args.format)
    source = json.loads((path / 'source_revision.json').read_text())
    if source != corpus['model_source']:
        raise ValueError('Model source does not match baseline')
    model, tokenizer = load(str(path))
    model.eval()
    mx.eval(model.parameters())
    ids = corpus['token_ids']
    if len(ids) > 1 and max(ids) >= model(mx.array([ids[:1]])).shape[-1]:
        raise ValueError('Invalid token IDs')
    result = {'status': 'running', 'format': args.format, 'source': source,
              'tokens_sha256': corpus['tokens_sha256'], 'context': args.context,
              'stride': args.stride, 'scored_tokens': 0, 'nll_sum': 0.0,
              'top1_correct': 0, 'top1_correct_by_token': [],
              'config': json.loads((path / 'config.json').read_text()),
              'versions': {p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm', 'transformers', 'pyarrow')},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'windows': []}
    output = args.directory / (args.format + '.json')
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        for index, (start, end, first) in enumerate(windows(len(ids), args.context, args.stride)):
            # Позиция p предсказывает p+1; цели из перекрывающегося контекста повторно не считаем
            logits = model(mx.array([ids[start:end-1]]))
            scores = logits[0, first-start-1:, :].astype(mx.float32)
            targets = mx.array(ids[first:end])
            selected = mx.take_along_axis(scores, targets[:, None], axis=-1).squeeze(-1)
            loss_sum = mx.sum(mx.logsumexp(scores, axis=-1) - selected)
            correct = (mx.argmax(scores, axis=-1) == targets).tolist()
            value = float(loss_sum.item())
            if not math.isfinite(value):
                raise ValueError('Nonfinite NLL')
            count = end - first
            result['windows'].append({'start': start, 'end': end, 'first_target': first,
                                      'scored_tokens': count, 'nll_sum': value})
            result['nll_sum'] += value
            result['scored_tokens'] += count
            result['top1_correct'] += sum(correct)
            result['top1_correct_by_token'].extend(bool(item) for item in correct)
            del logits, scores, targets, selected, loss_sum
            save()
            if index % 10 == 0:
                print(f'{args.format}: {result["scored_tokens"]}/{len(ids)-1} targets', flush=True)
        if result['scored_tokens'] != len(ids) - 1:
            raise ValueError('Missing or duplicated target tokens')
        result['mean_nll'] = result['nll_sum'] / result['scored_tokens']
        result['perplexity'] = math.exp(result['mean_nll'])
        result['top1_accuracy'] = result['top1_correct'] / result['scored_tokens']
        result['top1_accuracy_wilson_95'] = wilson_interval(
            result['top1_correct'], result['scored_tokens'])
        result['status'] = 'complete'
        save()
    except Exception as exc:
        result.update(status='failed', error=str(exc))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-tokens', type=int, default=16384, help='Fixed test prefix; 0 uses full split')
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--format', choices=['fp16', 'int8', 'int4'], help=argparse.SUPPRESS)
    parser.add_argument('--directory', type=Path, help=argparse.SUPPRESS)
    add_model_argument(parser)
    args = parser.parse_args()
    if args.max_tokens < 0 or args.max_tokens == 1 or not 1 <= args.stride < args.context or args.context > 2048:
        parser.error('Require max-tokens=0 or >=2; 1<=stride<context<=2048')
    if args.format:
        worker(args)
        return
    directory = ROOT / 'results' / ('perplexity-' + args.model + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True)
    corpus = prepare(directory, args.max_tokens, args.model)
    (directory / 'arguments.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    print(f'Prepared {len(corpus["token_ids"])} tokens; corpus revision {corpus["revision"]}', flush=True)
    runs = []
    for fmt in ('fp16', 'int8', 'int4'):
        command = [sys.executable, str(Path(__file__).resolve()), '--model', args.model, '--format', fmt, '--directory', str(directory),
                   '--context', str(args.context), '--stride', str(args.stride)]
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            (directory / (fmt + '-failure.json')).write_text(json.dumps({'returncode': completed.returncode}))
            raise SystemExit(f'{fmt} failed; partial results in {directory}')
        runs.append(json.loads((directory / (fmt + '.json')).read_text()))
    lines = [f'# {corpus["model_source"]["repo"]}: WikiText-2 test-prefix perplexity', '',
             f'{len(corpus["token_ids"])-1} scored tokens; context={args.context}, stride={args.stride}.', '',
             '| Format | Perplexity | Top-1 accuracy (95% CI) | Quality retention vs FP16 (95% paired bootstrap CI) |',
             '|---|---:|---:|---:|']
    for run in runs:
        if run['status'] != 'complete' or run['scored_tokens'] != runs[0]['scored_tokens']:
            raise ValueError('Incomplete or incomparable results')
        change = run['perplexity'] / runs[0]['perplexity'] - 1
        accuracy_ci = run['top1_accuracy_wilson_95']
        if run['format'] == 'fp16':
            retention = (1.0, 1.0, 1.0)
        else:
            low, high = paired_retention_interval(
                runs[0]['top1_correct_by_token'], run['top1_correct_by_token'])
            retention = (run['top1_accuracy'] / runs[0]['top1_accuracy'], low, high)
        lines.append(
            f'| {run["format"]} | {run["perplexity"]:.3f} ({change:+.2%}) | '
            f'{run["top1_accuracy"]:.2%} ({accuracy_ci[0]:.2%}–{accuracy_ci[1]:.2%}) | '
            f'{retention[0]:.2%} ({retention[1]:.2%}–{retention[2]:.2%}) |')
    lines += ['', 'Quality retention здесь означает отношение next-token top-1 accuracy к FP16.',
              'Это узкая языковая метрика на WikiText-2, а не процент всех способностей модели.',
              'PPL change is not a percentage of quality retained.',
              'This fixed prefix is a pilot, not a full-corpus published benchmark score.',
              'Use validation data for tuning; do not tune configurations on this test result.',
              'Possible model pretraining overlap with WikiText is not ruled out.',
              'Dataset: https://huggingface.co/datasets/Salesforce/wikitext']
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Saved: {directory / "summary.md"}')


if __name__ == '__main__':
    main()
