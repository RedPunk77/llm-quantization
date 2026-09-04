"""Потоковое продолжение текста локальной моделью через MLX"""
import argparse
import math
import sys

if __package__:
    from .model_config import add_model_argument, artifact_path
else:
    from model_config import add_model_argument, artifact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--format', choices=['fp16', 'int8', 'int4'], default='int4')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.max_tokens < 1 or not math.isfinite(args.temperature) or args.temperature < 0 or not args.prompt.strip():
        parser.error('Нужны непустой prompt, max-tokens>=1 и конечная temperature>=0')
    path = artifact_path(args.model, args.format)
    if not (path / 'source_revision.json').is_file():
        parser.error('Сначала подготовьте веса через run_smoke_matrix.py --model ' + args.model)
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(args.seed)
    model, tokenizer = load(str(path))
    # Обе модели — base: продолжаем текст без шаблона чата и system prompt
    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    last = None
    for response in stream_generate(model, tokenizer, prompt=ids,
                                    max_tokens=args.max_tokens,
                                    sampler=make_sampler(temp=args.temperature)):
        print(response.text, end='', flush=True)
        last = response
    print()
    if last:
        print(f'Сгенерировано {last.generation_tokens} токенов. '
              'Это интерактивный запуск, не benchmark.', file=sys.stderr)


if __name__ == '__main__':
    main()
