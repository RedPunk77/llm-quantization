"""Подготовка FP16/INT8/INT4 и проверка генерации в отдельных процессах"""
import argparse
from pathlib import Path
import subprocess
import sys

if __package__:
    from .model_config import add_model_argument
else:
    from model_config import add_model_argument

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    args = parser.parse_args()
    def run(script, *extra):
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                        '--model', args.model, *extra], cwd=ROOT, check=True)
    # Первый вызов скачивает исходную модель, остальные используют локальные веса
    run('smoke_mlx.py')
    for bits in (8, 4):
        run('quantize_mlx.py', '--bits', str(bits))
        run('smoke_mlx.py', '--format', f'int{bits}')
    print('Все три варианта готовы. Результаты проверки находятся в results/.')
    print('Для оценки скорости используйте benchmark_mlx.py; качества — evaluate_ppl.py.')


if __name__ == '__main__':
    main()
