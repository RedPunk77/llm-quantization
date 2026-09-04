"""Общий список моделей и имена каталогов с весами."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'smollm2-135m': 'HuggingFaceTB/SmolLM2-135M',
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B',
}


def add_model_argument(parser):
    # Старые команды продолжают работать со SmolLM2; Qwen выбирается явно.
    parser.add_argument('--model', choices=MODELS, default='smollm2-135m')


def artifact_path(model, precision='fp16', group_size=64):
    if model not in MODELS or precision not in ('fp16', 'int8', 'int4'):
        raise ValueError('Неизвестная модель или точность')
    suffix = precision if precision == 'fp16' else f'{precision}-g{group_size}'
    return ROOT / 'models' / f'{model}-{suffix}'
