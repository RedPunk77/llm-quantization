# LLM Quantization & Efficient Inference

Пет-проект про посттренировочное квантование LLM на Apple Silicon (M2) 🍏

Я сравнил FP16, INT8 и INT4 для Qwen2.5-0.5B через MLX, добавил CLI для инференса и собрал воспроизводимые замеры скорости, памяти и качества

## Что получилось

| Формат | Decode, токенов/с | TTFT, мс | Пик MLX, МБ | Ускорение | Экономия памяти | Perplexity | Top-1 accuracy | Сохранение качества |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 26,4 | 255,7 | 1161,0 | 1,00× | — | 12,764 | 48,70% | 100,00% |
| INT8, g64 | 37,6 | 289,4 | 679,1 | **1,43×** | 41,5% | 12,776 | 48,61% | **99,81%** |
| INT4, g64 | 43,6 | 206,8 | 439,9 | **1,65×** | **62,1%** | 15,061 | 46,13% | **94,72%** |

INT8 оказался самым оптимальным вариантом: ускорение x1,43, экономия памяти 41,5% и 99,81% сохранённой next-token accuracy относительно FP16

INT4 дал максимальную экономию и скорость (x1,65), но это ударило по качеству: perplexity выросла на 18%, сохранение next-token accuracy составило 94,72%

### Условия замеров

- Apple M2, 16 ГБ unified memory
- Qwen2.5-0.5B
- Affine group-wise квантование весов, group size=64, неквантованные параметры в FP16
- Batch size=1, 128 входных и 128 выходных токенов
- 3 прогрева и 10 измерений на каждый формат
- В таблице указаны медианы

Качество измерялось на первых 16 384 токенах test split датасета WikiText-2 с окном 512 и шагом 256

`Quality retention = accuracy_quantized / accuracy_fp16`, где accuracy - доля точных top-1 предсказаний следующего токена

Это конкретная языковая метрика на WikiText-2

- [Полный benchmark скорости и памяти](results/benchmark-qwen2.5-0.5b-20260904T115928621970Z/summary.md)
- [Perplexity, accuracy и доверительные интервалы](results/perplexity-qwen2.5-0.5b-20260904T223510218992Z/summary.md)
- [Подробный протокол](docs/experiment_protocol.md)

## Быстрый старт

Нужен Mac с Apple Silicon и Python 3.10+

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mlx.lock.txt
python scripts/run_smoke_matrix.py --model qwen2.5-0.5b
```

Последняя команда скачает исходную модель и подготовит локальные FP16, INT8 и INT4 варианты

Веса, кэш и `.venv` не хранятся в Git, поэтому для моделей понадобится несколько гигабайт свободного места

## Инференс

```sh
python scripts/infer.py \
  --model qwen2.5-0.5b \
  --format int4 \
  --prompt "Квантование нейросетей позволяет" \
  --max-tokens 128
```

Можно выбрать `fp16`, `int8` или `int4`, а также передать `--temperature` и `--seed`

Здесь используется base-модель, поэтому CLI продолжает текст, а не ведёт диалог как Instruct-модель

## Повторить эксперименты

```sh
# Скорость и память
python scripts/benchmark_mlx.py --model qwen2.5-0.5b

# Perplexity, top-1 accuracy и quality retention
python scripts/evaluate_ppl.py --model qwen2.5-0.5b
```

Каждый формат запускается в отдельном процессе, а исходные наблюдения, версии библиотек и параметры эксперимента сохраняются в `results/`

## Что внутри

- `src/llm_quantization/reference.py`: учебная реализация симметричного группового квантования и упаковки INT4
- `scripts/quantize_mlx.py`: настоящее affine-квантование моделей через MLX
- `scripts/benchmark_mlx.py`: замеры TTFT, decode throughput и памяти
- `scripts/evaluate_ppl.py`: perplexity, top-1 accuracy и quality retention
- `scripts/infer.py`: потоковый CLI для локального инференса
- `tests/`: тесты математики, упаковки, оконной оценки и выбора модели
- `results/`: сырые данные и итоговые таблицы

## Проверка тестов

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Документация

Модели: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) и [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)

Данные: [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)
