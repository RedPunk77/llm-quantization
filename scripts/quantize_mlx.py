"""Создаём affine INT8/INT4 из локальной FP16-модели."""
import argparse
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

if __package__:
    from .model_config import MODELS, add_model_argument, artifact_path
else:
    from model_config import MODELS, add_model_argument, artifact_path

ROOT = Path(__file__).resolve().parents[1]


def prepare_quantized(baseline, target, bits, group_size, convert_fn):
    source = json.loads((baseline / 'source_revision.json').read_text())
    spec = {'source': source, 'bits': bits, 'group_size': group_size,
            'mode': 'affine', 'non_quantized_dtype': 'float16'}
    marker = target / 'quantization_run.json'
    if marker.is_file():
        if json.loads(marker.read_text()) != spec:
            raise ValueError('Existing quantized artifact has different settings')
        return
    if target.exists():
        backup = target.with_name(target.name + '.incomplete-' +
                                 datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        target.rename(backup)
        print(f'Preserved incomplete artifact: {backup}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.quantization-', dir=target.parent) as tmp:
        staged = Path(tmp) / 'model'
        convert_fn(str(baseline), mlx_path=str(staged), quantize=True,
                   q_bits=bits, q_group_size=group_size, q_mode='affine', dtype='float16')
        config = json.loads((staged / 'config.json').read_text())
        quant = config.get('quantization', config.get('quantization_config', {}))
        if quant.get('bits') != bits or quant.get('group_size') != group_size or quant.get('mode', 'affine') != 'affine':
            raise ValueError('Saved quantization config does not match requested settings')
        (staged / 'source_revision.json').write_text(json.dumps(source, indent=2) + '\n')
        (staged / 'quantization_run.json').write_text(json.dumps(spec, indent=2) + '\n')
        staged.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bits', type=int, choices=[4, 8], required=True)
    parser.add_argument('--group-size', type=int, choices=[32, 64, 128], default=64)
    add_model_argument(parser)
    args = parser.parse_args()
    from mlx_lm import convert
    baseline = artifact_path(args.model)
    target = artifact_path(args.model, f'int{args.bits}', args.group_size)
    prepare_quantized(baseline, target, args.bits, args.group_size, convert)
    print(f'Ready: {target}')


if __name__ == '__main__':
    main()
