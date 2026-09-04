"""Учебное симметричное квантование групп весов без тензорных библиотек.

Диапазоны [-7, 7] и [-127, 127] симметричны относительно нуля: крайний
отрицательный код не используется. Это эталон вычислений, не ядро инференса.
"""
from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class QuantizedWeights:
    codes: tuple[int, ...]
    scales: tuple[float, ...]
    bits: int
    group_size: int

    def dequantize(self) -> list[float]:
        return [q * self.scales[i // self.group_size]
                for i, q in enumerate(self.codes)]


def quantize_symmetric(values: Sequence[float], bits: int = 4,
                       group_size: int = 128) -> QuantizedWeights:
    """Квантуем каждую последовательную группу по максимальному модулю веса.

    q = round(x / scale), scale = max(abs(x)) / qmax.
    round округляет середины к чётному. Для нулевой группы берём scale=1,
    чтобы избежать деления на ноль; восстановленные веса остаются нулевыми.
    """
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    weights = [float(v) for v in values]
    if not all(math.isfinite(v) for v in weights):
        raise ValueError("weights must be finite")
    qmax = (1 << (bits - 1)) - 1
    codes, scales = [], []
    for start in range(0, len(weights), group_size):
        group = weights[start:start + group_size]
        peak = max(abs(v) for v in group)
        scale = peak / qmax if peak else 1.0
        if scale == 0:
            raise ValueError("scale underflow")
        scales.append(scale)
        codes.extend(max(-qmax, min(qmax, round(v / scale))) for v in group)
    return QuantizedWeights(tuple(codes), tuple(scales), bits, group_size)


def pack_int4(codes: Sequence[int]) -> bytes:
    """Упаковываем два знаковых INT4-кода в байт в дополнительном коде.

    Первый код занимает младшие 4 бита. При нечётной длине старшие биты
    последнего байта нулевые. Длину и масштабы нужно хранить отдельно.
    """
    if any(not isinstance(q, int) or isinstance(q, bool) or not -8 <= q <= 7
           for q in codes):
        raise ValueError("INT4 codes must be integers in [-8, 7]")
    return bytes((codes[i] & 15) |
                 (((codes[i + 1] & 15) if i + 1 < len(codes) else 0) << 4)
                 for i in range(0, len(codes), 2))


def unpack_int4(payload: bytes, count: int) -> tuple[int, ...]:
    """Восстанавливаем count кодов; размер буфера должен точно соответствовать длине."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if len(payload) != (count + 1) // 2:
        raise ValueError("payload size does not match count")
    result = []
    for i in range(count):
        q = (payload[i // 2] >> (4 * (i % 2))) & 15
        result.append(q - 16 if q >= 8 else q)
    return tuple(result)
