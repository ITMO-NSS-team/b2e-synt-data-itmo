"""Детерминированный источник случайности, адресуемый по координате.

Зачем не обычный ``random.Random``
---------------------------------
Корпус на 300 000 человек нельзя ни держать в памяти, ни писать целиком: 4 599
колонок каталога дают 1,4 млрд ячеек. Поэтому длинный хвост колонок **не
материализуется вовсе** — значение вычисляется на чтении из координаты
``(seed, витрина, колонка, номер строки)``.

Чтобы это работало, генератор обязан быть *адресуемым*: значение ячейки не
зависит ни от порядка обращения, ни от того, сколько колонок прочитали раньше.
Последовательный ``Random`` этого не даёт — он зависит от истории вызовов.

Здесь используется splitmix64 поверх 64-битного хэша координаты. Он векторизуется
в numpy (целая колонка на 300 000 строк считается одним выражением) и даёт
одинаковый результат в любом процессе, что важно: API может обслуживаться
несколькими воркерами, и они обязаны согласиться о значении.
"""
from __future__ import annotations

import hashlib

import numpy as np

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)


def key64(*parts: object) -> np.uint64:
    """64-битный хэш координаты. Стабилен между запусками и версиями Python."""
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return np.uint64(int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big"))


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Финализатор splitmix64: перемешивает счётчик в равномерные 64 бита."""
    x = (x ^ (x >> _S30)) * _M1
    x = (x ^ (x >> _S27)) * _M2
    return x ^ (x >> _S31)


def cells(base: np.uint64, index: np.ndarray) -> np.ndarray:
    """Равномерные 64-битные значения для набора номеров строк.

    ``base`` — хэш координаты колонки, ``index`` — номера строк. Результат зависит
    только от пары, поэтому чтение подмножества строк даёт те же значения, что и
    чтение всей колонки.
    """
    with np.errstate(over="ignore"):
        return _splitmix64(base + index.astype(np.uint64) * _GOLDEN)


def unit(base: np.uint64, index: np.ndarray) -> np.ndarray:
    """Равномерное [0, 1) по координате."""
    return (cells(base, index) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def pick(base: np.uint64, index: np.ndarray, size: int) -> np.ndarray:
    """Равномерный индекс в ``[0, size)``."""
    return (cells(base, index) % np.uint64(size)).astype(np.int64)


def weighted(base: np.uint64, index: np.ndarray, cum: np.ndarray) -> np.ndarray:
    """Индекс по кумулятивным весам. ``cum`` — неубывающий массив, cum[-1] == 1."""
    return np.searchsorted(cum, unit(base, index), side="right").clip(0, len(cum) - 1)


def normal(base: np.uint64, index: np.ndarray, mu: float = 0.0,
           sigma: float = 1.0) -> np.ndarray:
    """Нормальное распределение через преобразование Бокса — Мюллера.

    Две независимые равномерные величины берутся из разных подкоординат, а не из
    соседних строк: иначе соседние люди получили бы скоррелированные значения.
    """
    u1 = np.clip(unit(base, index), 1e-12, 1.0)
    u2 = unit(base ^ _GOLDEN, index)
    return mu + sigma * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def lognormal(base: np.uint64, index: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(normal(base, index, mu, sigma))


def exponential(base: np.uint64, index: np.ndarray, scale: float) -> np.ndarray:
    return -scale * np.log(np.clip(unit(base, index), 1e-12, 1.0))


def bernoulli(base: np.uint64, index: np.ndarray, p: float) -> np.ndarray:
    return unit(base, index) < p


def integers(base: np.uint64, index: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Целое в ``[lo, hi]`` включительно."""
    return lo + (cells(base, index) % np.uint64(hi - lo + 1)).astype(np.int64)


def zipf_cum(n: int, s: float = 1.07) -> np.ndarray:
    """Кумулятивные веса закона Ципфа для словаря из ``n`` элементов.

    Реальные фамилии и города распределены далеко не равномерно: «Иванов»
    встречается на три порядка чаще редкой фамилии. Равномерный выбор даёт
    неправдоподобно ровные частоты и ломает задачи поиска по имени.
    """
    w = 1.0 / np.power(np.arange(1, n + 1, dtype=np.float64), s)
    return np.cumsum(w / w.sum())


def shuffled(base: np.uint64, n: int) -> np.ndarray:
    """Детерминированная перестановка ``0..n-1``."""
    return np.argsort(cells(base, np.arange(n)))
