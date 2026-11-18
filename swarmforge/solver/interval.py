from __future__ import annotations

import numpy as np

EPS = float(np.finfo(float).eps)


def _dn(x, k: int = 1):
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        y = np.nextafter(x - np.abs(x) * (k * EPS), -np.inf)
    return np.where(np.isnan(x), -np.inf, np.where(np.isinf(x), x, y))


def _up(x, k: int = 1):
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        y = np.nextafter(x + np.abs(x) * (k * EPS), np.inf)
    return np.where(np.isnan(x), np.inf, np.where(np.isinf(x), x, y))


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=float)


    @property
    def shape(self):
        return np.broadcast_shapes(self.lo.shape, self.hi.shape)

    def __getitem__(self, index) -> "I":
        return I(self.lo[index], self.hi[index])

    def broadcast_to(self, shape) -> "I":
        return I(np.broadcast_to(self.lo, shape), np.broadcast_to(self.hi, shape))

    @property
    def width(self):
        return self.hi - self.lo


    def __neg__(self) -> "I":
        return I(-self.hi, -self.lo)

    def __add__(self, other) -> "I":
        o = as_i(other)
        with np.errstate(invalid="ignore"):
            return I(_dn(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __sub__(self, other) -> "I":
        o = as_i(other)
        with np.errstate(invalid="ignore"):
            return I(_dn(self.lo - o.hi), _up(self.hi - o.lo))

    def __rsub__(self, other) -> "I":
        return as_i(other) - self

    def __mul__(self, other) -> "I":
        if not isinstance(other, I):
            c = np.asarray(other, dtype=float)
            with np.errstate(invalid="ignore", over="ignore"):
                p, q = self.lo * c, self.hi * c
            p = np.where(np.isnan(p), 0.0, p)
            q = np.where(np.isnan(q), 0.0, q)
            return I(_dn(np.minimum(p, q)), _up(np.maximum(p, q)))
        with np.errstate(invalid="ignore", over="ignore"):
            ps = [self.lo * other.lo, self.lo * other.hi,
                  self.hi * other.lo, self.hi * other.hi]
        ps = [np.where(np.isnan(p), 0.0, p) for p in ps]
        lo = np.minimum(np.minimum(ps[0], ps[1]), np.minimum(ps[2], ps[3]))
        hi = np.maximum(np.maximum(ps[0], ps[1]), np.maximum(ps[2], ps[3]))
        return I(_dn(lo), _up(hi))

    __rmul__ = __mul__

    def __truediv__(self, other) -> "I":
        if not isinstance(other, I):
            c = np.asarray(other, dtype=float)
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                p, q = self.lo / c, self.hi / c
            return I(_dn(np.minimum(p, q)), _up(np.maximum(p, q)))
        return self * recip(other)


def as_i(x) -> I:
    return x if isinstance(x, I) else I(x)


def zeros(shape) -> I:
    z = np.zeros(shape)
    return I(z, z)


def stack(items, axis: int = 0) -> I:
    items = [as_i(x) for x in items]
    return I(np.stack([x.lo for x in items], axis=axis),
             np.stack([x.hi for x in items], axis=axis))


def concat(items, axis: int = 0) -> I:
    items = [as_i(x) for x in items]
    return I(np.concatenate([x.lo for x in items], axis=axis),
             np.concatenate([x.hi for x in items], axis=axis))


def expand(x: I, axis: int) -> I:
    return I(np.expand_dims(x.lo, axis), np.expand_dims(x.hi, axis))


def mig(x: I):
    return np.where((x.lo <= 0.0) & (x.hi >= 0.0), 0.0,
                    np.minimum(np.abs(x.lo), np.abs(x.hi)))


def mag(x: I):
    return np.maximum(np.abs(x.lo), np.abs(x.hi))


def absolute(x: I) -> I:
    return I(mig(x), mag(x))


def sqr(x: I) -> I:
    a, b = mig(x), mag(x)
    with np.errstate(over="ignore"):
        return I(np.maximum(_dn(a * a), 0.0), _up(b * b))


def sqrt(x: I) -> I:
    with np.errstate(invalid="ignore"):
        lo = np.sqrt(np.maximum(x.lo, 0.0))
        hi = np.sqrt(np.maximum(x.hi, 0.0))
    return I(np.maximum(_dn(lo, 2), 0.0), _up(hi, 2))


def exp(x: I) -> I:
    with np.errstate(over="ignore"):
        return I(np.maximum(_dn(np.exp(x.lo), 4), 0.0), _up(np.exp(x.hi), 4))


def recip(x: I) -> I:
    ok = (x.lo > 0.0) | (x.hi < 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo, hi = 1.0 / x.hi, 1.0 / x.lo
    return I(np.where(ok, _dn(lo), -np.inf), np.where(ok, _up(hi), np.inf))


def powi(x: I, n: int) -> I:
    if n == 0:
        return I(np.ones(x.shape))
    if n < 0:
        return recip(powi(x, -n))
    if n == 1:
        return x
    if n % 2 == 0:
        a, b = mig(x), mag(x)
        with np.errstate(over="ignore"):
            return I(np.maximum(_dn(a ** n, n), 0.0), _up(b ** n, n))
    with np.errstate(over="ignore"):
        return I(_dn(x.lo ** n, n), _up(x.hi ** n, n))


def powr(x: I, e: float) -> I:
    if e == 0.0:
        return I(np.ones(x.shape))
    if e < 0.0:
        return recip(powr(x, -e))
    with np.errstate(invalid="ignore", over="ignore"):
        lo = np.maximum(x.lo, 0.0) ** e
        hi = np.maximum(x.hi, 0.0) ** e
    return I(np.maximum(_dn(lo, 4), 0.0), _up(hi, 4))


def minimum(a, b) -> I:
    a, b = as_i(a), as_i(b)
    return I(np.minimum(a.lo, b.lo), np.minimum(a.hi, b.hi))


def maximum(a, b) -> I:
    a, b = as_i(a), as_i(b)
    return I(np.maximum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def hull(a, b) -> I:
    a, b = as_i(a), as_i(b)
    return I(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def clip(x: I, lo, hi) -> I:
    return I(np.clip(x.lo, lo, hi), np.clip(x.hi, lo, hi))


def total(x: I, axis: int) -> I:
    n = max(1, x.lo.shape[axis])
    with np.errstate(invalid="ignore"):
        return I(_dn(x.lo.sum(axis=axis), n), _up(x.hi.sum(axis=axis), n))


def arccos(x: I) -> I:
    lo = np.arccos(np.clip(x.hi, -1.0, 1.0))
    hi = np.arccos(np.clip(x.lo, -1.0, 1.0))
    return I(np.maximum(_dn(lo, 4), 0.0), np.minimum(_up(hi, 4), _up(np.pi)))


def atan2_upper(y: I, x: I) -> I:
    ylo, yhi = np.maximum(y.lo, 0.0), np.maximum(y.hi, 0.0)
    least = np.arctan2(np.where(x.hi > 0.0, ylo, yhi), x.hi)
    most = np.arctan2(np.where(x.lo > 0.0, yhi, ylo), x.lo)
    most = np.where((ylo == 0.0) & (x.lo <= 0.0), np.pi, most)
    least = np.where((ylo == 0.0) & (x.hi >= 0.0), 0.0, least)
    return I(np.maximum(_dn(least, 4), 0.0), np.minimum(_up(most, 4), _up(np.pi)))


def norm(v: I) -> I:
    x, y = v[..., 0], v[..., 1]
    with np.errstate(over="ignore"):
        lo = np.hypot(mig(x), mig(y))
        hi = np.hypot(mag(x), mag(y))
    return I(np.maximum(_dn(lo, 2), 0.0), _up(hi, 2))


def _unit_part(x: I, y: I) -> I:
    near, far = mig(y), mag(y)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        lo = np.where(x.lo > 0.0, x.lo / np.hypot(x.lo, far),
                      np.where(x.lo < 0.0, x.lo / np.hypot(x.lo, near), 0.0))
        hi = np.where(x.hi > 0.0, x.hi / np.hypot(x.hi, near),
                      np.where(x.hi < 0.0, x.hi / np.hypot(x.hi, far), 0.0))
    return I(np.clip(_dn(lo, 4), -1.0, 1.0), np.clip(_up(hi, 4), -1.0, 1.0))


def unit(v: I) -> I:
    x, y = v[..., 0], v[..., 1]
    return stack([_unit_part(x, y), _unit_part(y, x)], axis=-1)


def dot(a, b) -> I:
    a, b = as_i(a), as_i(b)
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def cross(a, b) -> I:
    a, b = as_i(a), as_i(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def perp(v: I) -> I:
    return stack([-v[..., 1], v[..., 0]], axis=-1)


class T:
    __slots__ = ("yes", "no")

    def __init__(self, yes, no):
        self.yes = np.asarray(yes, dtype=bool)
        self.no = np.asarray(no, dtype=bool)

    @classmethod
    def of(cls, value) -> "T":
        value = np.asarray(value, dtype=bool)
        return cls(value, ~value)

    @property
    def shape(self):
        return np.broadcast_shapes(self.yes.shape, self.no.shape)

    def __getitem__(self, index) -> "T":
        return T(self.yes[index], self.no[index])

    def __and__(self, other: "T") -> "T":
        return T(self.yes & other.yes, self.no | other.no)

    def __or__(self, other: "T") -> "T":
        return T(self.yes | other.yes, self.no & other.no)

    def __invert__(self) -> "T":
        return T(self.no, self.yes)

    def any(self, axis) -> "T":
        return T(np.any(self.yes, axis=axis), np.all(self.no, axis=axis))

    def all(self, axis) -> "T":
        return T(np.all(self.yes, axis=axis), np.any(self.no, axis=axis))

    def count(self, axis) -> I:
        return I(np.sum(self.yes, axis=axis), np.sum(~self.no, axis=axis))


def lt(a, b) -> T:
    a, b = as_i(a), as_i(b)
    return T(a.hi < b.lo, a.lo >= b.hi)


def le(a, b) -> T:
    a, b = as_i(a), as_i(b)
    return T(a.hi <= b.lo, a.lo > b.hi)


def gt(a, b) -> T:
    return lt(b, a)


def ge(a, b) -> T:
    return le(b, a)


def select(t: T, a, b) -> I:
    a, b = as_i(a), as_i(b)
    lo = np.where(t.yes, a.lo, np.where(t.no, b.lo, np.minimum(a.lo, b.lo)))
    hi = np.where(t.yes, a.hi, np.where(t.no, b.hi, np.maximum(a.hi, b.hi)))
    return I(lo, hi)


def select_t(t: T, a: T, b: T) -> T:
    return (t & a) | (~t & b)
