def count(xs):
    n = 0
    for _ in xs:
        n += 1
    return n


def mean(xs):
    total = 0.0
    for x in xs:
        total += x
    return total / count(xs)


def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (count(xs) - 1)


def std(xs):
    return var(xs) ** 0.5


def minimum(xs):
    lo = xs[0]
    for x in xs:
        if x < lo:
            lo = x
    return lo


def maximum(xs):
    hi = xs[0]
    for x in xs:
        if x > hi:
            hi = x
    return hi


def percentile(xs, p):
    """Linear interpolation between closest ranks (same as numpy/pandas default)."""
    s = sorted(xs)
    k = (count(s) - 1) * p / 100
    lo = int(k)
    hi = lo + 1 if lo + 1 < count(s) else lo
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def skew(xs):
    m, s = mean(xs), std(xs)
    return mean([((x - m) / s) ** 3 for x in xs])


STATS = [
    ("Count", count),
    ("Mean", mean),
    ("Std", std),
    ("Min", minimum),
    ("25%", lambda xs: percentile(xs, 25)),
    ("50%", lambda xs: percentile(xs, 50)),
    ("75%", lambda xs: percentile(xs, 75)),
    ("Max", maximum),
    # bonus fields
    ("Var", var),
    ("Range", lambda xs: maximum(xs) - minimum(xs)),
    ("IQR", lambda xs: percentile(xs, 75) - percentile(xs, 25)),
    ("Skew", skew),
]


def describe(cols):
    """{feature: {stat: value}}; NaN for stats undefined on too few values."""
    table = {}
    for name, values in cols.items():
        xs = [v for v in values if v is not None]
        table[name] = {"Missing": float(count(values) - count(xs))}
        for stat, fn in STATS:
            try:
                table[name][stat] = float(fn(xs))
            except (ZeroDivisionError, IndexError):
                table[name][stat] = float("nan")
    return table
