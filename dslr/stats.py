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
    total = 0.0
    for x in xs:
        total += (x - m) ** 2
    return total / (count(xs) - 1)


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


def sort(xs):
    """Merge sort returning a new list."""
    n = count(xs)
    if n <= 1:
        return list(xs)
    left, right = sort(xs[:n // 2]), sort(xs[n // 2:])
    nl, nr = count(left), count(right)
    out, i, j = [], 0, 0
    while i < nl and j < nr:
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    return out + left[i:] + right[j:]


def percentile(xs, p):
    """Linear interpolation between closest ranks (same as numpy/pandas default)."""
    s = sort(xs)
    n = count(s)
    k = (n - 1) * p / 100
    lo = int(k)
    hi = lo + 1 if lo + 1 < n else lo
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
