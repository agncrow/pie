# Statistics: `use stats`

Pie's `stats` is a package of formulas over crow's math builtins: what
describes a series, what relates two, and the classical tests. A program
imports it beside `math` and calls it qualified, as it calls any package:

```crow
use math, stats

xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
log[stats.median[xs]]
log[stats.var[xs]]
log[stats.std[xs]]
log[stats.percentile[xs, 50.0]]
log[stats.percentile[xs, 90.0]]
log[stats.zscore[[1.0, 2.0, 3.0]]]
```

```text
4.5
4
2
4.5
7.6
[-1.224744871391589, 0, 1.224744871391589]
```

`var` and `std` are the population's (they divide by n, as numpy's do by
default), and a percentile is interpolated between the two values it falls
between. A statistic of no values is NaN; a percentile outside 0 to 100
stops with the reason.

## Two series

`cov` and `corr` relate two series of one length (the covariance is the
population's too), and `linreg[x, y]` fits the line y = slope · x +
intercept through them by least squares. It answers a map: the slope, the
intercept and the r² of the fit.

```crow
use math, stats

xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
log[fmt[stats.cov[xs, ys], decimals = 3]]
log[fmt[stats.corr[xs, ys], decimals = 3]]
line = stats.linreg[ys, xs]
log[fmt[line['slope'], decimals = 3]]
log[fmt[line['intercept'], decimals = 3]]
log[fmt[line['r2'], decimals = 3]]
```

```text
4.250
0.927
0.810
1.357
0.860
```

## Tests

Each test answers its statistic and its p-value in a map. `ttest` is
Welch's two-sample t-test, `chisq` the chi-square goodness of fit of
observed counts against expected ones, `anova` the one-way analysis of
variance of a list of groups, and `kstest` the Kolmogorov–Smirnov test of
the data against `'normal'` (with the data's own mean and spread) or
`'uniform'` (from 0 to 1). The t, F and chi-square distributions are read
through crow's validated special functions (`betainc`, `gammainc`).

```crow
use math, stats

xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
t = stats.ttest[xs, ys]
log['t `fmt[t[\'t\'], decimals = 4]`, p `fmt[t[\'p\'], decimals = 4]`']
c = stats.chisq[[10.0, 20.0, 30.0], [20.0, 20.0, 20.0]]
log['chi2 `c[\'chi2\']`, p `fmt[c[\'p\'], decimals = 4]`']
a = stats.anova[[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]
log['f `fmt[a[\'f\'], decimals = 4]`, p `fmt[a[\'p\'], decimals = 4]`']
k = stats.kstest[xs, 'normal']
log['d `fmt[k[\'d\'], decimals = 4]`, p `fmt[k[\'p\'], decimals = 4]`']
```

```text
t 0.4350, p 0.6703
chi2 10, p 0.0067
f 13.0000, p 0.0066
d 0.2500, p 0.6325
```

The t, chi-square and F answers are scipy's (`ttest_ind` with
`equal_var=False`, `chisquare`, `f_oneway`). The Kolmogorov–Smirnov p is
Kolmogorov's distribution at the distance scaled as Stephens scales it,
(√n + 0.12 + 0.11/√n) · d, as Numerical Recipes' `probks` computes it:
nearer the exact p for a small sample than scipy's `method='asymp'`, which
scales by √n alone (0.6325 here, against the exact 0.6134 and scipy's
0.6994), and the same as the sample grows. Like scipy's, it is the p of a
distribution named in advance: with the mean and the spread read from the
data, the true p is smaller (Lilliefors' test).

## Fitting a distribution

`fit` answers a distribution's parameters fitted to the data: `'normal'`
(`mu` and `sigma`), `'exponential'` (`lambda`), `'uniform'` (`a` and `b`)
or `'poisson'` (`lambda`).

```crow
use math, stats

xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
fitted = stats.fit[xs, 'normal']
log['mu `fitted[\'mu\']`, sigma `fitted[\'sigma\']`']
log[stats.fit[xs, 'uniform']]
```

```text
mu 5, sigma 2
[a = 2, b = 9]
```

The package's tests are its own (`packages/stats/main.co`):
`crow test --packages stats` from pie's root runs them in both modes.
