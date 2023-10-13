# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import scipy.linalg
import scipy.special
import scipy.stats

import jax
from jax import grad
from jax import lax
from jax import numpy as jnp
from jax import random
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu
from jax import vmap

config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
uint_dtypes = jtu.dtypes.all_unsigned


@jtu.with_config(jax_legacy_prng_key='allow')
class LaxRandomTest(jtu.JaxTestCase):

  def _CheckCollisions(self, samples, nbits):
    fail_prob = 0.01  # conservative bound on statistical fail prob by Chebyshev
    nitems = len(samples)
    nbins = 2 ** nbits
    nexpected = nbins * (1 - ((nbins - 1) / nbins) ** nitems)
    ncollisions = len(np.unique(samples))
    sq_percent_deviation = ((ncollisions - nexpected) / nexpected) ** 2
    self.assertLess(sq_percent_deviation, 1 / np.sqrt(nexpected * fail_prob))

  def _CheckKolmogorovSmirnovCDF(self, samples, cdf):
    # conservative bound on statistical fail prob by Kolmo CDF
    # bfloat16 quantization creates much lower p-values in large distributions
    fail_prob = 0.003 if samples.dtype == jnp.bfloat16 else 0.01
    # TODO(frostig): This reads enable_custom_prng as a proxy for
    # whether RBG keys may be involved, but that's no longer exact.
    if config.enable_custom_prng.value and samples.dtype == jnp.bfloat16:
      return
    self.assertGreater(scipy.stats.kstest(samples, cdf).pvalue, fail_prob)

  def _CheckChiSquared(self, samples, pmf):
    if samples.dtype == bool:
      samples = samples.astype(int)
    alpha = 0.01  # significance level, threshold for p-value

    # scipy.stats.chisquare requires the sum of expected and actual to
    # match; this is only the case if we compute the expected frequency
    # at *all* nonzero values of the pmf. We don't know this a priori,
    # so we add extra values past the largest observed value. The number
    # below is empirically enough to get full coverage for the current set
    # of tests. If a new test is added where this is not enough, chisquare()
    # below will error due to the sums of the inputs not matching.
    extra_values = 100
    actual_freq = np.bincount(samples, minlength=samples.max() + extra_values)
    values = np.arange(len(actual_freq))

    expected_freq = pmf(values) * samples.size

    valid = expected_freq > 0
    actual_freq = actual_freq[valid]
    expected_freq = expected_freq[valid]

    _, p_value = scipy.stats.chisquare(actual_freq, expected_freq)
    self.assertGreater(
        p_value, alpha,
        msg=f'Failed chi-squared test with p={p_value}.\n'
            'Expected vs. actual frequencies:\n'
            f'{expected_freq}\n{actual_freq}')

  def make_key(self, seed):
    return random.PRNGKey(seed, impl='threefry2x32')

  @jtu.sample_product(
    num=(None, 6, (6,), (2, 3), (2, 3, 4)),
  )
  def test_split_size_shape(self, num):
    key = self.make_key(0)
    if num is None:
      key_split = jax.random.split(key)
    else:
      key_split = jax.random.split(key, num)

    if num is None:
      self.assertEqual(key_split.shape, (2, *key.shape))
    elif type(num) is tuple:
      self.assertEqual(key_split.shape, (*num, *key.shape))
    else:
      self.assertEqual(key_split.shape, (num, *key.shape))

  @jtu.sample_product(dtype=jtu.dtypes.floating)
  def testNumpyAndXLAAgreeOnFloatEndianness(self, dtype):
    bits_dtype = np.uint32 if jnp.finfo(dtype).bits == 32 else np.uint64
    numpy_bits = np.array(1., dtype).view(bits_dtype)
    xla_bits = jax.jit(
        lambda: lax.bitcast_convert_type(np.array(1., dtype), bits_dtype))()
    self.assertEqual(numpy_bits, xla_bits)

  @jtu.sample_product(dtype=float_dtypes)
  def testRngUniform(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.uniform(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckCollisions(samples, jnp.finfo(dtype).nmant)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.uniform().cdf)

  @jtu.sample_product(dtype=int_dtypes + uint_dtypes)
  def testRngRandint(self, dtype):
    lo = 5
    hi = 10

    key = self.make_key(0)
    rand = lambda key: random.randint(key, (10000,), lo, hi, dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertTrue(np.all(lo <= samples))
      self.assertTrue(np.all(samples < hi))

  @jtu.sample_product(dtype=float_dtypes)
  def testNormal(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.norm().cdf)

  def testNormalBfloat16(self):
    # Passing bfloat16 as dtype string.
    # https://github.com/google/jax/issues/6813
    res_bfloat16_str = random.normal(self.make_key(0), dtype='bfloat16')
    res_bfloat16 = random.normal(self.make_key(0), dtype=jnp.bfloat16)
    self.assertAllClose(res_bfloat16, res_bfloat16_str)

  @jtu.sample_product(dtype=complex_dtypes)
  def testNormalComplex(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.normal(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(jnp.real(samples), scipy.stats.norm(scale=1/np.sqrt(2)).cdf)
      self._CheckKolmogorovSmirnovCDF(jnp.imag(samples), scipy.stats.norm(scale=1/np.sqrt(2)).cdf)
      self.assertEqual(dtype, samples.dtype)

  @jtu.sample_product(dtype=float_dtypes)
  def testTruncatedNormal(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.truncated_normal(key, -0.3, 0.3, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    min_val = np.min(uncompiled_samples)
    max_val = np.max(uncompiled_samples)
    self.assertTrue(min_val > -0.3)
    self.assertTrue(max_val < 0.3)
    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.truncnorm(-0.3, 0.3).cdf)

  @jtu.sample_product(dtype=jtu.dtypes.floating + jtu.dtypes.integer)
  def testShuffle(self, dtype):
    key = self.make_key(0)
    x = np.arange(100).astype(dtype)
    rand = lambda key: random.shuffle(key, x)
    crand = jax.jit(rand)

    with self.assertWarns(FutureWarning):
      perm1 = rand(key)
    with self.assertWarns(FutureWarning):
      perm2 = crand(key)

    self.assertAllClose(perm1, perm2)
    self.assertFalse(np.all(perm1 == x))  # seems unlikely!
    self.assertAllClose(np.sort(perm1), x, check_dtypes=False)

  @jtu.sample_product(
    [dict(shape=shape, replace=replace, axis=axis,
          input_range_or_shape=input_range_or_shape)
      for shape in [(), (5,), (4, 5)]
      for replace in [True, False]
      for input_range_or_shape in [100, (10, 10), (10, 5, 2), 1, (1, 5)]
      for is_range in [type(input_range_or_shape) is int]
      for ndim in [1 if is_range else len(input_range_or_shape)]
      for axis in range(-ndim, ndim or 1)
      for ninputs in [input_range_or_shape if is_range else input_range_or_shape[axis]]
      if replace or math.prod(shape) <= ninputs
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.integer,
    weighted=[True, False],
  )
  def testChoice(self, dtype, input_range_or_shape, shape, replace, weighted, axis):
    # This is the function API that we test against (note that self.rng().choice differs)
    np_choice = np.random.default_rng(0).choice
    p_dtype = dtypes.to_inexact_dtype(dtype)

    key = self.make_key(0)
    is_range = type(input_range_or_shape) is int
    x = (input_range_or_shape if is_range else
         self.rng().permutation(np.arange(math.prod(
           input_range_or_shape), dtype=dtype)).reshape(input_range_or_shape))
    N = x if is_range else x.shape[axis]
    if weighted:
      p = np.arange(N, dtype=p_dtype) + 1
      p /= p.sum()
    else:
      p = None
    rand = lambda key, x: random.choice(key, x, shape, replace, p, axis)
    sample = rand(key, x)
    if not is_range:
      self.assertEqual(dtype, sample.dtype)
    expected_shape = np.shape(np_choice(x, shape or None, replace, p, axis))
    self.assertEqual(expected_shape, sample.shape)
    expected_dtype = dtypes.result_type(int if is_range else x)
    self.assertEqual(expected_dtype, sample.dtype)
    if not replace and shape:
      def lsort(x):
        if not math.prod(x.shape): return x
        ind = np.lexsort(np.swapaxes(x, axis, -1).reshape((-1, x.shape[axis])))
        return jnp.take(x, ind, axis)
      self.assertArraysEqual(lsort(sample), lsort(np.unique(sample, axis=axis)))
    self.assertArraysEqual(sample, rand(key, np.array(x)))
    self.assertArraysEqual(sample, jax.jit(rand, static_argnames=
      'x' if is_range else None)(key, x))

  @jtu.sample_product(
    [dict(range_or_shape=range_or_shape, axis=axis)
      for range_or_shape in [0, 1, 100, (0,), (1,), (100,),
                             (10, 10), (10, 5, 2), (0, 5), (1, 5)]
      for ndim in [1 if type(range_or_shape) is int else len(range_or_shape)]
      for axis in range(-ndim, ndim or 1)
    ],
    dtype=jtu.dtypes.floating + jtu.dtypes.integer,
    independent=[True, False],
  )
  def testPermutation(self, dtype, range_or_shape, axis, independent):
    key = self.make_key(0)
    is_range = type(range_or_shape) is int
    x = (range_or_shape if is_range else
         self.rng().permutation(np.arange(
           math.prod(range_or_shape), dtype=dtype)).reshape(range_or_shape))
    shape = ((range_or_shape,) if is_range else range_or_shape)
    x_ = np.copy(x)
    rand = lambda key, x: random.permutation(key, x, axis, independent=independent)
    perm = rand(key, x)
    if shape[axis] >= 10:
      self.assertFalse(np.all(perm == x))  # seems unlikely!
    arr = np.arange(x) if is_range else x
    def lsort(x):
      if not math.prod(x.shape): return x
      ind = np.lexsort(np.swapaxes(x, axis, -1).reshape((-1, x.shape[axis])))
      return jnp.take(x, ind, axis)
    if not independent:
      self.assertArraysEqual(lsort(arr), lsort(perm), check_dtypes=not is_range)
    if independent and (arr.shape[axis] > 4) and (arr.size // arr.shape[axis] > 4):
      # Check for independent shuffling if there are >4 vectors of size >4.
      # Chance of false positive is 1 in (5!)^4
      with self.assertRaises(AssertionError):
        self.assertArraysEqual(lsort(arr), lsort(perm), check_dtypes=not is_range)
    self.assertArraysEqual(x_, x)
    self.assertArraysEqual(perm, rand(key, np.array(x)))
    self.assertArraysEqual(perm, jax.jit(rand, static_argnames=
      'x' if is_range else None)(key, x))

  def testPermutationErrors(self):
    key = self.make_key(0)
    with self.assertRaises(ValueError):
      random.permutation(key, 10, axis=3)
    with self.assertRaises(TypeError):
      random.permutation(key, 10.)
    with self.assertRaises(core.ConcretizationTypeError):
      jax.jit(random.permutation)(key, 10)

  @jtu.sample_product(
    p=[0.1, 0.5, 0.9],
    dtype=jtu.dtypes.floating,
  )
  def testBernoulli(self, p, dtype):
    key = self.make_key(0)
    p = np.array(p, dtype=dtype)
    rand = lambda key, p: random.bernoulli(key, p, (10000,))
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.bernoulli(p).pmf)

  @jtu.sample_product(
    [dict(p=p, axis=axis)
      for (p, axis) in [
        ([.25] * 4, -1),
        ([.1, .2, .3, .4], -1),
        ([[.5, .5], [.1, .9]], 1),
        ([[.5, .1], [.5, .9]], 0),
      ]
    ],
    sample_shape=[(10000,), (5000, 2)],
    dtype=jtu.dtypes.floating,
  )
  def testCategorical(self, p, axis, dtype, sample_shape):
    key = self.make_key(0)
    p = np.array(p, dtype=dtype)
    logits = np.log(p) - 42 # test unnormalized
    out_shape = tuple(np.delete(logits.shape, axis))
    shape = sample_shape + out_shape
    rand = partial(random.categorical, shape=shape, axis=axis)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, logits)
    compiled_samples = crand(key, logits)

    if axis < 0:
      axis += len(logits.shape)

    for samples in [uncompiled_samples, compiled_samples]:
      assert samples.shape == shape
      samples = jnp.reshape(samples, (10000,) + out_shape)
      if len(p.shape[:-1]) > 0:
        ps = np.transpose(p, (1, 0)) if axis == 0 else p
        for cat_samples, cat_p in zip(samples.transpose(), ps):
          pmf = lambda x: np.where(x < len(cat_p), cat_p[np.minimum(len(cat_p) - 1, x)], 0.0)
          self._CheckChiSquared(cat_samples, pmf=pmf)
      else:
        pmf = lambda x: np.where(x < len(p), p[np.minimum(len(p) - 1, x)], 0.0)
        self._CheckChiSquared(samples, pmf=pmf)

  def testBernoulliShape(self):
    key = self.make_key(0)
    with jax.numpy_rank_promotion('allow'):
      x = random.bernoulli(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @jtu.sample_product(
    a=[0.2, 5.],
    b=[0.2, 5.],
    dtype=[np.float64],  # NOTE: KS test fails with float32
  )
  def testBeta(self, a, b, dtype):
    if not config.enable_x64.value:
      raise SkipTest("skip test except on X64")
    key = self.make_key(0)
    rand = lambda key, a, b: random.beta(key, a, b, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, a, b)
    compiled_samples = crand(key, a, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.beta(a, b).cdf)

  @jtu.skip_on_devices("tpu")  # TPU precision causes issues.
  def testBetaSmallParameters(self, dtype=np.float32):
    # Regression test for beta version of https://github.com/google/jax/issues/9896
    key = self.make_key(0)
    a, b = 0.0001, 0.0002
    samples = random.beta(key, a, b, shape=(100,), dtype=dtype)

    # With such small parameters, all samples should be exactly zero or one.
    tol = 5E-2 if jtu.test_device_matches(["tpu"]) else 1E-3

    zeros = samples[samples < 0.5]
    self.assertAllClose(zeros, jnp.zeros_like(zeros), atol=tol)

    ones = samples[samples >= 0.5]
    self.assertAllClose(ones, jnp.ones_like(ones), atol=tol)

  @jtu.sample_product(dtype=float_dtypes)
  def testCauchy(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.cauchy(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.cauchy().cdf)

  @jtu.sample_product(
    alpha=[np.array([0.2, 1., 5.]),],
    dtype=jtu.dtypes.floating,
  )
  @jtu.skip_on_devices("tpu")  # TODO(mattjj): slow compilation times
  def testDirichlet(self, alpha, dtype):
    key = self.make_key(0)
    rand = lambda key, alpha: random.dirichlet(key, alpha, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, alpha)
    compiled_samples = crand(key, alpha)

    for samples in [uncompiled_samples, compiled_samples]:
      self.assertAllClose(samples.sum(-1), np.ones(10000, dtype=dtype))
      alpha_sum = sum(alpha)
      for i, a in enumerate(alpha):
        self._CheckKolmogorovSmirnovCDF(samples[..., i], scipy.stats.beta(a, alpha_sum - a).cdf)

  @jtu.skip_on_devices("tpu")  # lower accuracy leads to failures.
  def testDirichletSmallAlpha(self, dtype=np.float32):
    # Regression test for https://github.com/google/jax/issues/9896
    key = self.make_key(0)
    alpha = 0.00001 * jnp.ones(3)
    samples = random.dirichlet(key, alpha, shape=(100,), dtype=dtype)

    # Check that results lie on the simplex.
    self.assertAllClose(samples.sum(1), jnp.ones(samples.shape[0]),
                        check_dtypes=False, rtol=1E-5)

    # Check that results contain 1 in one of the dimensions:
    # this is highly likely to be true when alpha is small.
    self.assertAllClose(samples.max(1), jnp.ones(samples.shape[0]),
                        check_dtypes=False, rtol=1E-4)

  @jtu.sample_product(dtype=float_dtypes)
  def testExponential(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.exponential(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.expon().cdf)

  @jtu.sample_product(
    a=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  @jtu.skip_on_devices("tpu")  # low accuracy leads to failures.
  def testGammaVsLogGamma(self, a, dtype):
    # Test that gamma() and loggamma() produce equivalent samples.
    key = self.make_key(0)
    rand_gamma = lambda key, a: random.gamma(key, a, (100,), dtype)
    rand_loggamma = lambda key, a: random.loggamma(key, a, (100,), dtype)
    crand_loggamma = jax.jit(rand_loggamma)
    tol = {np.float32: 1E-6, np.float64: 1E-12}

    self.assertAllClose(rand_gamma(key, a), jnp.exp(rand_loggamma(key, a)),
                        atol=tol, rtol=tol)
    self.assertAllClose(rand_gamma(key, a), jnp.exp(crand_loggamma(key, a)),
                        atol=tol, rtol=tol)

  @jtu.sample_product(
    a=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  def testGamma(self, a, dtype):
    key = self.make_key(1)
    rand = lambda key, a: random.gamma(key, a, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, a)
    compiled_samples = crand(key, a)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gamma(a).cdf)

  def testGammaShape(self):
    key = self.make_key(0)
    x = random.gamma(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @jtu.sample_product(
    log_space=[True, False],
    alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],
  )
  def testGammaGrad(self, log_space, alpha):
    rng = self.make_key(0)
    alphas = np.full((100,), alpha)
    z = random.gamma(rng, alphas)
    if log_space:
      actual_grad = jax.grad(lambda x: lax.exp(random.loggamma(rng, x)).sum())(alphas)
    else:
      actual_grad = jax.grad(lambda x: random.gamma(rng, x).sum())(alphas)

    eps = 0.01 * alpha / (1.0 + np.sqrt(alpha))
    cdf_dot = (scipy.stats.gamma.cdf(z, alpha + eps)
               - scipy.stats.gamma.cdf(z, alpha - eps)) / (2 * eps)
    with np.errstate(over='ignore'):
      pdf = scipy.stats.gamma.pdf(z, alpha)
    expected_grad = -cdf_dot / pdf

    rtol = 2e-2 if jtu.test_device_matches(["tpu"]) else 7e-4
    self.assertAllClose(actual_grad, expected_grad, check_dtypes=True,
                        rtol=rtol)

  def testGammaGradType(self):
    # Regression test for https://github.com/google/jax/issues/2130
    key = self.make_key(0)
    a = jnp.array(1., dtype=jnp.float32)
    b = jnp.array(3., dtype=jnp.float32)
    f = lambda x, y: random.gamma(key=key, a=x, dtype=jnp.float32) / y
    # Should not crash with a type error.
    jax.vjp(f, a, b)

  @jtu.sample_product(
    lam=[0.5, 3, 9, 11, 50, 500],
    dtype=jtu.dtypes.supported([np.int16, np.int32, np.int64]),
  )
  def testPoisson(self, lam, dtype):
    key = self.make_key(0)
    rand = lambda key, lam: random.poisson(key, lam, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, lam)
    compiled_samples = crand(key, lam)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.poisson(lam).pmf)
      # TODO(shoyer): determine error bounds for moments more rigorously (e.g.,
      # based on the central limit theorem).
      self.assertAllClose(samples.mean(), lam, rtol=0.02, check_dtypes=False)
      self.assertAllClose(samples.var(), lam, rtol=0.03, check_dtypes=False)

  def testPoissonBatched(self):
    key = self.make_key(1)
    lam = jnp.concatenate([2 * jnp.ones(10000), 20 * jnp.ones(10000)])
    samples = random.poisson(key, lam, shape=(20000,))
    self._CheckChiSquared(samples[:10000], scipy.stats.poisson(2.0).pmf)
    self._CheckChiSquared(samples[10000:], scipy.stats.poisson(20.0).pmf)

  def testPoissonWithoutShape(self):
    key = self.make_key(1)
    lam = 2 * jnp.ones(10000)
    samples = random.poisson(key, lam)
    self._CheckChiSquared(samples, scipy.stats.poisson(2.0).pmf)

  def testPoissonShape(self):
    key = self.make_key(0)
    x = random.poisson(key, np.array([2.0, 20.0]), shape=(3, 2))
    assert x.shape == (3, 2)

  def testPoissonZeros(self):
    key = self.make_key(0)
    lam = jnp.concatenate([jnp.zeros(10), 20 * jnp.ones(10)])
    samples = random.poisson(key, lam, shape=(2, 20))
    self.assertArraysEqual(samples[:, :10], jnp.zeros_like(samples[:, :10]))

  def testPoissonCornerCases(self):
    key = self.make_key(0)
    lam = jnp.array([-1, 0, jnp.nan])
    samples = random.poisson(key, lam, shape=(3,))
    self.assertArraysEqual(samples, jnp.array([-1, 0, -1]), check_dtypes=False)

  @jtu.sample_product(dtype=jtu.dtypes.floating)
  def testGumbel(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.gumbel(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.gumbel_r().cdf)

  @jtu.sample_product(dtype=float_dtypes)
  def testLaplace(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.laplace(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.laplace().cdf)

  @jtu.sample_product(dtype=float_dtypes)
  def testLogistic(self, dtype):
    key = self.make_key(0)
    rand = lambda key: random.logistic(key, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.logistic().cdf)

  @jtu.sample_product(
    n=range(1, 5),
    shape=[(), (5,), (10, 5)],
    dtype=jtu.dtypes.floating + jtu.dtypes.complex,
  )
  @jax.default_matmul_precision("float32")
  def testOrthogonal(self, n, shape, dtype):
    key = self.make_key(0)
    q = random.orthogonal(key, n, shape, dtype)
    self.assertEqual(q.shape, (*shape, n, n))
    self.assertEqual(q.dtype, dtype)
    with jax.numpy_rank_promotion('allow'):
      self.assertAllClose(
        jnp.einsum('...ij,...jk->...ik', q, jnp.conj(q).swapaxes(-2, -1)),
        jnp.broadcast_to(jnp.eye(n, dtype=dtype), (*shape, n, n))
      )

  @jtu.sample_product(
    p=[.5, 1., 1.5, 2., 2.5],
    shape=[(), (5,), (10, 5)],
    dtype=jtu.dtypes.floating,
  )
  def testGeneralizedNormal(self, p, shape, dtype):
    key = self.make_key(0)
    rand = lambda key, p: random.generalized_normal(key, p, shape, dtype)
    crand = jax.jit(rand)
    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)
    for samples in [uncompiled_samples, compiled_samples]:
      self.assertEqual(samples.shape, shape)
      self.assertEqual(samples.dtype, dtype)
      self._CheckKolmogorovSmirnovCDF(samples.ravel(), scipy.stats.gennorm(p).cdf)

  @jtu.sample_product(
    d=range(1, 5),
    p=[.5, 1., 1.5, 2., 2.5],
    shape=[(), (5,), (10, 5)],
    dtype=jtu.dtypes.floating,
  )
  @jtu.skip_on_devices("tpu")  # TPU precision causes issues.
  def testBall(self, d, p, shape, dtype):
    key = self.make_key(123)
    rand = lambda key, p: random.ball(key, d, p, shape, dtype)
    crand = jax.jit(rand)
    uncompiled_samples = rand(key, p)
    compiled_samples = crand(key, p)
    for samples in [uncompiled_samples, compiled_samples]:
      self.assertEqual(samples.shape, (*shape, d))
      self.assertEqual(samples.dtype, dtype)
      self.assertTrue(((jnp.abs(samples) ** p).sum(-1) <= 1).all())
      norms = (jnp.abs(samples) ** p).sum(-1) ** (d / p)
      self._CheckKolmogorovSmirnovCDF(norms.ravel(), scipy.stats.uniform().cdf)

  @jtu.sample_product(
    b=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  def testPareto(self, b, dtype):
    key = self.make_key(0)
    rand = lambda key, b: random.pareto(key, b, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, b)
    compiled_samples = crand(key, b)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.pareto(b).cdf)

  def testParetoShape(self):
    key = self.make_key(0)
    with jax.numpy_rank_promotion('allow'):
      x = random.pareto(key, np.array([0.2, 0.3]), shape=(3, 2))
    assert x.shape == (3, 2)

  @jtu.sample_product(
    df=[0.1, 1., 10.],
    dtype=jtu.dtypes.floating,
  )
  @jtu.skip_on_devices("cpu", "tpu")  # TODO(phawkins): slow compilation times
  def testT(self, df, dtype):
    key = self.make_key(1)
    rand = lambda key, df: random.t(key, df, (10000,), dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, df)
    compiled_samples = crand(key, df)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.t(df).cdf)

  @jtu.sample_product(
    dim=[1, 3, 5],
    dtype=float_dtypes,
    method=['svd', 'eigh', 'cholesky'],
  )
  def testMultivariateNormal(self, dim, dtype, method):
    r = self.rng()
    mean = r.randn(dim)
    cov_factor = r.randn(dim, dim)
    cov = np.dot(cov_factor, cov_factor.T) + dim * np.eye(dim)

    key = self.make_key(0)
    rand = partial(random.multivariate_normal, mean=mean, cov=cov,
                   shape=(10000,), method=method)
    crand = jax.jit(rand)

    with jax.numpy_rank_promotion('allow'):
      uncompiled_samples = np.asarray(rand(key), np.float64)
      compiled_samples = np.asarray(crand(key), np.float64)

    inv_scale = scipy.linalg.lapack.dtrtri(np.linalg.cholesky(cov), lower=True)[0]
    for samples in [uncompiled_samples, compiled_samples]:
      centered = samples - mean
      whitened = np.einsum('nj,ij->ni', centered, inv_scale)

      # This is a quick-and-dirty multivariate normality check that tests that a
      # uniform mixture of the marginals along the covariance matrix's
      # eigenvectors follow a standard normal distribution.
      self._CheckKolmogorovSmirnovCDF(whitened.ravel(), scipy.stats.norm().cdf)

  @jtu.sample_product(
    dim=[1, 2, 4],
    mean_batch_size=[(), (3,), (2, 3)],
    cov_batch_size=[(), (3,), (2, 3)],
    shape=[(), (1,), (5,)],
    method=['cholesky', 'svd', 'eigh'],
  )
  def testMultivariateNormalShapes(self, dim, mean_batch_size, cov_batch_size,
                                   shape, method):
    r = self.rng()
    key = self.make_key(0)
    eff_batch_size = mean_batch_size \
      if len(mean_batch_size) > len(cov_batch_size) else cov_batch_size
    mean = r.randn(*(mean_batch_size + (dim,)))
    cov_factor = r.randn(*(cov_batch_size + (dim, dim)))
    cov = np.einsum('...ij,...kj->...ik', cov_factor, cov_factor)
    cov += 1e-3 * np.eye(dim)
    shape = shape + eff_batch_size
    with jax.numpy_rank_promotion('allow'):
      samples = random.multivariate_normal(key, mean, cov, shape=shape, method=method)
    assert samples.shape == shape + (dim,)

  def testMultivariateNormalCovariance(self):
    # test code based on https://github.com/google/jax/issues/1869
    N = 100000
    mean = jnp.zeros(4)
    cov = jnp.array([[  0.19,  0.00, -0.13,  0.00],
                     [  0.00,  0.29,  0.00, -0.23],
                     [ -0.13,  0.00,  0.39,  0.00],
                     [  0.00, -0.23,  0.00,  0.49]], dtype=mean.dtype)

    out_np = self.rng().multivariate_normal(mean, cov, N)

    key = self.make_key(0)
    with jax.numpy_rank_promotion('allow'):
      out_jnp = random.multivariate_normal(key, mean=mean, cov=cov, shape=(N,))

    var_np = out_np.var(axis=0)
    var_jnp = out_jnp.var(axis=0)
    self.assertAllClose(var_np, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

    var_np = np.cov(out_np, rowvar=False)
    var_jnp = np.cov(out_jnp, rowvar=False)
    self.assertAllClose(var_np, var_jnp, rtol=1e-2, atol=1e-2,
                        check_dtypes=False)

  @jtu.sample_product(method=['cholesky', 'eigh', 'svd'])
  @jtu.skip_on_devices('gpu', 'tpu')  # Some NaNs on accelerators.
  def testMultivariateNormalSingularCovariance(self, method):
    # Singular covariance matrix https://github.com/google/jax/discussions/13293
    mu = jnp.zeros((2,))
    sigma = jnp.ones((2, 2))
    key = self.make_key(0)
    result = random.multivariate_normal(key, mean=mu, cov=sigma, shape=(10,), method=method)
    self.assertAllClose(result[:, 0], result[:, 1], atol=1e-3, rtol=1e-3)

    # Cholesky fails for singular inputs.
    if method == 'cholesky':
      self.assertTrue(np.all(np.isnan(result)))
    else:
      self.assertFalse(np.any(np.isnan(result)))

  def testIssue222(self):
    x = random.randint(self.make_key(10003), (), 0, 0)
    assert x == 0

  def testFoldIn(self):
    key = self.make_key(0)
    keys = [random.key_data(random.fold_in(key, i)) for i in range(10)]
    assert np.unique(keys, axis=0).shape[0] == 10

  def testFoldInBig(self):
    key = self.make_key(0)
    seeds = [2 ** 32 - 2, 2 ** 32 - 1]
    keys = [random.key_data(random.fold_in(key, seed)) for seed in seeds]
    assert np.unique(keys, axis=0).shape[0] == 2

  def testStaticShapeErrors(self):
    if config.disable_jit.value:
      raise SkipTest("test only relevant when jit enabled")

    @jax.jit
    def feature_map(n, d, sigma=1.0, seed=123):
      key = self.make_key(seed)
      W = random.normal(key, (d, n)) / sigma
      w = random.normal(key, (d, )) / sigma
      b = 2 * jnp.pi * random.uniform(key, (d, ))

      phi = lambda x, t: jnp.sqrt(2.0 / d) * jnp.cos(jnp.matmul(W, x) + w*t + b)
      return phi

    self.assertRaisesRegex(TypeError, 'Shapes must be 1D.*',
                           lambda: feature_map(5, 3))

  def testIssue756(self):
    key = self.make_key(0)
    w = random.normal(key, ())
    self.assertEqual(w.dtype, dtypes.canonicalize_dtype(jnp.float_))

  def testIssue1789(self):
    def f(x):
      return random.gamma(self.make_key(0), x)

    grad(lambda x: jnp.sum(vmap(f)(x)))(jnp.ones(2))

  def testDtypeErrorMessage(self):
    with self.assertRaisesRegex(ValueError, r"dtype argument to.*"):
      random.normal(self.make_key(0), (), dtype=jnp.int32)

  def testRandomBroadcast(self):
    """Issue 4033"""
    # test for broadcast issue in https://github.com/google/jax/issues/4033
    key = self.make_key(0)
    shape = (10, 2)
    with jax.numpy_rank_promotion('allow'):
      x1 = random.uniform(key, shape, minval=jnp.zeros(2), maxval=jnp.ones(2))
      x2 = random.randint(key, shape, jnp.array([0, 1]), jnp.array([1, 2]))
    assert x1.shape == shape
    assert x2.shape == shape

  def testMaxwellSample(self):
    num_samples = 10**5
    rng = self.make_key(0)

    rand = lambda x: random.maxwell(x, (num_samples, ))
    crand = jax.jit(rand)

    loc = jtu.to_default_dtype(scipy.stats.maxwell.mean())
    std = jtu.to_default_dtype(scipy.stats.maxwell.std())

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(np.mean(samples), loc, atol=0., rtol=0.1)
      self.assertAllClose(np.std(samples), std, atol=0., rtol=0.1)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.maxwell().cdf)

  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testWeibullSample(self, concentration, scale):
    num_samples = 10**5
    rng = self.make_key(0)

    rand = lambda x: random.weibull_min(x, scale, concentration, (num_samples,))
    crand = jax.jit(rand)

    loc = jtu.to_default_dtype(scipy.stats.weibull_min.mean(c=concentration, scale=scale))
    std = jtu.to_default_dtype(scipy.stats.weibull_min.std(c=concentration, scale=scale))

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(np.mean(samples), loc, atol=0., rtol=0.1)
      self.assertAllClose(np.std(samples), std, atol=0., rtol=0.1)
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.weibull_min(
          c=concentration, scale=scale).cdf)

  @parameterized.named_parameters(
      ('test1', 4.0, 1.0),
      ('test2', 2.0, 3.0))
  def testDoublesidedMaxwellSample(self, loc, scale):
    num_samples = 10**4
    rng = self.make_key(0)

    rand = lambda key: random.double_sided_maxwell(
        rng, loc, scale, (num_samples,))
    crand = jax.jit(rand)

    mean = loc
    std = np.sqrt(3.) * scale

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    # Compute the double sided maxwell CDF through the one sided maxwell cdf.
    # This is done as follows:
    # P(DSM <= x) = P (loc + scale * radamacher_sample * one_sided_sample <=x) =
    # P (radamacher_sample * one_sided_sample <= (x - loc) / scale) =
    # 1/2 P(one_sided_sample <= (x - loc) / scale)
    #    + 1/2 P( - one_sided_sample <= (x - loc) / scale) =
    #  1/2 P(one_sided_sample <= (x - loc) / scale)
    #    + 1/2 P(one_sided_sample >= - (x - loc) / scale) =
    # 1/2 CDF_one_maxwell((x - loc) / scale))
    #   + 1/2 (1 - CDF_one_maxwell(- (x - loc) / scale)))
    def double_sided_maxwell_cdf(x, loc, scale):
      pos = scipy.stats.maxwell().cdf((x - loc) / scale)
      neg = (1 - scipy.stats.maxwell().cdf((-x + loc) / scale))
      return (pos + neg) / 2

    for samples in [uncompiled_samples, compiled_samples]:
      # Check first and second moments.
      self.assertEqual((num_samples,), samples.shape)
      self.assertAllClose(samples.mean(), jtu.to_default_dtype(mean), atol=0., rtol=0.1)
      self.assertAllClose(samples.std(), jtu.to_default_dtype(std), atol=0., rtol=0.1)

      self._CheckKolmogorovSmirnovCDF(
          samples, lambda x: double_sided_maxwell_cdf(x, loc, scale))

  def testRadamacher(self):
    rng = self.make_key(0)
    num_samples = 10**5

    rand = lambda x: random.rademacher(x, (num_samples,))
    crand = jax.jit(rand)

    uncompiled_samples = rand(rng)
    compiled_samples = crand(rng)

    for samples in [uncompiled_samples, compiled_samples]:
      unique_values, counts = np.unique(samples, return_counts=True)
      assert len(unique_values) == 2
      assert len(counts) == 2

      self.assertAllClose(
          counts[0] / num_samples, 0.5, rtol=1e-02, atol=1e-02)
      self.assertAllClose(
          counts[1] / num_samples, 0.5, rtol=1e-02, atol=1e-02)

  def testChoiceShapeIsNotSequenceError(self):
    key = self.make_key(0)
    with self.assertRaises(TypeError):
      random.choice(key, 5, 2, replace=False)
    with self.assertRaises(TypeError):
      random.choice(key, 5, 2, replace=True)

  def test_eval_shape_big_random_array(self):
    def f(x):
      return random.normal(self.make_key(x), (int(1e12),))
    with jax.enable_checks(False):  # check_jaxpr will materialize array
      jax.eval_shape(f, 0)  # doesn't error

  @jtu.sample_product(
    type_=["int", "np.array", "jnp.array"],
    seed=[-1, 0, 1, (1 << 32) - 1, (1 << 63) - 1, np.uint64((1 << 64) - 1)],
  )
  def test_prng_jit_invariance(self, seed, type_):
    if type_ == "int" and seed == (1 << 64) - 1:
      self.skipTest("Expected failure: Python int too large.")
    if not config.enable_x64.value and seed > np.iinfo(np.int32).max:
      self.skipTest("Expected failure: Python int too large.")
    type_ = {"int": int, "np.array": np.array, "jnp.array": jnp.array}[type_]
    args_maker = lambda: [type_(seed)]
    f = lambda s: random.key_data(self.make_key(s))
    self._CompileAndCheck(f, args_maker)

  def test_prng_errors(self):
    seed = np.iinfo(np.int64).max + 1
    with self.assertRaises(OverflowError):
      self.make_key(seed)
    with self.assertRaises(OverflowError):
      jax.jit(self.make_key)(seed)

  def test_random_split_doesnt_device_put_during_tracing(self):
    key = self.make_key(1).block_until_ready()
    with jtu.count_device_put() as count:
      jax.jit(random.split)(key)
    self.assertLessEqual(count[0], 1)  # 1 for the argument device_put

  @jtu.sample_product(dtype=int_dtypes + uint_dtypes)
  def test_randint_bounds(self, dtype):
    min = np.iinfo(dtype).min
    max = np.iinfo(dtype).max
    key = self.make_key(1701)
    shape = (10,)
    if np.iinfo(dtype).bits < np.iinfo(dtypes.canonicalize_dtype(int)).bits:
      expected = random.randint(key, shape, min, max + 1, dtype)
      self.assertArraysEqual(expected, random.randint(key, shape, min - 12345, max + 12345, dtype))
    else:
      self.assertRaises(OverflowError, random.randint, key, shape, min - 12345, max + 12345, dtype)

  def test_randint_out_of_range(self):
    key = self.make_key(0)

    r = random.randint(key, (10,), 255, 256, np.uint8)
    self.assertAllClose(r, jnp.full_like(r, 255))

    r = random.randint(key, (1000,), -128, 128, np.int8)
    self.assertGreater((r == -128).sum(), 0)
    self.assertGreater((r == 127).sum(), 0)

    r = random.randint(key, (1000,), -1000, 1000, np.uint8)
    self.assertGreater((r == 0).sum(), 0)
    self.assertGreater((r == 255).sum(), 0)

  def test_large_prng(self):
    # https://github.com/google/jax/issues/11010
    def f():
      return random.uniform(
          self.make_key(3), (308000000, 128), dtype=jnp.bfloat16)

    # just lower, don't run, takes too long
    jax.jit(f).lower()

  @jtu.sample_product(shape=[(3, 4)],
                      logits_shape_base=[(3, 4), (3, 1), (1, 4)],
                      axis=[-3, -2, -1, 0, 1, 2])
  def test_categorical_shape_argument(self, shape, logits_shape_base, axis):
    # https://github.com/google/jax/issues/13124
    logits_shape = list(logits_shape_base)
    logits_shape.insert(axis % (len(logits_shape_base) + 1), 10)
    assert logits_shape[axis] == 10
    logits = jnp.ones(logits_shape)
    samples = random.categorical(self.make_key(0), logits=logits,
                                 axis=axis, shape=shape)
    self.assertEqual(samples.shape, shape)

  @jtu.sample_product(
      df = [0.2, 1., 10., 100.],
      dtype=jtu.dtypes.floating)
  def testChisquare(self, df, dtype):
    key = self.make_key(1)

    def rand(key, df):
      return random.chisquare(key, df, shape=(10000,), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key, df)
    compiled_samples = crand(key, df)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.chi2(df).cdf)

  @jtu.sample_product(
      dfnum = [1., 2., 10. ,100.],
      dfden = [1. ,2., 10., 100.],
      dtype=jtu.dtypes.floating)
  def testF(self, dfnum, dfden, dtype):
    key = self.make_key(1)
    rand = lambda key: random.f(key, dfnum, dfden, shape = (10000, ), dtype = dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.f(dfnum, dfden).cdf)

  @jtu.sample_product(
      scale= [0.2, 1., 2., 10. ,100.],
      dtype=jtu.dtypes.floating)
  def testRayleigh(self, scale, dtype):
    key = self.make_key(0)
    rand = lambda key: random.rayleigh(key, scale, shape = (10000, ), dtype = dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.rayleigh(scale=scale).cdf)

  @jtu.sample_product(
      mean= [0.2, 1., 2., 10. ,100.],
      dtype=jtu.dtypes.floating)
  def testWald(self, mean, dtype):
    key = self.make_key(0)
    rand = lambda key: random.wald(key, mean, shape=(10000, ), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.invgauss(mu=mean).cdf)

  @jtu.sample_product(
      p=[0.2, 0.3, 0.4, 0.5 ,0.6],
      dtype=jtu.dtypes.supported([np.int16, np.int32, np.int64]))
  def testGeometric(self, p, dtype):
    key = self.make_key(1)
    rand = lambda key: random.geometric(key, p, shape=(10000, ), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckChiSquared(samples, scipy.stats.geom(p).pmf)
      self.assertAllClose(samples.mean(), 1 / p, rtol=0.02, check_dtypes=False)
      self.assertAllClose(samples.var(), (1 - p) / (p * p) , rtol=0.05, check_dtypes=False)

  @jtu.sample_product(
      left = [0.2, 0.5, 1., 2.],
      mode = [3., 5., 8., 9.],
      right= [10., 20., 30., 40.],
      dtype= jtu.dtypes.floating)
  def testTriangular(self, left, mode, right, dtype):
    key = self.make_key(1)
    rand = lambda key: random.triangular(key, left, mode, right, shape=(10000, ), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.triang((mode - left) / (right - left), loc=left, scale=right - left).cdf)

  @jtu.sample_product(
    sigma = [0.2, 0.5, 1., 2.],
    dtype=jtu.dtypes.floating)
  def testLogNormal(self, sigma, dtype):
    key = self.make_key(0)
    rand = lambda key: random.lognormal(key, sigma, shape=(10000, ), dtype=dtype)
    crand = jax.jit(rand)

    uncompiled_samples = rand(key)
    compiled_samples = crand(key)

    for samples in [uncompiled_samples, compiled_samples]:
      self._CheckKolmogorovSmirnovCDF(samples, scipy.stats.lognorm(s=sigma).cdf)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
