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

from unittest import SkipTest

from absl.testing import absltest

import numpy as np

import jax
from jax import numpy as jnp
from jax import random
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax import vmap

from jax._src import prng as prng_internal

config.parse_flags_with_absl()

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
uint_dtypes = jtu.dtypes.all_unsigned


threefry_seed = prng_internal.threefry_seed
threefry_split = prng_internal.threefry_split
threefry_random_bits = prng_internal.threefry_random_bits
threefry_fold_in = prng_internal.threefry_fold_in

def _double_threefry_seed(seed):
  int_t = seed.dtype.type if hasattr(seed, 'dtype') else type(seed)
  s1, s2 = seed, seed ^ int_t(3)
  return jnp.vstack([threefry_seed(s1),
                     threefry_seed(s2)])

def _double_threefry_split(key, shape):
  return vmap(
      threefry_split, (0, None), len(shape))(key, shape)

def _double_threefry_random_bits(key, bit_width, shape):
  bits0 = threefry_random_bits(key[0], bit_width, shape)
  bits1 = threefry_random_bits(key[1], bit_width, shape)
  del bits1
  # TODO(frostig): Currently this behaves like normal threefry, to
  # avoid a few probabilistic test failures. Ideally we might want to
  # test different generation behavior here (e.g. `bits0 ^ bits1`).
  return bits0

def _double_threefry_fold_in(key, data):
  return jnp.vstack([threefry_fold_in(key[0], data),
                     threefry_fold_in(key[1], data)])

double_threefry_prng_impl = prng_internal.PRNGImpl(
    key_shape=(2, 2),
    seed=_double_threefry_seed,
    split=_double_threefry_split,
    random_bits=_double_threefry_random_bits,
    fold_in=_double_threefry_fold_in,
    tag='fry2')


@jtu.with_config(jax_default_prng_impl='threefry2x32', jax_legacy_prng_key='allow')
class LaxRandomWithCustomPRNGTest(jtu.JaxTestCase):
  def make_key(self, seed):
    return prng_internal.seed_with_impl(double_threefry_prng_impl, seed)

  def test_split_shape(self):
    key = self.make_key(73)
    keys = random.split(key, 10)
    self.assertEqual(keys.shape, (10,))

  def test_vmap_fold_in_shape(self):
    # broadcast with scalar
    keys = random.split(self.make_key(73), 2)
    msgs = jnp.arange(3)
    out = vmap(lambda i: random.fold_in(keys[0], i))(msgs)
    self.assertEqual(out.shape, (3,))
    out = vmap(lambda k: random.fold_in(k, msgs[0]))(keys)
    self.assertEqual(out.shape, (2,))
    out = vmap(random.fold_in, in_axes=(None, 0))(keys[0], msgs)
    self.assertEqual(out.shape, (3,))
    out = vmap(random.fold_in, in_axes=(0, None))(keys, msgs[0])
    self.assertEqual(out.shape, (2,))

    # vmap all
    msgs = jnp.arange(2)
    out = vmap(random.fold_in)(keys, msgs)
    self.assertEqual(out.shape, (2,))

    # nested vmap
    keys = random.split(self.make_key(73), 2 * 3).reshape((2, 3))
    msgs = jnp.arange(2 * 3).reshape((2, 3))
    out = vmap(vmap(random.fold_in), in_axes=(0, 1))(keys, msgs.T)
    self.assertEqual(out.shape, (2, 3))
    out = vmap(vmap(random.fold_in), in_axes=(1, 0))(keys, msgs.T)
    self.assertEqual(out.shape, (3, 2))

  def test_vmap_split_mapped_key(self):
    key = self.make_key(73)
    mapped_keys = random.split(key, num=3)
    forloop_keys = [random.split(k) for k in mapped_keys]
    vmapped_keys = vmap(random.split)(mapped_keys)
    self.assertEqual(vmapped_keys.shape, (3, 2))
    for fk, vk in zip(forloop_keys, vmapped_keys):
      self.assertArraysEqual(random.key_data(fk),
                             random.key_data(vk))

  def test_cannot_add(self):
    key = self.make_key(73)
    self.assertRaisesRegex(
        ValueError, r'dtype=key<.*> is not a valid dtype for JAX type promotion.',
        lambda: key + 47)

  def test_grad_of_prng_key(self):
    key = self.make_key(73)
    with self.assertRaisesRegex(TypeError, 'grad requires real- or complex-valued inputs'):
      jax.grad(lambda x: 1.)(key)
    out = jax.grad(lambda x: 1., allow_int=True)(key)
    self.assertArraysEqual(out, np.zeros(key.shape, jax.dtypes.float0))


@jtu.with_config(jax_default_prng_impl='rbg', jax_legacy_prng_key='allow')
class LaxRandomWithRBGPRNGTest(jtu.JaxTestCase):
  def make_key(self, seed):
    return random.PRNGKey(seed, impl='rbg')

  def test_split_shape(self):
    key = self.make_key(73)
    keys = random.split(key, 10)
    self.assertEqual(keys.shape, (10, *key.shape))

  def test_vmap_fold_in_shape(self):
    # broadcast with scalar
    keys = random.split(self.make_key(73), 2)
    msgs = jnp.arange(3)

    out = vmap(lambda i: random.fold_in(keys[0], i))(msgs)
    self.assertEqual(out.shape, (3, *keys[0].shape))
    out = vmap(random.fold_in, in_axes=(None, 0))(keys[0], msgs)
    self.assertEqual(out.shape, (3, *keys[0].shape))

    out = vmap(lambda k: random.fold_in(k, msgs[0]))(keys)
    self.assertEqual(out.shape, keys.shape)
    out = vmap(random.fold_in, in_axes=(0, None))(keys, msgs[0])
    self.assertEqual(out.shape, keys.shape)

  def test_vmap_split_not_mapped_key(self):
    key = self.make_key(73)
    single_split_key = random.split(key)
    vmapped_keys = vmap(lambda _: random.split(key))(jnp.zeros(3,))
    self.assertEqual(vmapped_keys.shape, (3, 2, *key.shape))
    for vk in vmapped_keys:
      self.assertArraysEqual(random.key_data(vk),
                             random.key_data(single_split_key))

  def test_vmap_split_mapped_key(self):
    key = self.make_key(73)
    mapped_keys = random.split(key, num=3)
    forloop_keys = [random.split(k) for k in mapped_keys]
    vmapped_keys = vmap(random.split)(mapped_keys)
    self.assertEqual(vmapped_keys.shape, (3, 2, *key.shape))
    for fk, vk in zip(forloop_keys, vmapped_keys):
      self.assertArraysEqual(random.key_data(fk),
                             random.key_data(vk))

  def test_vmap_random_bits(self):
    rand_fun = lambda key: random.randint(key, (), 0, 100)
    key = self.make_key(73)
    mapped_keys = random.split(key, num=3)
    forloop_rand_nums = [rand_fun(k) for k in mapped_keys]
    rand_nums = vmap(rand_fun)(mapped_keys)
    self.assertEqual(rand_nums.shape, (3,))
    self.assertArraysEqual(rand_nums, jnp.array(forloop_rand_nums))

  def test_cannot_add(self):
    key = self.make_key(73)
    if not jnp.issubdtype(key.dtype, dtypes.prng_key):
      raise SkipTest('relies on typed key arrays')
    self.assertRaisesRegex(
        ValueError, r'dtype=key<.*> is not a valid dtype for JAX type promotion.',
        lambda: key + 47)

  def test_grad_of_prng_key(self):
    key = self.make_key(73)
    with self.assertRaisesRegex(TypeError, 'grad requires real- or complex-valued inputs'):
      jax.grad(lambda x: 1.)(key)
    out = jax.grad(lambda x: 1., allow_int=True)(key)
    self.assertArraysEqual(out, np.zeros(key.shape, jax.dtypes.float0))

  def test_random_split_doesnt_device_put_during_tracing(self):
    return  # this test doesn't apply to the RBG PRNG

  def test_randint_out_of_range(self):
    # TODO(mattjj): enable this test if/when RngBitGenerator supports it
    raise SkipTest('8-bit types not supported with RBG PRNG')


@jtu.with_config(jax_default_prng_impl='unsafe_rbg')
class LaxRandomWithUnsafeRBGPRNGTest(LaxRandomWithRBGPRNGTest):
  def make_key(self, seed):
    return random.PRNGKey(seed, impl="unsafe_rbg")


def _sampler_unimplemented_with_custom_prng(*args, **kwargs):
  raise SkipTest('sampler only implemented for default RNG')

for test_prefix in [
    'testPoisson',
    'testPoissonBatched',
    'testPoissonShape',
    'testPoissonZeros',
]:
  for attr in dir(LaxRandomWithCustomPRNGTest):
    if attr.startswith(test_prefix):
      setattr(LaxRandomWithCustomPRNGTest, attr,
              _sampler_unimplemented_with_custom_prng)
      setattr(LaxRandomWithRBGPRNGTest, attr,
              _sampler_unimplemented_with_custom_prng)
      setattr(LaxRandomWithUnsafeRBGPRNGTest, attr,
              _sampler_unimplemented_with_custom_prng)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
