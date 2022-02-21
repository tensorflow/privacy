# Copyright 2021, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from tensorflow_privacy.privacy.privacy_tests.secret_sharer import generate_secrets as gs


class UtilsTest(absltest.TestCase):

  def __init__(self, methodname):
    """Initialize the test class."""
    super().__init__(methodname)

  def test_generate_random_sequences(self):
    """Test generate_random_sequences."""
    # Test when n is larger than total number of possible sequences.
    seqs = gs.generate_random_sequences(['A', 'b', 'c'], '{}+{}', 10, seed=27)
    expected_seqs = [
        'A+c', 'c+c', 'b+b', 'A+b', 'b+c', 'c+A', 'c+b', 'A+A', 'b+A'
    ]
    self.assertEqual(seqs, expected_seqs)

    # Test when n is smaller than total number of possible sequences.
    seqs = gs.generate_random_sequences(
        list('01234'), 'prefix {}{}{}?', 8, seed=9)
    expected_seqs = [
        'prefix 143?', 'prefix 031?', 'prefix 302?', 'prefix 042?',
        'prefix 404?', 'prefix 024?', 'prefix 021?', 'prefix 403?'
    ]
    self.assertEqual(seqs, expected_seqs)

  def test_construct_secret(self):
    secret_config = gs.SecretConfig(
        num_repetitions=[1, 2, 8],
        num_secrets_for_repetitions=[2, 3, 1],
        num_references=3,
        name='random secrets',
        properties=gs.TextSecretProperties(vocab=None, pattern=''))
    seqs = list('0123456789')
    secrets = gs.construct_secret(secret_config, seqs)
    self.assertEqual(secrets.config, secret_config)
    self.assertDictEqual(secrets.secrets, {
        1: ['0', '1'],
        2: ['2', '3', '4'],
        8: ['5']
    })
    self.assertEqual(secrets.references, ['7', '8', '9'])

    # Test when the number of elements in seqs is not enough.
    seqs = list('01234567')
    self.assertRaises(ValueError, gs.construct_secret, secret_config, seqs)

  def test_generate_secrets_and_references(self):
    secret_configs = [
        gs.SecretConfig(
            num_repetitions=[1, 12],
            num_secrets_for_repetitions=[2, 1],
            num_references=3,
            name='secret1',
            properties=gs.TextSecretProperties(
                vocab=['w1', 'w2', 'w3'], pattern='{} {} suf'),
        ),
        gs.SecretConfig(
            num_repetitions=[1, 2, 8],
            num_secrets_for_repetitions=[2, 3, 1],
            num_references=3,
            name='secert2',
            properties=gs.TextSecretProperties(
                vocab=['W 1', 'W 2', 'W 3'],
                pattern='{}-{}',
            ))
    ]
    secrets = gs.generate_text_secrets_and_references(secret_configs, seed=27)
    self.assertEqual(secrets[0].config, secret_configs[0])
    self.assertDictEqual(secrets[0].secrets, {
        1: ['w3 w2 suf', 'w2 w1 suf'],
        12: ['w1 w1 suf']
    })
    self.assertEqual(secrets[0].references,
                     ['w2 w3 suf', 'w2 w2 suf', 'w3 w1 suf'])

    self.assertEqual(secrets[1].config, secret_configs[1])
    self.assertDictEqual(
        secrets[1].secrets, {
            1: ['W 3-W 2', 'W 1-W 3'],
            2: ['W 3-W 1', 'W 2-W 1', 'W 1-W 1'],
            8: ['W 2-W 2']
        })
    self.assertEqual(secrets[1].references, ['W 2-W 3', 'W 3-W 3', 'W 1-W 2'])


if __name__ == '__main__':
  absltest.main()
