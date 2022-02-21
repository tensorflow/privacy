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
"""Generate random sequences."""

import dataclasses
import itertools
import string
from typing import Any, Dict, MutableSequence, Sequence
import numpy as np


def generate_random_sequences(vocab: Sequence[str],
                              pattern: str,
                              n: int,
                              seed: int = 1) -> MutableSequence[str]:
  """Generates random sequences.

  Args:
    vocab: the vocabulary for the sequences
    pattern: the pattern of the sequence. The length of the sequence will be
      inferred from the pattern.
    n: number of sequences to generate
    seed: random seed for numpy.random

  Returns:
    A sequence of different random sequences from the given vocabulary
  """

  def count_placeholder(pattern):
    return sum([x[1] is not None for x in string.Formatter().parse(pattern)])

  length = count_placeholder(pattern)
  rng = np.random.RandomState(seed)
  vocab_size = len(vocab)
  if vocab_size**length <= n:
    # Generate all possible sequences of the length
    seq = np.array(list(itertools.product(vocab, repeat=length)))
    if vocab_size**length < n:
      print(f'The total number of random sequences is less than n={n}.',
            f'Will return {vocab_size**length} sequences only.')
      n = vocab_size**length
  else:
    # Generate idx where each row contains the indices for one random sequence
    idx = np.empty((0, length), dtype=int)
    while idx.shape[0] < n:
      # Generate a new set of indices
      idx_new = rng.randint(vocab_size, size=(n, length))
      idx = np.concatenate([idx, idx_new], axis=0)  # Add to existing indices
      idx = np.unique(idx, axis=0)  # Remove duplicated indices
    idx = idx[:n]
    seq = np.array(vocab)[idx]
  # Join each row to get the sequence
  seq = np.apply_along_axis(lambda x: pattern.format(*list(x)), 1, seq)
  seq = seq[rng.permutation(n)]
  return list(seq)


@dataclasses.dataclass
class TextSecretProperties:
  """Properties of text secret.

  vocab: the vocabulary for the secrets
  pattern: the pattern of the secrets
  """
  vocab: Sequence[str]
  pattern: str


@dataclasses.dataclass
class SecretConfig:
  """Configuration of secret for secrets sharer.

  num_repetitions: numbers of repetitions for the secrets
  num_secrets_for_repetitions: numbers of secrets to be used for each
    number of repetitions
  num_references: number of references sequences, i.e. random sequences that
    will not be inserted into training data
  name: name that identifies the secrets set
  properties: any properties of the secret, e.g. the vocabulary, the pattern
  """
  num_repetitions: Sequence[int]
  num_secrets_for_repetitions: Sequence[int]
  num_references: int
  name: str = ''
  properties: Any = None


@dataclasses.dataclass
class SecretsSet:
  """A secrets set for secrets sharer.

  config: configuration of the secrets
  secrets: a dictionary, key is the number of repetitions, value is a sequence
    of different secrets
  references: a sequence of references
  """
  config: SecretConfig
  secrets: Dict[int, Sequence[Any]]
  references: Sequence[Any]


def construct_secret(secret_config: SecretConfig,
                     seqs: Sequence[Any]) -> SecretsSet:
  """Constructs a SecretsSet instance given a sequence of samples.

  Args:
    secret_config: configuration of secret.
    seqs: a sequence of samples that will be used for secrets and references.

  Returns:
    a SecretsSet instance.
  """
  if len(seqs) < sum(
      secret_config.num_secrets_for_repetitions) + secret_config.num_references:
    raise ValueError('seqs does not contain enough elements.')
  secrets = {}
  i = 0
  for num_repetition, num_secrets in zip(
      secret_config.num_repetitions, secret_config.num_secrets_for_repetitions):
    secrets[num_repetition] = seqs[i:i + num_secrets]
    i += num_secrets
  return SecretsSet(
      config=secret_config,
      secrets=secrets,
      references=seqs[-secret_config.num_references:])


def generate_text_secrets_and_references(
    secret_configs: Sequence[SecretConfig],
    seed: int = 0) -> MutableSequence[SecretsSet]:
  """Generates a sequence of text secret sets given a sequence of configurations.

  Args:
    secret_configs: a sequence of text secret configurations.
    seed: random seed.

  Returns:
    A sequence of SecretsSet instances.
  """
  secrets_sets = []
  for i, secret_config in enumerate(secret_configs):
    n = secret_config.num_references + sum(
        secret_config.num_secrets_for_repetitions)
    seqs = generate_random_sequences(secret_config.properties.vocab,
                                     secret_config.properties.pattern, n,
                                     seed + i)
    if len(seqs) < n:
      raise ValueError(
          f'generate_random_sequences was not able to generate {n} sequences. Need to increase vocabulary.'
      )
    secrets_sets.append(construct_secret(secret_config, seqs))
  return secrets_sets


def construct_secret_dataset(
    secrets_sets: Sequence[SecretsSet]) -> MutableSequence[Any]:
  """Repeats secrets for the required number of times to get a secret dataset.

  Args:
    secrets_sets: a sequence of secert sets.

  Returns:
    A sequence of samples.
  """
  secrets_dataset = []
  for secrets_set in secrets_sets:
    for r, seqs in secrets_set.secrets.items():
      secrets_dataset += list(seqs) * r
  return secrets_dataset
