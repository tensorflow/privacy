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

import itertools
import string
from typing import Dict, List
from dataclasses import dataclass
import numpy as np


def generate_random_sequences(vocab: List[str], pattern: str, n: int,
                              seed: int = 1) -> List[str]:
  """Generate random sequences.

  Args:
    vocab: a list, the vocabulary for the sequences
    pattern: the pattern of the sequence. The length of the sequence will be
      inferred from the pattern.
    n: number of sequences to generate
    seed: random seed for numpy.random

  Returns:
    A list of different random sequences from the given vocabulary
  """
  def count_placeholder(pattern):
    return sum([x[1] is not None for x in string.Formatter().parse(pattern)])

  length = count_placeholder(pattern)
  np.random.seed(seed)
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
      idx_new = np.random.randint(vocab_size, size=(n, length))
      idx = np.concatenate([idx, idx_new], axis=0)  # Add to existing indices
      idx = np.unique(idx, axis=0)  # Remove duplicated indices
    idx = idx[:n]
    seq = np.array(vocab)[idx]
  # Join each row to get the sequence
  seq = np.apply_along_axis(lambda x: pattern.format(*list(x)), 1, seq)
  seq = seq[np.random.permutation(n)]
  return list(seq)


@dataclass
class SecretConfig:
  """Configuration of secret for secrets sharer.

  vocab: a list, the vocabulary for the secrets
  pattern: the pattern of the secrets
  num_repetitions: a list, number of repetitions for the secrets
  num_secrets_for_repetitions: a list, number of secrets to be used for
                               different number of repetitions
  num_references: number of references sequences, i.e. random sequences that
    will not be inserted into training data
  """
  vocab: List[str]
  pattern: str
  num_repetitions: List[int]
  num_secrets_for_repetitions: List[int]
  num_references: int


@dataclass
class Secrets:
  """Secrets for secrets sharer.

  config: configuration of the secrets
  secrets: a dictionary, key is the number of repetitions, value is a list of
           different secrets
  references: a list of references
  """
  config: SecretConfig
  secrets: Dict[int, List[str]]
  references: List[str]


def construct_secret(secret_config: SecretConfig, seqs: List[str]) -> Secrets:
  """Construct a secret instance.

  Args:
    secret_config: configuration of secret.
    seqs: a list of random sequences that will be used for secrets and
          references.
  Returns:
    a secret instance.
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
  return Secrets(config=secret_config,
                 secrets=secrets,
                 references=seqs[-secret_config.num_references:])


def generate_secrets_and_references(secret_configs: List[SecretConfig],
                                    seed: int = 0) -> List[Secrets]:
  """Generate a list of secret instances given a list of configurations.

  Args:
    secret_configs: a list of secret configurations.
    seed: random seed.
  Returns:
    A list of secret instances.
  """
  secrets = []
  for i, secret_config in enumerate(secret_configs):
    n = secret_config.num_references + sum(
        secret_config.num_secrets_for_repetitions)
    seqs = generate_random_sequences(secret_config.vocab, secret_config.pattern,
                                     n, seed + i)
    if len(seqs) < n:
      raise ValueError(
          f'generate_random_sequences was not able to generate {n} sequences. Need to increase vocabulary.'
      )
    secrets.append(construct_secret(secret_config, seqs))
  return secrets
