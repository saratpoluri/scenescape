# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import collections.abc as collections
import torch
string_classes = str


def map_tensor(input_, func):
  if isinstance(input_, torch.Tensor):
    return func(input_)
  elif isinstance(input_, string_classes):
    return input_
  elif isinstance(input_, collections.Mapping):
    return {k: map_tensor(sample, func) for k, sample in input_.items()}
  elif isinstance(input_, collections.Sequence):
    return [map_tensor(sample, func) for sample in input_]
  else:
    raise TypeError(
        f'input must be tensor, dict or list; found {type(input_)}')
