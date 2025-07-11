# SPDX-FileCopyrightText: (C) 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

from warnings import filterwarnings
import logging
from packaging import version

__version__ = "1.3"

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("hloc")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

try:
  import pycolmap
except ImportError:
  logger.warning("pycolmap is not installed, some features may not work.")
else:
  minimal_version = version.parse('0.3.0')
  found_version = version.parse(getattr(pycolmap, '__version__'))
  if found_version < minimal_version:
    logger.warning(
        "hloc now requires pycolmap>=%s but found pycolmap==%s, "
        "please upgrade with `pip install --upgrade pycolmap`",
        minimal_version,
        found_version,
    )

# Warnings filter

filterwarnings(
    "ignore",
    message=r"""__floordiv__ is deprecated, and its behavior will change in a future """
    r"""version of pytorch. It currently rounds toward 0.*""",
)
