# SPDX-FileCopyrightText: (C) 2023 - 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import pytest

from scene_common.schema import SchemaValidation
import tests.common_test_utils as common

TEST_NAME = "NEX-T10458"
SCHEMA_PATH = "controller/src/schema/metadata.schema.json"
INVALID_SCHEMA_PATH = "../schema/metadata.schema.json"

def pytest_sessionstart():
  print("Executing: " + TEST_NAME)
  return

def pytest_sessionfinish(exitstatus):
  common.record_test_result(TEST_NAME, exitstatus)
  return

@pytest.fixture
def schemaObject():
  schemaObj = SchemaValidation(SCHEMA_PATH)
  return schemaObj

@pytest.fixture
def objData3D():
  timestamp = "1970-01-01T00:00:00.000Z"

  jdata = {"id": "camera1",
          "objects": {},
          "timestamp": timestamp,
          "rate": 9.8}

  obj = {"id": 1,
        "category": "person",
        "confidence": 1,
        "translation": [0.5, 1.0, -0.23],
        "rotation": [0.43, -0.52, 0.12, 0.21],
        "size": [1.0, 2.0, 0.5]}
  jdata['objects']['person'] = [obj]
  return jdata

@pytest.fixture
def objData():
  timestamp = "1970-01-01T00:00:00.000Z"
  point = { "x": 0.56, "y": 0.0, "width": 0.24, "height": 0.49}

  jdata = {"id": "camera1",
          "objects": {},
          "timestamp": timestamp,
          "rate": 9.8}

  obj = {"id": 1,
        "category": "person",
        "confidence": 1,
        "bounding_box": point,
        "center_of_mass": point,
        "bounding_box_px": point}
  jdata['objects']['person'] = [obj]
  return jdata

@pytest.fixture
def singletonData():
  timestamp = "1970-01-01T00:00:00.000Z"
  jdata = {"id": "temp1",
          "timestamp": timestamp,
          "value": 95.8}

  return jdata

@pytest.fixture
def emptyObjData(objData):
  objData = {}
  return objData
