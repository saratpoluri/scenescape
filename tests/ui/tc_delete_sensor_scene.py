#!/usr/bin/env python3

# SPDX-FileCopyrightText: (C) 2022 - 2025 Intel Corporation
# SPDX-License-Identifier: LicenseRef-Intel-Edge-Software
# This file is licensed under the Limited Edge Software Distribution License Agreement.

import os
import time
from tests.ui.browser import Browser
import tests.ui.common_ui_test_utils as common

def test_del_sensor_scene_main(params, record_xml_attribute):
  """! Checks that sensor can still be deleted after the scene the sensor was attached to is deleted.
  @param    params                  Dict of test parameters.
  @param    record_xml_attribute    Pytest fixture recording the test name.
  @return   exit_code               Indicates test success or failure.
  """
  TEST_NAME = "NEX-T10397"
  record_xml_attribute("name", TEST_NAME)
  exit_code = 1
  try:
    print("Executing: " + TEST_NAME)
    print("Test that sensors are not deleted when the parent scene is deleted")
    browser = Browser()
    assert common.check_page_login(browser, params)
    assert common.check_db_status(browser)

    sensor_id = "test_sensor"
    sensor_name = "Sensor_0"
    scene_name = common.TEST_SCENE_NAME
    scale = 1000
    map_image = os.path.join(common.TEST_MEDIA_PATH, "HazardZoneScene.png")

    print("Creating Scene " + scene_name)
    assert common.create_scene(browser, scene_name, scale, map_image)
    assert common.create_sensor_from_scene(browser, sensor_id, sensor_name, scene_name)
    assert common.delete_scene(browser, scene_name)

    time.sleep(1)
    common.verify_sensor_list(browser, [sensor_name])
    assert common.delete_sensor(browser, sensor_name)
    exit_code = 0

  finally:
    browser.close()
    common.record_test_result(TEST_NAME, exit_code)
  assert exit_code == 0
  return
