# Copyright (C) 2021-2025 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the License.

import math

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation

from fast_geometry import Point, Line, Rectangle, Polygon, Size

DEFAULTZ = 0
ROI_Z_HEIGHT = 0.25

# Re-export modules from fast geometry as our own
__all__ = ['Point', 'Line', 'Rectangle', 'Size']

def isarray(a):
  return isinstance(a, (list, tuple, np.ndarray))

class Region:
  REGION_SCENE = 0
  REGION_POLY = 1
  REGION_CIRCLE = 2

  def __init__(self, uuid, name, info):
    self.uuid = uuid
    self.name = name
    self.area = None
    self.mesh = None
    self.updatePoints(info)
    self.objects = {}
    self.when = -1
    self.points_list = None
    self.polygon = None
    self.singleton_type = None
    self.updateSingletonType(info)
    self.compute_intersection = False
    self.region_height = ROI_Z_HEIGHT
    self.buffer_size = 0.0
    return

  def updatePoints(self, newPoints):
    if (not isarray(newPoints) and 'center' in newPoints):
      pt = newPoints['center']
      self.center = pt if isinstance(pt, Point) else Point(pt)

    if isarray(newPoints) or ('area' in newPoints and newPoints['area'] == "poly"):
      self.area = Region.REGION_POLY
      self.points = []
      if not isarray(newPoints):
        newPoints = newPoints['points']
      for pt in newPoints:
        self.points.append(pt if isinstance(pt, Point) else Point(pt))
      self.findBoundingBox()
      self.points_list = [x.as2Dxy.asCartesianVector for x in self.points]
      if len(self.points_list) > 2:
        self.polygon = Polygon(self.points_list)
    elif 'area' in newPoints and newPoints['area'] == "circle":
      self.area = Region.REGION_CIRCLE
      self.radius = newPoints['radius']
      # Rectangle is created using Point, Point constructor.
      self.boundingBox = Rectangle(self.center - (self.radius, self.radius),
                                   self.center + (self.radius, self.radius))
    elif 'area' in newPoints and newPoints['area'] == "scene":
      self.area = Region.REGION_SCENE
    else:
      raise ValueError("Unrecognized point data", newPoints)
    return

  def updateSingletonType(self, info):
    if isinstance(info, dict):
      self.singleton_type = info.get('singleton_type', None)
    return

  def findBoundingBox(self):
    tx, ty = self.points[0].as2Dxy.asCartesianVector
    bx = tx
    by = ty
    for point in self.points:
      tx = min(tx, point.x)
      ty = min(ty, point.y)
      bx = max(bx, point.x)
      by = max(by, point.y)
    self.boundingBox = Rectangle(origin=Point(tx, ty),
                                 opposite=Point(bx, by))
    return
  
  def createMesh(self):
    roi_pts = [[pt.x, pt.y, 0] for pt in self.points]
    roi_pts += [[pt.x, pt.y] + [self.region_height] for pt in self.points]
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
      o3d.utility.Vector3dVector(np.array(roi_pts, dtype=np.float64)),
      extent = np.array([self.buffer_size * 2] * 3)
      )
    translation_bc_adjust = np.vstack([
      np.hstack([
        np.identity(3),
        np.array([0, 0, obb.extent[2]/2]).reshape((3,1))
        ]),
      np.array([0, 0, 0, 1])
    ])
    R_center = np.vstack([
      np.hstack([np.array(obb.R), np.zeros((3,1))]),
      np.array([0, 0, 0, 1])
      ])
    translation_center = np.vstack([
      np.hstack([
        np.identity(3),
        np.array(obb.center).reshape(3,1)
        ]),
      np.array([0, 0, 0, 1])
    ])
    T_RC = translation_center @ R_center @ np.linalg.inv(translation_bc_adjust)
    self.translation = T_RC[:-1,3].tolist()
    self.rotation = Rotation.from_matrix(np.array(obb.R)).as_quat().tolist()
    self.size = obb.extent.tolist()
    self.mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb).compute_vertex_normals()
    
  def createObjectMesh(self, obj):
    # populate object
    mesh = o3d.geometry.TriangleMesh.create_box(
      obj.size[0],
      obj.size[1],
      obj.size[2]
      ).translate(np.array([
        -obj.size[0]/2,
        -obj.size[1]/2,
        0
        ])).rotate(
          Rotation.from_quat(np.array(obj.rotation)).as_matrix(),
          center = np.zeros(3)
          ).translate(obj.sceneLoc.asNumpyCartesian)
    obj.mesh = mesh.compute_vertex_normals()
    return obj

  def is_intersecting(self, obj):
    if not self.compute_intersection:
      return False

    if self.mesh == None:
      self.createMesh()
    self.createObjectMesh(obj)
    return obj.mesh.is_intersecting(self.mesh)

  def isPointWithin(self, coord):
    if self.area == Region.REGION_SCENE:
      return True

    if not self.boundingBox.isPointWithin(coord):
      return False

    if self.area == Region.REGION_POLY:

      # if len(self.points) == 4:
      #   # Quadrilateral speed optimization
      #   diag1_x = abs(self.points[2].x - self.points[0].x)
      #   diag1_y = abs(self.points[2].y - self.points[0].y)
      #   diag2_x = abs(self.points[3].x - self.points[1].x)
      #   diag2_y = abs(self.points[3].y - self.points[1].y)
      #   dist_x = coord.x - self.points[0].x
      #   dist_y = coord.y - self.points[0].y
      #   area1 = diag1_x * diag1_x + diag1_y * diag1_y
      #   area2 = diag1_x * dist_x + diag1_y * dist_y
      #   area3 = diag2_x * diag2_x + diag2_y * diag2_y
      #   area4 = diag2_x * dist_x + diag2_y * dist_y
      #   return area1 >= area2 >= 0 and area3 >= area4 >= 0

      if len(self.points) > 2:
        if self.polygon is None:
          pts = [x.as2Dxy.asNumpyCartesian.flatten().tolist() for x in self.points]
          self.polygon = Polygon(pts)
        return self.polygon.isPointInside(coord.x, coord.y)

      return False

    dx = abs(coord.x - self.center.x)
    dy = abs(coord.y - self.center.y)

    if dx + dy <= self.radius:
      return True
    if dx*dx + dy*dy <= self.radius*self.radius:
      return True
    return False

  def serialize(self):
    data = {'points':[], 'title':self.name, 'uuid':self.uuid}
    if self.area == self.REGION_SCENE:
      data['area'] = "scene"
    elif self.area == self.REGION_CIRCLE:
      data['area'] = "circle"
      data['radius'] = self.radius
    elif self.area == self.REGION_POLY:
      data['area'] = "poly"
      data['points'] = self.coordinates
    if hasattr(self, "center"):
      data['x'], data['y'] = self.center.x, self.center.y
    return data

  @property
  def cv(self):
    return [x.cv for x in self.points]

  @property
  def coordinates(self):
    if hasattr(self, 'points'):
      return [np.array(x.asCartesianVector).tolist() for x in self.points]
    return None

  def __repr__(self):
    return "%s: person:%i vehicle:%i %s" % \
      (self.__class__.__name__,
       len(self.objects.get('person', [])), len(self.objects.get('vehicle', [])),
       self.coordinates)

class Tripwire(Region):
  def lineCrosses(self, line):
    for idx in range(len(self.points) - 1):
      pt1 = self.points[idx]
      pt2 = self.points[(idx + 1) % len(self.points)]
      segment = Line(pt1, pt2)
      isect = line.intersection(segment)
      if isect[0] and line.isPointOnLine(Point(isect[1])) \
          and segment.isPointOnLine(Point(isect[1])):
        direction = (line.x2 - segment.x1) * (segment.y2 - segment.y1) \
            - (line.y2 - segment.y1) * (segment.x2 - segment.x1)
        return int(math.copysign(1, direction))
    return 0

  def serialize(self):
    data = {
      'title': self.name,
      'points': self.coordinates,
      'uuid': self.uuid,
    }
    return data
