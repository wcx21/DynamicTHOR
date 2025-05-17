from __future__ import annotations
import copy
from typing import List
import numpy as np
from shapely.geometry import Polygon

class Point:
    def __init__(self, x, y, z, rgb=(0, 0, 0), eps=3, stride=10, status=None):
        self.x = round(x, eps)
        self.y = round(y, eps)
        self.z = round(z, eps)
        self.stride = stride
        self.r = int(rgb[0])
        self.g = int(rgb[1])
        self.b = int(rgb[2])
        self.eps = eps
        self.status = status

    def __eq__(self, other):
        # max_dis = max(abs(self.x - other.x), max(abs(self.y - other.y), abs(self.z - other.z)))
        # max_col_dif = max(abs(self.r - other.r), max(abs(self.g - other.g), abs(self.b - other.b)))
        # dis = np.sqrt(np.square(self.x - other.x) + np.square(self.y - other.y) + np.square(self.z - other.z))
        # return dis <= 0.02
        return self.x == other.x and self.z == other.z

    def __str__(self):
        return 'x:{} y:{} z:{} r:{} g:{} b:{}'.format(self.x, self.y, self.z, self.r, self.g, self.b)

    def __hash__(self):
        return hash(1e4 * self.x + 1e2 * self.y + self.z)

    def __add__(self, other: Point):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: Point):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def length(self):
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))
    
    @property
    def dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}
    
def generate_point_list_with_rgb(frame, point_cloud, res=[]):
    # frame: HxWx3
    # point_cloud: HxWx3, the word coordinate of the points
    # return: points: a list contains class Point with XYZ、RGB info
    points = set()
    points.update(copy.deepcopy(res))
    h = frame.shape[0]
    w = frame.shape[1]
    for h_ind in range(h):
        for w_ind in range(w):
            x, y, z = point_cloud[h_ind, w_ind]
            r, g, b = frame[h_ind, w_ind]
            point = Point(x, y, z, (r, g, b))
            points.add(point)

    return list(points)


class Room2D:
    def __init__(self, vertices):
        """
        vertices: list of dict {"x": x, "y": y, "z": z} but the y is the same in all vertices
        """
        self.vertices = [(v["x"], v["z"]) for v in vertices]
        self.polygon = Polygon(self.vertices)
    
    def contain_point(self, position: dict):
        """
        ar
        position: dict {"x": x, "y": y, "z": z}
        """
        corner = (position["x"], position["z"])
        from shapely.geometry import Point as ShapelyPoint
        return self.polygon.contains(ShapelyPoint(corner))

    def get_center(self):
        x = sum([v[0] for v in self.vertices]) / len(self.vertices)
        z = sum([v[1] for v in self.vertices]) / len(self.vertices)
        return {"x": x, "y": 0, "z": z}


class Bbox3D:

    def __init__(self, vertices, obj_cls=0):
        points: List[Point] = []
        for vertex in vertices:
            point = Point(vertex[0], vertex[1], vertex[2])
            points.append(point)

        self.width = vertices[0][0] - vertices[7][0]  # x-axis aligned
        self.height = vertices[0][1] - vertices[7][1]  # y-axis aligned
        self.thick = vertices[0][2] - vertices[7][2]  # z-axis aligned
        self.lbb = points[-1]
        self.voxel = self.width * self.height * self.thick
        self.center = (self.lbb.x + self.width / 2., self.lbb.y + self.height / 2., self.lbb.z + self.thick / 2.)
        self.type = obj_cls

    def __str__(self):
        return 'the center of bbox is: {}\nthe size of bbox is: {} * {} * {}\n'.format(
            self.center, self.width, self.height, self.thick
        )

    # def get_iou_with_point_cloud(self, points: List[Point]):
    #     total = len(points)
    #     # print('total points:',total)
    #     inside = 0
    #     for point in points:
    #         if self.is_in_bbox(point):
    #             inside += 1
    #     return inside / total
    
    def expand_bbox(self, expand_length: float):
        """expand the bbox with the expand_length
        :param expand_length: the length to expand
        """
        self.lbb.x -= expand_length
        self.lbb.y -= expand_length
        self.lbb.z -= expand_length
        self.width += 2 * expand_length
        self.height += 2 * expand_length
        self.thick += 2 * expand_length
        self.voxel = self.width * self.height * self.thick
    
    def is_in_bbox(self, point: tuple[float, float]):
        if self.type == 0 or self.type == 2:
            x_exp = 0.03
            z_exp = 0.03
        else:
            x_exp = 0.01
            z_exp = 0.01

        x, z = point
        if self.lbb.x - x_exp <= x <= (self.lbb.x + self.width + x_exp) and \
            self.lbb.z - z_exp <= z <= (self.lbb.z + self.thick + z_exp):
            return True
        return False

    def contain_bbox(self, bbox: Bbox3D):
        if self.lbb.x <= bbox.lbb.x and self.lbb.y <= bbox.lbb.y and self.lbb.z <= bbox.lbb.z and \
                self.lbb.x + self.width >= bbox.lbb.x + bbox.width and \
                self.lbb.y + self.height >= bbox.lbb.y + bbox.height and \
                self.lbb.z + self.thick >= bbox.lbb.z + bbox.thick:
            return True
        else:
            return False

    def get_iou_with_bbox(self, bbox: Bbox3D):
        """calculate the iou between two bbox
        :param bbox: the other bbox
        :return: iou
        """
        if self.contain_bbox(bbox) or bbox.contain_bbox(self):
            return 1.
        if self.lbb.x > bbox.lbb.x + bbox.width or bbox.lbb.x > self.lbb.x + self.width or \
                self.lbb.y > bbox.lbb.y + bbox.height or bbox.lbb.y > self.lbb.y + self.height or \
                self.lbb.z > bbox.lbb.z + bbox.thick or bbox.lbb.z > self.lbb.z + self.thick:
            return 0.
        # calculate the intersection
        inter_x = min(self.lbb.x + self.width, bbox.lbb.x + bbox.width) - max(self.lbb.x, bbox.lbb.x)
        inter_y = min(self.lbb.y + self.height, bbox.lbb.y + bbox.height) - max(self.lbb.y, bbox.lbb.y)
        inter_z = min(self.lbb.z + self.thick, bbox.lbb.z + bbox.thick) - max(self.lbb.z, bbox.lbb.z)
        inter = inter_x * inter_y * inter_z
        union = self.voxel + bbox.voxel - inter
        return inter / union

