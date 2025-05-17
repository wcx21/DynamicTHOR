import os
import sys
import json
import pickle
import datetime
import copy

def get_obj_set(object_dict):
    res = set()
    for obj in object_dict:
        res.add(obj['objectName'])
    return res

class traj_cache():
    def __init__(self, object_poses_dict):
        origin_object_poses = object_poses_dict[list(object_poses_dict.keys())[0]]
        origin_obj_set = get_obj_set(origin_object_poses)

        for time, obj_poses in object_poses_dict.items():
            obj_set = get_obj_set(obj_poses)
            excess = obj_set - origin_obj_set
            if len(excess)>0:
                for obj in obj_poses:
                    if obj['objectName'] in excess:
                        obj_poses.remove(obj)
                object_poses_dict[time] = obj_poses
        object_poses_dict = dict(sorted(object_poses_dict.items(), key=lambda x:x[0]))
        self.object_poses_dict = object_poses_dict

    def get_latest_object_poses(self, new_time):
        for k,v in self.object_poses_dict.items():
            if k >= new_time:
                return v
        return None
        