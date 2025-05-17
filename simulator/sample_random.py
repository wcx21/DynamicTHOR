import pickle
import json
import os
import numpy as np
import random
import copy
import datetime

from simulator.constants import PRIVATE_OBJECT_LIST,COMMONLY_USED_OBJECT_LIST
from simulator.dynamic_thor import Dynamic_Thor
from tqdm import tqdm


def read_data(data_path):
    with open(os.path.join(data_path, "schedules.pkl"), 'rb') as f:
        schedules= pickle.load(f)
    with open(os.path.join(data_path, "tasks.pkl"),'rb') as f:
        tasks = pickle.load(f)
    with open(os.path.join(data_path, "house_info.json"), 'r', encoding='utf-8') as f:
        house_info = json.load(f)
    with open(os.path.join(data_path, "object_poses_dict.pkl"),'rb') as f:
        object_poses_dict = pickle.load(f)
    return schedules, tasks, house_info, object_poses_dict

class Data_Traj():
    def __init__(self):
        pass

    def get_object_poses(self, metadata):
        objects = metadata['objects']
        obj_object_poses = []
        for obj in objects:
            if obj['pickupable'] or obj['moveable']:
                obj_pose = {
                    'objectName':obj['name'],
                    'rotation':obj['rotation'],
                    'position':obj['position'],
                }
                obj_object_poses.append(obj_pose)
        return obj_object_poses

    def play(self, env, schedules):
        object_poses_dict = {}  #key:time, value:object_pose
        for date,day_schedules in schedules.items():
            for activity in day_schedules:
                start_time = activity['start_time']
                time = datetime.datetime.combine(date.date(), start_time)
                env.pass_time(new_time=time)
                object_poses = self.get_object_poses(env.last_event.metadata)
                object_poses_dict[time] = object_poses
        
        last_date = list(schedules.keys())[-1]
        last_hour_minute = datetime.time(hour=23, minute=59)
        last_time = datetime.datetime.combine(last_date.date(), last_hour_minute)
        env.pass_time(new_time=last_time)
        object_poses = self.get_object_poses(env.last_event.metadata)
        object_poses_dict[last_time] = object_poses


        object_poses_dict = dict(sorted(object_poses_dict.items(), key=lambda x:x[0]))
        return object_poses_dict

    def get_random_obj_pose(self, total_data_dir, total_output_dir, start=0, end=-1):
        data_dirs = os.listdir(total_data_dir)
        if end==-1:
            end = len(data_dirs)
        data_dirs = data_dirs[start:end]
        for data_dir in tqdm(data_dirs):
            output_dir = os.path.join(total_output_dir, data_dir)
            if os.path.exists(output_dir):
                continue
            schedules, tasks, house_info, object_poses_dict = read_data(os.path.join(total_data_dir, data_dir))
            random_object_poses = []
            env = Dynamic_Thor(house_info=house_info, fully_scene=True, schedule_path=schedules, object_poses_dict=object_poses_dict)
            for task in tasks:
                event = env.set_task_setting(task)
                env.random_put()
                object_poses = self.get_object_poses(env.last_event.metadata)
                random_object_poses.append(object_poses)
                
            env.stop()
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, 'random_put.json'),'w') as f:
                json.dump(random_object_poses,f,indent=2)
            
                
if __name__=='__main__':
    total_data_dir = '/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/sampled_scenes_2024_no_entropy'
    total_output_dir = '/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/random_put'
    dt = Data_Traj()
    dt.get_random_obj_pose(total_data_dir, total_output_dir)

