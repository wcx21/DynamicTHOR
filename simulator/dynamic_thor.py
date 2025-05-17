from collections import OrderedDict, defaultdict
from datetime import timedelta
import datetime
import json
import numpy as np
import copy
from PIL import Image
import random
from copy import deepcopy
from typing import Any, Dict, Optional
from ai2thor.platform import CloudRendering
import tqdm
from simulator.scene_graph import Fully_Known_Semantic_Map, Partly_Known_Semantic_Map
from simulator.schedule import Schedule_Table
from simulator.utils import get_object_poses
from simulator.smooth_thor import Smooth_Thor
from simulator.constants import ACTIONS_SET, PRIVATE_OBJECT_LIST, COMMONLY_USED_OBJECT_LIST
from simulator.traj_cache import traj_cache

class Dynamic_Thor(Smooth_Thor):
    def __init__(
        self, house_info: dict, fully_scene: bool, schedule_path: str, object_poses_dict = None, headless=True, gridSize=0.25, 
        snapToGrid=True, smooth_nav=False, record_video=False, **init_params
    ):
        # dynamic_info_schedule is OrderedDict[float, List], and it is sorted by key
        self.disable_object = set()
        # self.schedule_table = Schedule_Table()
        self.agent_init_state = house_info['metadata']['agent']
        if headless:
            init_params.pop('gridSize',None)
            # print(init_params)
            super().__init__(
                scene=house_info, platform=CloudRendering, gridSize=gridSize, 
                snapToGrid=snapToGrid, smooth_nav=smooth_nav, record_video=record_video,
                **init_params
            )
        else:
            super().__init__(
                scene=house_info, gridSize=gridSize, snapToGrid=snapToGrid, 
                smooth_nav=smooth_nav, record_video=record_video
            )
        metadata = self.last_event.metadata
        if fully_scene:
            self.scene_graph = Fully_Known_Semantic_Map(metadata, house_info)
        else:
            self.scene_graph = Partly_Known_Semantic_Map(metadata, house_info)
        self.schedule_table = Schedule_Table(scene_graph=self.scene_graph, path=schedule_path)
        self.object_traj_cache = None
        if object_poses_dict is not None:
            self.object_traj_cache = traj_cache(object_poses_dict)
        self.step_count = 0
        self.trace = []

    def get_object_metadata_by_object_id(self, target_object_id):
        for obj in self.last_event.metadata['objects']:
            if obj['objectId'] == target_object_id:
                return obj
        raise Exception("No object with id {}".format(target_object_id))

    def check_receptacle_is_direct_parent(self, target_object_id, target_receptacle_id):
        target_object_metadata = self.get_object_metadata_by_object_id(target_object_id)
        if target_object_metadata['parentReceptacles'] is None or not target_receptacle_id in target_object_metadata['parentReceptacles']:
            return False
        for rep in target_object_metadata['parentReceptacles']:
            if rep == target_receptacle_id:
                continue
            rep_metadata = self.get_object_metadata_by_object_id(rep)
            if rep_metadata['parentReceptacles'] is not None and target_receptacle_id in rep_metadata['parentReceptacles']:
                # the target_receptacle_id is the grandparent of target_object_id
                return False
        return True

    def change_env_with_dynamic_info(self, infos: list[dict], scan_put=True) -> None:
        for info in infos:
            target_object_id = info['object']
            target_receptacle_id = info['receptacle']
            target_object_metadata = self.get_object_metadata_by_object_id(target_object_id)
            
            if target_object_id in self.disable_object:
                event = self.step(
                    action='EnableObject',
                    objectId=target_object_id,
                )
                self.disable_object.remove(target_object_id)
                target_from_receptacle_id = 'Outdoor'
            elif target_object_metadata['parentReceptacles'] is not None and len(target_object_metadata['parentReceptacles']) > 0:
                target_from_receptacle_id = target_object_metadata['parentReceptacles'][0]
            else:
                target_from_receptacle_id = ""

            if target_receptacle_id == 'Outdoor':
                event = self.step(
                    action='DisableObject',
                    objectId=target_object_id,
                )
                self.disable_object.add(target_object_id)
                print(f"move {target_object_id} from {target_from_receptacle_id} to {target_receptacle_id} successful")
            else:
                # if target object is alfredy in the target receptacle, then skip, and we don't need to take the photo of this action
                if target_from_receptacle_id != 'Outdoor' and self.check_receptacle_is_direct_parent(target_object_id, target_receptacle_id):
                    continue
                
                # get the reachable points above the target receptacle
                event = self.step(
                    action="GetSpawnCoordinatesAboveReceptacle",
                    objectId=target_receptacle_id,
                    anywhere=True
                )
            
                points = event.metadata['actionReturn']
                if points is not None:
                    random.shuffle(points)
                    for point in points:
                        event = self.step(
                            action="PlaceObjectAtPoint",
                            objectId=target_object_id,
                            position=point
                        )
                        if event.metadata['lastActionSuccess'] and self.check_receptacle_is_direct_parent(target_object_id, target_receptacle_id):
                            print(f"move {target_object_id} from {target_from_receptacle_id} to {target_receptacle_id} successful")
                            break
                    else:
                        #todo scan points put
                        if scan_put:
                            put_success = self.scan_put(target_object_id, target_receptacle_id)
                            if not put_success:
                                print(f"move {target_object_id} from {target_from_receptacle_id} to {target_receptacle_id} failed")
                            else:
                                print(f"move {target_object_id} from {target_from_receptacle_id} to {target_receptacle_id} successful")
                else:
                    if scan_put:
                        put_success = self.scan_put(target_object_id, target_receptacle_id)
                        if not put_success:
                            print(f'{target_receptacle_id} no points for {target_object_id}')
                        else:
                            print(f"move {target_object_id} from {target_from_receptacle_id} to {target_receptacle_id} successful")
  
    def pass_time(self, delta_time: timedelta=None, new_time: datetime=None):
        assert (delta_time is None) or (new_time is None)
        if delta_time is not None:
            new_time = self.schedule_table.current_time + delta_time
        elif new_time is None:
            raise Exception("delta_time and new_time can not be both None")
        
        if self.object_traj_cache is not None:
            object_poses = self.object_traj_cache.get_latest_object_poses(new_time)
            if object_poses is not None:
                event = self.step(action='SetObjectPoses', objectPoses=object_poses)
                if event.metadata['lastActionSuccess']:
                    self.schedule_table.set_current_time(new_time)
        action_list = self.schedule_table.pass_time(delta_time, new_time)
        # the action_list is a sequence of actions, if two action move the same object, we can just delete the first one
        move_object_set = set()
        new_action_list = []
        for action in action_list[::-1]:
            # action is a dict like {"object": "Cloth|surface|3|11", "receptacle": "Dresser|5|1"}
            if action['object'] in move_object_set:
                continue
            new_action_list.append(action)
            move_object_set.add(action['object'])
        new_action_list.reverse()
        self.change_env_with_dynamic_info(new_action_list)
        
    def step(self, action, **action_args):
        if isinstance(action, dict):
            action_name = action['action']
        else:
            action_name = action
        event = super().step(action, **action_args)
        if action_name in ACTIONS_SET:
            self.step_count += 1
            self.trace.append(event.metadata['agent'])
        return event
    
    def initialize_agent_position(self):
        self.step(action='TeleportFull', **self.agent_init_state)
    
    def reset_task(self, task: dict):
        """
        TODO: now we use the pass time to generate the dynamic info, but it can be useful only when the new task_time > current_time
        """
        # reset the agent position and the object pose by time in ai2thor use the step function by random
        # self.initialize_agent_position()
        # self.step_count = 0
        self.trace = []
        # if use_cache_object_poses and len(task.object_poses) > 0:
        #     self.step(action='SetObjectPoses', objectPoses=task.object_poses)
        # else:
        #     self.pass_time(new_time=task.current_time)
        #     # save the pose in task
        #     task.object_poses = get_object_poses(self.last_event.metadata)
        self.set_task_setting(task)

    def reset(self, scene=None, **init_params):
        self.step_count = 0
        return super().reset(scene, **init_params)

    def set_task_setting(self, task, static=False, random_obj=False, random_object_pose=None):
        time = task['time']
        target_object = task['target_object']
        target_object_type = target_object.split('|')[0]
        self.target_obj_id = target_object

        self.initialize_agent_position()
        disable_object_list = list(self.disable_object)
        for obj in disable_object_list:
            event = self.step(
                    action='EnableObject',
                    objectId=obj,
                )
            self.disable_object.remove(obj)
        if not static:
            self.pass_time(new_time=time)
        self.step_count = 0
        

        objects = self.last_event.metadata['objects']
        for obj in objects:
            if obj['objectId'] != target_object and obj['objectType'] == target_object_type and obj['objectId'] not in self.disable_object:
                event = self.step(
                    action='DisableObject',
                    objectId=obj['objectId'],
                )
                self.disable_object.add(obj['objectId'])
        
        if random_obj:
            if random_object_pose is not None:
                # objects = self.last_event.metadata['objects']
                new_obj_pose = []
                for obj_pose in random_object_pose:
                    obj_name = obj_pose['objectName']
                    if obj_name in self.disable_object:
                        continue
                    new_obj_pose.append(obj_pose)
                        
                event = self.step(action='SetObjectPoses', objectPoses=new_obj_pose)
                if not event.metadata['lastActionSuccess']:
                    print(event.metadata['errorMessage'])
                    self.random_put()
            else:
                self.random_put()
            
  
        return self.last_event

    def scan_put(self, target_object_id, receptacle_id):
        event = self.last_event
        objects = event.metadata['objects']
        target_recep_metadata = None
        for obj in objects:
            if obj['objectId'] == receptacle_id:
                target_recep_metadata = obj
        if target_recep_metadata is None:
            print('no target')
            return False
        recep_center = target_recep_metadata['axisAlignedBoundingBox']['center']
        recep_size = target_recep_metadata['axisAlignedBoundingBox']['size']
        interval = 0.05
        height_interval = 0.3
        start_position = dict(x = recep_center['x'] - recep_size['x']/2,
                            y = recep_center['y'] + recep_size['y']/2 + height_interval,
                            z = recep_center['z'] - recep_size['z']/2)
        x_scan_len = int(recep_size['x']/interval)
        z_scan_len = int(recep_size['z']/interval)
        put_success = False
        for i in range(x_scan_len):
            for j in range(z_scan_len):
                position = dict(x = start_position['x'] + interval * i,
                                y = start_position['y'],
                                z = start_position['z'] + interval * j)
                event = self.step(
                        action="PlaceObjectAtPoint",
                        objectId=target_object_id,
                        position=position
                    )
                if event.metadata['lastActionSuccess']:
                    # print("move  successful")
                    # for obj in event.metadata['objects']:
                    #     if obj['objectId'] == target_object_id:
                    #         print(obj)
                    put_success = True
                    break    
            if put_success:
                break
        
        return put_success


    def random_put(self):
        objs = self.last_event.metadata['objects']
        candidate_small_objs = [self.target_obj_id]
        candidate_receptacles = []
        for obj in objs:
            if obj['objectId'] in candidate_small_objs:
                continue
            if '___' in obj['objectId']:
                continue
            receptacle = obj['receptacle']
            pickupable = obj['pickupable']
            if pickupable:
                candidate_small_objs.append(obj['objectId'])
            elif receptacle:
                candidate_receptacles.append(obj['objectId'])
        
        dynamic_infos = []
        for small_obj in candidate_small_objs:
            if small_obj in self.disable_object:
                continue
            recep = random.sample(candidate_receptacles, 1)[0]
            dynamic_info = {
                "object":small_obj,
                "receptacle":recep
            }
            dynamic_infos.append(dynamic_info)
            
        self.change_env_with_dynamic_info(dynamic_infos, False)
        return self.last_event
            
        
        