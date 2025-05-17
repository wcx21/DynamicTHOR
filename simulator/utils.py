import copy
from PIL import Image
from ai2thor.controller import Controller
from sortedcontainers import SortedDict
from typing import List
from collections import Counter
from ai2thor.server import MetadataWrapper
from ai2thor.util.metrics import (
    get_shortest_path_to_point,
    get_shortest_path_to_object,
    path_distance,
)
from datetime import datetime
from simulator.constants import FIXED_RECEPTACLE_LIST
from simulator.constants import PICKUPABLE_OBJECT_LIST, RECEPTACLE_LIST

class FlexibleDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

class SortedDictDefaultList(SortedDict):
    def __missing__(self, key: datetime) -> list:
        self[key] = value = []
        return value

    def __getitem__(self, __key: datetime) -> list:
        return super().__getitem__(__key)

class SortedDictDefaultDict(SortedDict):
    def __missing__(self, key: datetime) -> dict:
        self[key] = value = {}
        return value

    def __getitem__(self, __key: datetime) -> dict:
        return super().__getitem__(__key)

def unique_list(args: list[any]) -> list[any]:
    new_args = []
    for arg in args:
        if arg not in new_args:
            new_args.append(arg)
    return new_args


def set_top_down_camera(controller: Controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound - 4.0
    pose['position']['x'] += 2.0
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    
    #toilet
    camera1_position = dict(x=9.0, y=1.5, z=1.75)
    camera1_rotation = dict(x=30, y=270, z=0)
    event = controller.step(
        action="AddThirdPartyCamera",
        position=camera1_position,
        rotation=camera1_rotation,
        fieldOfView=90
    )

    #sink
    camera2_position = dict(x=8.75, y=1.25, z=1.0)
    camera2_rotation = dict(x=0, y=180, z=0)
    event = controller.step(
        action="AddThirdPartyCamera",
        position=camera2_position,
        rotation=camera2_rotation,
        fieldOfView=90
    )

    #dining table
    camera3_position = dict(x=7.0, y=1.75, z=2.0)
    camera3_rotation = dict(x=30, y=180, z=0)
    event = controller.step(
        action="AddThirdPartyCamera",
        position=camera3_position,
        rotation=camera3_rotation,
        fieldOfView=90
    )

    #counter top
    camera4_position = dict(x=7.0, y=2.75, z=5.75)
    camera4_rotation = dict(x=60, y=0, z=0)
    event = controller.step(
        action="AddThirdPartyCamera",
        position=camera4_position,
        rotation=camera4_rotation,
        fieldOfView=90
    )

    #dresser
    camera5_position = dict(x=8.75, y=1.25, z=7.25)
    camera5_rotation = dict(x=0, y=0, z=0)
    event = controller.step(
        action="AddThirdPartyCamera",
        position=camera5_position,
        rotation=camera5_rotation,
        fieldOfView=90
    )

    #bed
    camera6_position = dict(x=11.50, y=1.25, z=5.0)
    camera6_rotation = dict(x=0, y=90, z=0)
    event = controller.step(
        action="AddThirdPartyCamera",
        position=camera6_position,
        rotation=camera6_rotation,
        fieldOfView=90
    )

    #sofa
    camera7_position = dict(x=1.75, y=1.25, z=2.25)
    camera7_rotation = dict(x=0, y=90, z=0)
    event = controller.step(
        action="AddThirdPartyCamera",
        position=camera7_position,
        rotation=camera7_rotation,
        fieldOfView=90
    )

def get_top_down_frame(controller: Controller, pid):
    frames = controller.last_event.third_party_camera_frames
    for i,frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save("demo/demo_{}_{}.png".format(i, pid))

def get_scene_object_name(objects: list[dict]):
    return [object_metadata["name"] for object_metadata in objects]

def get_rooms(metadata: MetadataWrapper):
    rooms = []
    for o in metadata['objects']:
        if o['objectType']:
            rooms.append(o)
    return rooms

def filter_name_in_objects(contain_name: str, objects: list[dict]):
    filter_objects = []
    for obj in objects:
        if contain_name.lower() in obj['name'].lower():
            filter_objects.append(obj)
    return filter_objects

def filter_type_in_objects(contain_type: str, objects: list[dict]):
    filter_objects = []
    for obj in objects:
        if contain_type.lower() in obj['objectType'].lower():
            filter_objects.append(obj)
    return filter_objects

def get_fixed_receptacles(metadata: MetadataWrapper):
    return [o for o in metadata['objects'] if o['objectType'] in FIXED_RECEPTACLE_LIST]

def get_object_counter_id_list(objects):
    object_type_counter = Counter()
    object_id_list = []
    for object in objects:
        if isinstance(object, str):
            object_type_counter[object] += 1
            object_id_list.append(f"{object}|{object_type_counter[object]}")
        else:
            object_type_counter[object['objectType']] += 1
            object_id_list.append(f"{object['objectType']}|{object_type_counter[object['objectType']]}")
    return object_id_list

def get_metadata_by_object_id(object_id: str, objects: list[dict]):
    for obj in objects:
        if obj['objectId'] == object_id:
            return obj
    return None

def get_distance_to_object_id(room_id: str, controller: Controller, position: dict[str, float]=None) -> tuple[float, dict[str, float]]:
    """
    return the distance to the room_id and the position of the room_id
    """
    if position is None:
        position = controller.last_event.metadata['agent']['position']
    paths = get_shortest_path_to_object(
        controller=controller,
        object_id=room_id,
        initial_position=position
    )
    return path_distance(paths), paths[-1]


def get_distance_to_point(point: dict[str, float], controller: Controller, initial_position: dict[str, float]=None):
    """
    return the distance to the point
    """
    if initial_position is None:
        initial_position = controller.last_event.metadata['agent']['position']
    paths = get_shortest_path_to_point(
        controller=controller,
        initial_position=initial_position,
        target_position=point
    )
    return path_distance(paths)

def get_path_to_point(target_point: dict[str, float], controller: Controller, initial_position: dict[str, float]=None):
    if initial_position is None:
        initial_position = controller.last_event.metadata['agent']['position']
    paths = get_shortest_path_to_point(
        controller=controller,
        initial_position=initial_position,
        target_position=target_point
    )
    return paths

def get_layout(house_info):
        #room: id, type, children
    #receptacle: id, children
    #pickupable object: id
    objects = house_info['objects']
    rooms = house_info['rooms']
    room_info = []
    receptacle_info = []
    pickupable_object_info = []

    for room in rooms:
        room_info.append({
            'roomId':room['id'],
            'roomType':room['roomType'],
            'roomPolygon':room['floorPolygon'],
            'children':[],
        })
    
    def isIn(obj, room):
        pos = obj['position']
        room_polygon = room['roomPolygon']
        polygon_n = len(room_polygon)
        j = polygon_n - 1
        res = False
        for i in range(polygon_n):
            if ((room_polygon[i]['z']>pos['z'])!=(room_polygon[j]['z']>pos['z']) and \
                pos['x']< (room_polygon[j]['x'] - room_polygon[i]['x']) * (pos['z'] - room_polygon[i]['z'])/(room_polygon[j]['z'] - room_polygon[i]['z']) + room_polygon[i]['x'] ):
                res = not res
            j = i

        return res

    def get_type(oid):
        return oid.split('|')[0]

    def get_receptacle_info(obj):
        rinfo = {
            'id':obj['id'],
        }
        if 'children' in obj:
            rc = []
            for child in obj['children']:
                rc.append(child['id'])
                if get_type(child['id']) in PICKUPABLE_OBJECT_LIST:
                    pickupable_object_info.append(child['id'])
                if get_type(child['id']) in RECEPTACLE_LIST:
                    get_receptacle_info(child)

            if len(rc) >0:
                rinfo['children'] = rc
        
        receptacle_info.append(rinfo)

    for obj in objects:        
        for room in room_info:
            if isIn(obj, room):
                room['children'].append(obj['id'])
                #print("{} in {}".format(obj['id'], room['roomType']))
                break
        if get_type(obj['id']) in RECEPTACLE_LIST:
            get_receptacle_info(obj)
        if get_type(obj['id']) in PICKUPABLE_OBJECT_LIST:
            pickupable_object_info.append(obj['id'])
            
    for r in room_info:
        del r['roomPolygon']

    layout_info = {
        'room':room_info,
        'receptacle': receptacle_info,
        'pickupable object' : pickupable_object_info
    }
    return layout_info

def get_object_poses(metadata):
    objects = metadata['objects']
    obj_poses = []
    for obj in objects:
        if obj['pickupable'] or obj['moveable']:
            obj_pose = {
                'objectName':obj['name'],
                'rotation':obj['rotation'],
                'position':obj['position'],
            }
            obj_poses.append(obj_pose)
    return obj_poses

def translate_participle(verb: str):
    # return conjugate(verb, tense=PARTICIPLE, parse=True)
    # return lexeme(verb)[2]
    # change a verb to its present participle form
    if verb.endswith("ing"):
        return verb
    elif verb.endswith("e"):
        return verb[:-1] + "ing"
    else:
        return verb + "ing"
