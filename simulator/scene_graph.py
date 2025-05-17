from collections import Counter, defaultdict
from enum import Enum
import json
from simulator.constants import FIXED_RECEPTACLE_LIST
import simulator.depth_utils as du
from simulator.utils import get_fixed_receptacles, get_object_counter_id_list
from ai2thor.server import MetadataWrapper
from abc import ABCMeta, abstractmethod
from collections import Counter

class Semantic_Map(metaclass=ABCMeta):
    def __init__(self, metadata: MetadataWrapper, house: dict) -> None:
        # init the semantic map with the room level position
        self.room_metadata_list = house['rooms']
        self.room_palyons = [du.Room2D(room["floorPolygon"]) for room in self.room_metadata_list]
        self.room_scene_id_list = [room["id"] for room in self.room_metadata_list]
        rooms_type = [room["roomType"] for room in self.room_metadata_list]
        self.room_counter_id_list = get_object_counter_id_list(rooms_type)
        self.explored_room_counter_id_set = set()
        self.room_step_count = Counter()
        
    @abstractmethod
    def get_scene_graph_prompt(self) -> str:
        pass
    
    @abstractmethod
    def get_receptacle_counter_id_list(self) -> list[str]:
        pass
    
    @abstractmethod
    def ground_receptacle_counter_id_in_room(self, receptacle_counter_id: str, room_counter_id: str):
        pass
    
    @abstractmethod
    def get_explored_receptacle_counter_id_list(self) -> list[str]:
        pass

    @abstractmethod
    def get_unexplored_receptacle_counter_id_list(self) -> list[str]:
        pass
    
    @abstractmethod
    def update_receptacle_and_object(self, receptacles: list[tuple[str, dict]], objects: list[tuple[str, dict]], relation: list[tuple[int, int]]):
        pass
    
    @abstractmethod
    def set_explored_receptacle(self, explored_receptacle_counter_id: str):
        pass
    
    @abstractmethod
    def check_receptacle_maybe_in_room(self, room_counter_id: str, receptacle_counter_id: str) -> bool:
        pass
    
    @abstractmethod
    def check_receptacle_sure_in_room(self, room_counter_id: str, receptacle_counter_id: str) -> bool:
        pass
    
    @abstractmethod
    def check_receptacle_maybe_in_house(self, receptacle_counter_id: str):
        pass
    
    @abstractmethod
    def check_receptacle_sure_in_house(self, receptacle_counter_id: str):
        pass
    
    @abstractmethod
    def get_receptacle_counter_id_from_object_type_in_it(self, object_type: str):
        pass
    
    @abstractmethod
    def get_room_counter_id_from_object_type_in_it(self, object_type: str):
        pass
    
    @abstractmethod
    def get_receptacle_scene_id_from_receptacle_counter_id(self, receptacle_counter_id: str):
        pass
    
    def get_room_counter_id_from_room_scene_id(self, rooom_scene_id: str):
        return self.room_counter_id_list[self.room_scene_id_list.index(rooom_scene_id)]
    
    @abstractmethod
    def get_room_counter_id_from_receptacle_counter_id(self, receptacle_counter_id: str):
        pass
    
    @abstractmethod
    def ground_receptacle_counter_id(self, receptacle_counter_id: str) -> list[str]:
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    def set_explored_room(self, room_counter_id: str):
        self.explored_room_counter_id_set.add(room_counter_id)
        
    def get_unexplored_room_counter_id_list(self) -> list[str]:
        return [room_counter_id for room_counter_id in self.room_counter_id_list if room_counter_id not in self.explored_room_counter_id_set]
    
    # 确认一个 point 在哪个房间中
    def get_room_counter_id_from_position(self, position: dict):
        for room_idx, room_paylon in enumerate(self.room_palyons):
            if room_paylon.contain_point(position):
                return self.room_counter_id_list[room_idx]
        raise ValueError(f"position {position} is not in any room")

    def get_room_scene_id_from_room_counter_id(self, room_counter_id: str):
        return self.room_scene_id_list[self.room_counter_id_list.index(room_counter_id)]

    def get_room_counter_id_list(self):
        return self.room_counter_id_list

    def get_room_position_from_counter_id(self, room_counter_id: str) -> dict[str, float]:
        room_idx = self.room_counter_id_list.index(room_counter_id)
        return self.room_palyons[room_idx].get_center()
        
    @staticmethod
    def json2str(data):
        return json.dumps(data)
    
    @staticmethod
    def get_type_from_id(counter_id: str):
        return counter_id.split("|")[0]
    
    @staticmethod
    def get_counter_idx_from_id(counter_id: str):
        return int(counter_id.split("|")[-1])

class Fully_Known_Semantic_Map(Semantic_Map):
    def __init__(self, metadata: MetadataWrapper, house: dict):
        """
        metadata: ai2thor.server.MetadataWrapper
        house: the procthor dataset house, for example:
            dataset = prior.load_dataset("procthor-10k")
            house = dataset["train"][1]
        """
        super().__init__(metadata, house)
        # we can get all receptacle data from metadata
        self.receptacle_metadata_list = get_fixed_receptacles(metadata)
        self.receptacle_centers = [receptacle['axisAlignedBoundingBox']['center'] for receptacle in self.receptacle_metadata_list]
        self.receptacle_scene_id_list = [receptacle["objectId"] for receptacle in self.receptacle_metadata_list]
        self.receptacle_counter_id_list = get_object_counter_id_list(self.receptacle_metadata_list)
        self.receptacle_bbox3d_list = [du.Bbox3D(receptacle['axisAlignedBoundingBox']['cornerPoints']) for receptacle in self.receptacle_metadata_list]
        
        # get the scene_id to counter_id mapping and counter_id to scene_id mapping
        self.scene_id2counter_id = {}
        self.counter_id2scene_id = {}
        for room_scene_id, room_counter_id in zip(self.room_scene_id_list, self.room_counter_id_list):
            self.scene_id2counter_id[room_scene_id] = room_counter_id
            self.counter_id2scene_id[room_counter_id] = room_scene_id
        for receptacle_scene_id, receptacle_counter_id in zip(self.receptacle_scene_id_list, self.receptacle_counter_id_list):
            self.scene_id2counter_id[receptacle_scene_id] = receptacle_counter_id
            self.counter_id2scene_id[receptacle_counter_id] = receptacle_scene_id

        # get the room-receptacle relations, which is a list of tuple (room_index, receptacle_index)
        # it means a receptacle is in a room
        self.room_receptacle_relations: list[tuple[int, int]] = []
        definite_receptacle_index_set = set()
        for room_index, room_paylon in enumerate(self.room_palyons):
            for receptacle_index, receptacle_center in enumerate(self.receptacle_centers):
                if receptacle_index not in definite_receptacle_index_set \
                    and room_paylon.contain_point(receptacle_center):
                    self.room_receptacle_relations.append((room_index, receptacle_index))
                    definite_receptacle_index_set.add(receptacle_index)

        # get the room2receptacles and receptacle2room to better describe the relations between rooms and receptacles
        self.room2receptacles = [[] for _ in range(len(self.room_counter_id_list))]
        self.receptacle2room = [0 for _ in range(len(self.receptacle_counter_id_list))]
        for room_idx, receptacle_idx in self.room_receptacle_relations:
            self.room2receptacles[room_idx].append(receptacle_idx)
            self.receptacle2room[receptacle_idx] = room_idx

        self.reset()
    
    def reset(self):
        # we just remove the tiny objects and fully explored receptacles
        self.receptacle2tinyobjects = [[] for _ in range(len(self.receptacle_counter_id_list))]
        self.fully_explored_receptacles = [False for _ in range(len(self.receptacle_counter_id_list))]
    
    def get_counter_id_from_scene_id(self, scene_id: str):
        return self.scene_id2counter_id[scene_id]
    
    def get_room_counter_id_from_receptacle_counter_id(self, receptacle_counter_id: str):
        room_idx = self.receptacle2room[self.receptacle_counter_id_list.index(receptacle_counter_id)]
        return self.room_counter_id_list[room_idx]
    
    def get_room_counter_id_from_receptacle_scene_id(self, receptacle_scene_id: str):
        room_idx = self.receptacle2room[self.receptacle_scene_id_list.index(receptacle_scene_id)]
        return self.room_counter_id_list[room_idx]
    
    def get_room_counter_id_from_object_type_in_it(self, object_type: str):
        receptacle_counter_id = self.get_receptacle_counter_id_from_object_type_in_it(object_type)
        if receptacle_counter_id is None:
            return None
        else:
            return self.get_room_counter_id_from_receptacle_counter_id(receptacle_counter_id)
    
    def get_receptacle_counter_id_from_object_type_in_it(self, object_type: str):
        # travel each receptacle to check the tiny objects in it
        for receptacle_idx in range(len(self.receptacle_counter_id_list)):
            receptacle_counter_id = self.receptacle_counter_id_list[receptacle_idx]
            tiny_objects = self.get_tiny_object(receptacle_idx)
            if object_type in tiny_objects:
                return receptacle_counter_id
        return None
            
    def check_receptacle_maybe_in_room(self, room_counter_id: str, receptacle_counter_id: str) -> bool:
        return receptacle_counter_id in self.get_receptacle_counter_id_list_from_room_counter_id(room_counter_id)
    
    def check_receptacle_sure_in_room(self, room_counter_id: str, receptacle_counter_id: str):
        return self.check_receptacle_maybe_in_room(room_counter_id, receptacle_counter_id)
    
    def check_receptacle_maybe_in_house(self, receptacle_counter_id: str):
        return receptacle_counter_id in self.receptacle_counter_id_list
    
    def check_receptacle_sure_in_house(self, receptacle_counter_id: str):
        """
        because the receptacle are fully known in this scene graph, so the maybe in and sure in is the same
        """
        return self.check_receptacle_maybe_in_house(receptacle_counter_id)
    
    def get_receptacle_counter_id_list_from_room_scene_id(self, room_scene_id: str):
        room_idx = self.room_scene_id_list.index(room_scene_id)
        receptacle_idx_list = self.room2receptacles[room_idx]
        return [self.receptacle_counter_id_list[receptacle_idx] for receptacle_idx in receptacle_idx_list]
    
    def get_receptacle_counter_id_list_from_room_counter_id(self, room_counter_id: str):
        room_idx = self.room_counter_id_list.index(room_counter_id)
        receptacle_idx_list = self.room2receptacles[room_idx]
        return [self.receptacle_counter_id_list[receptacle_idx] for receptacle_idx in receptacle_idx_list]    
    
    def get_room_receptacle_relations(self):
        data = defaultdict(list)
        for room_idx in range(len(self.room_counter_id_list)):
            room_id = self.room_counter_id_list[room_idx]
            receptacle_idx_list = self.room2receptacles[room_idx]
            receptacle_counter_id_list = [self.receptacle_counter_id_list[receptacle_idx] for receptacle_idx in receptacle_idx_list]
            data['rooms'].append({
                "id": room_id,
                "receptacles": receptacle_counter_id_list
            })
        return data
    
    def get_scene_graph_introduction(self, mask_tiny_objects: bool=False):
        # "This is a scene graph of the house, which is a dict with two keys: rooms and receptacles.\n"
        # "Each room has a list of receptacles, it means the receptacles are in the room.\n"
        # "Each receptacle has two attributes: 'inReceptacle' and 'target_object', 'inReceptacle' means which objects we have found in this receptacle, 'target_object' means whether the target object is in this receptacle, if the value is 'Maybe here', it means we have not fully explored this receptacle, so we don't know whether the target object is in this receptacle. If the value is 'Not Found here', it means we have fully explored this receptacle, and we are sure that the target object is not in this receptacle.\n"
        if not mask_tiny_objects:
            return (
                f"The house consists of {len(self.room_counter_id_list)} rooms and some receptacles, which are as follows:\n"
                f"This is a scene graph of the house, which is a dict with two keys: rooms and receptacles.\n"
                f"Each room has a list of receptacles, it means the receptacles are in the room.\n"
                f"Each receptacle has two attributes: 'inReceptacle' and 'target_object', 'inReceptacle' means which objects we have found in this receptacle, 'target_object' means whether the target object is in this receptacle, if the value is 'Maybe here', it means we have not fully explored this receptacle, so we don't know whether the target object is in this receptacle. If the value is 'Not Found here', it means we have fully explored this receptacle, and we are sure that the target object is not in this receptacle.\n"
            )
        else:
            return (
                f"The house consists of {len(self.room_counter_id_list)} rooms and some receptacles, which are as follows:\n"
                f"This is a scene graph of the house, which is a list.\n"
                f"Each element in the list is a room metadata dict. The room dict has two attributes: 'id' and 'receptalces'. 'id' is the room id, and 'receptacles' is a list contains all the receptacles in the room.\n"
            )
        
    def get_scene_graph_prompt(self):
        introduction = self.get_scene_graph_introduction()
        data = self.get_room_receptacle_relations()
        for receptacle_idx in range(len(self.receptacle_counter_id_list)):
            receptacle_id = self.receptacle_counter_id_list[receptacle_idx]
            tiny_objects = self.get_tiny_object(receptacle_idx)
            explored = self.fully_explored_receptacles[receptacle_idx]
            if len(tiny_objects) == 0 and not explored:
                continue
            data['receptacles'].append({
                "id": receptacle_id,
                'inReceptacle': sorted(tiny_objects),
                "target_object": "Not Found here" if explored else "Maybe here"
            })
        return introduction + self.json2str(data)

    def get_position_in_scene_graph(self, position: dict):
        """
        args:
            position: dict {"x": x, "y": y, "z": z}
        return: tuple (room_counter_id, receptacle_counter_id)
        """
        min_distance = float("inf")
        receptacle_counter_id = None
        room_counter_id = self.get_room_counter_id_from_position(position)
        room_idx = self.room_counter_id_list.index(room_counter_id)
        for receptacle_idx in self.room2receptacles[room_idx]:
            receptacle_center = self.receptacle_centers[receptacle_idx]
            distance = (receptacle_center["x"] - position["x"]) ** 2 + (receptacle_center["y"] - position["y"]) ** 2 + (receptacle_center["z"] - position["z"]) ** 2
            if distance < min_distance:
                min_distance = distance
                receptacle_counter_id = self.receptacle_counter_id_list[receptacle_idx]
        return room_counter_id, receptacle_counter_id

    def get_receptacle_center_from_counter_id(self, receptacle_counter_id: str):
        receptacle_idx = self.receptacle_counter_id_list.index(receptacle_counter_id)
        return self.receptacle_centers[receptacle_idx]
    
    def get_receptacle_bbox3d_from_counter_id(self, receptacle_counter_id: str):
        receptacle_idx = self.receptacle_counter_id_list.index(receptacle_counter_id)
        return self.receptacle_bbox3d_list[receptacle_idx]
    
    def get_receptacle_counter_id_list(self):
        return self.receptacle_counter_id_list

    def add_tiny_object(self, receptacle_counter_id: str, tiny_object_scene_id: str):
        receptacle_idx = self.receptacle_counter_id_list.index(receptacle_counter_id)
        if tiny_object_scene_id not in self.receptacle2tinyobjects[receptacle_idx]:
            self.receptacle2tinyobjects[receptacle_idx].append(tiny_object_scene_id)
    
    def get_tiny_object(self, receptacle_index: int):
        return list(set([self.get_type_from_id(tiny_object_scene_id) for tiny_object_scene_id in self.receptacle2tinyobjects[receptacle_index]]))
    
    def set_explored_receptacle(self, receptacle_counter_id: str):
        receptacle_idx = self.receptacle_counter_id_list.index(receptacle_counter_id)
        self.fully_explored_receptacles[receptacle_idx] = True

    def ground_receptacle_counter_id_in_room(self, receptacle_counter_id: str, room_counter_id: str):
        """
        In fully scene graph, the receptacle counter id must in the room counter id and we can return it.
        else asser error 
        """
        # check the receptacle in room
        if not self.check_receptacle_sure_in_room(room_counter_id, receptacle_counter_id):
            return None
        return receptacle_counter_id

    def get_explored_receptacle_counter_id_list(self):
        return [receptacle_counter_id for receptacle_counter_id, fully_explored in zip(self.receptacle_counter_id_list, self.fully_explored_receptacles) if fully_explored]

    def get_unexplored_receptacle_counter_id_list(self):
        return [receptacle_counter_id for receptacle_counter_id, fully_explored in zip(self.receptacle_counter_id_list, self.fully_explored_receptacles) if not fully_explored]

    def update_receptacle_and_object(self, receptacles: list[tuple[str, dict]], objects: list[tuple[str, dict]], receptacle_contin_object_relation: list[tuple[int, int]]):
        known_object_idx_list = []
        for receptacle_idx, object_idx in receptacle_contin_object_relation:
            receptacle_scene_id = receptacles[receptacle_idx][0]
            receptacle_counter_id = self.get_counter_id_from_scene_id(receptacle_scene_id)
            object_scene_id = objects[object_idx][0]
            self.add_tiny_object(receptacle_counter_id, object_scene_id)
            known_object_idx_list.append(object_idx)
            
        for object_idx in range(len(objects)):
            if object_idx not in known_object_idx_list:
                object_scene_id = objects[object_idx][0]
                object_position = objects[object_idx][1]
                _, receptacle_counter_id = self.get_position_in_scene_graph(object_position)
                self.add_tiny_object(receptacle_counter_id, object_scene_id)            
            
    def get_receptacle_scene_id_from_receptacle_counter_id(self, receptacle_counter_id: str):
        return self.counter_id2scene_id[receptacle_counter_id]
    
    def ground_receptacle_counter_id(self, receptacle_counter_id: str) -> list[str]:
        if receptacle_counter_id not in self.receptacle_counter_id_list:
            return []
        else:
            return [receptacle_counter_id]
    
class Partly_Known_Semantic_Map(Semantic_Map):
    receptacle_type_set: set[str] = set(FIXED_RECEPTACLE_LIST)
    def __init__(self, metadata: MetadataWrapper, house: dict) -> None:
        super().__init__(metadata, house)
        self.reset()
        
    def reset(self):
        # set all the seen objects and receptacles
        self.seen_receptacle_scene_id_list = []
        self.receptacle_type_counter = Counter()
        self.seen_receptacle_counter_id_list = []
        self.seen_room2receptacles = defaultdict(list)
        self.seen_receptacle2room = {}
        self.full_explored_receptacle_counter_id_set = set()
        
        self.seen_object_scene_id_list = []
        self.object_type_counter = Counter()
        self.seen_object_counter_id_list = []
        self.seen_object2room = {}
        self.seen_object2receptacle = {}
        self.seen_receptacle2objects = defaultdict(list)
        # we add this because sometimes we don't know the receptacle that the object is in, so we need to add the object in the room
        self.seen_room2objects = defaultdict(list)
        
    def get_explored_receptacle_counter_id_list(self):
        return list(self.full_explored_receptacle_counter_id_set)
    
    def ground_receptacle_counter_id_in_room(self, receptacle_counter_id: str, target_room_counter_id: str):
        """
        In partly scene graph, the receptacle counter id must be belief and then we can ground it. and the receptacle id room must be equal as we know
        """
        # check the receptacle counter id a standard belief receptacle counter id
        if self.check_belief_counter_id(receptacle_counter_id):
            if not self.check_standard_belief_counter_id(receptacle_counter_id):
                return None
            source_room_counter_id = self.get_room_counter_id_from_receptacle_counter_id(receptacle_counter_id)
            if source_room_counter_id == target_room_counter_id:
                return receptacle_counter_id
            if not self.check_equal_room(source_room_counter_id, target_room_counter_id):
                return None
            receptacle_type = self.get_type_from_id(receptacle_counter_id)
            return self.get_belief_receptacle_counter_id(receptacle_type, target_room_counter_id)
        else:
            if not self.check_receptacle_sure_in_room(target_room_counter_id, receptacle_counter_id):
                return None
            return receptacle_counter_id
    
    def get_unexplored_receptacle_counter_id_list(self):
        return [receptacle_counter_id for receptacle_counter_id in self.seen_receptacle_counter_id_list if receptacle_counter_id not in self.full_explored_receptacle_counter_id_set]
    
    def create_seen_receptacle_counter_id(self, receptacle_type: str):
        self.receptacle_type_counter[receptacle_type] += 1
        return f"{receptacle_type}|{self.receptacle_type_counter[receptacle_type]}"
    
    def create_seen_object_counter_id(self, object_type: str):
        self.object_type_counter[object_type] += 1
        return f"{object_type}|{self.object_type_counter[object_type]}"
    
    def update_receptacle_in_room(self, receptacle_scene_id: str, room_counter_id: str):
        if receptacle_scene_id not in self.seen_receptacle_scene_id_list:
            self.seen_receptacle_scene_id_list.append(receptacle_scene_id)
            receptacle_type = self.get_type_from_id(receptacle_scene_id)
            receptacle_counter_id = self.create_seen_receptacle_counter_id(receptacle_type)
            self.seen_receptacle_counter_id_list.append(receptacle_counter_id)
            self.seen_room2receptacles[room_counter_id].append(receptacle_counter_id)
            self.seen_receptacle2room[receptacle_counter_id] = room_counter_id
            return receptacle_counter_id
        else:
            receptacle_counter_id = self.get_receptacle_counter_id_from_receptacle_scene_id(receptacle_scene_id)
            assert receptacle_counter_id in self.seen_room2receptacles[room_counter_id]
            return receptacle_counter_id
        
    def update_object_in_room(self, object_scene_id: str, room_counter_id: str) -> str:
        if object_scene_id not in self.seen_object_scene_id_list:
            self.seen_object_scene_id_list.append(object_scene_id)
            object_type = self.get_type_from_id(object_scene_id)
            object_counter_id = self.create_seen_object_counter_id(object_type)
            self.seen_object_counter_id_list.append(object_counter_id)
            self.seen_room2objects[room_counter_id].append(object_counter_id)
            self.seen_object2room[object_counter_id] = room_counter_id
            return object_counter_id
        else:
            object_counter_id = self.get_object_counter_id_from_object_scene_id(object_scene_id)
            assert object_counter_id in self.seen_room2objects[room_counter_id]
            return object_counter_id
        
    def update_object_in_receptacle_relation(self,receptacle_counter_id: str, object_counter_id: str):
        self.seen_receptacle2objects[receptacle_counter_id].append(object_counter_id)
        self.seen_object2receptacle[object_counter_id] = receptacle_counter_id
        
    def update_receptacle_and_object(self, receptacles: list[tuple[str, dict]], objects: list[tuple[str, dict]], relation: list[tuple[int, int]]):
        update_receptacle_counter_id_list = []
        for receptacle_scene_id, position in receptacles:
            # only if it is the fisrt time we see this receptacle, we add it to the explored_receptacle_scene_id_list and add the room receptacle relation
            room_counter_id = self.get_room_counter_id_from_position(position)
            receptacle_counter_id = self.update_receptacle_in_room(receptacle_scene_id, room_counter_id)
            update_receptacle_counter_id_list.append(receptacle_counter_id)

        update_object_counter_id_list = []
        for object_scene_id, position in objects:
            room_counter_id = self.get_room_counter_id_from_position(position)
            object_counter_id = self.update_object_in_room(object_scene_id, room_counter_id)
            update_object_counter_id_list.append(object_counter_id)
        
        for receptacle_index, object_index in relation:
            receptacle_counter_id = update_receptacle_counter_id_list[receptacle_index]
            object_counter_id = update_object_counter_id_list[object_index]
            self.update_object_in_receptacle_relation(receptacle_counter_id, object_counter_id)
            
    def get_receptacle_counter_id_list(self):
        return self.seen_receptacle_counter_id_list
    
    @staticmethod
    def get_belief_receptacle_counter_id(belief_receptacle_type: str, room_counter_id: str):
        return f"{belief_receptacle_type}|Unknown|{room_counter_id}"
    
    @staticmethod
    def get_belief_receptacle_type_and_room_counter_id(belief_receptacle_counter_id: str):
        belief_receptacle_type = belief_receptacle_counter_id.split("|")[0]
        # notice that there is '|' in room_counter_id, if you split the string with '|', you will split room_counter id too
        # so you need to change the way to get the room_counter_id, a string is like Shelf|Unknown|Kitchen|1, you need to extract the 'Kitchen|1' part
        room_counter_id = belief_receptacle_counter_id.split("|")[-2:]
        room_counter_id = "|".join(room_counter_id)
        return belief_receptacle_type, room_counter_id
    
    def get_belief_receptacles_in_room(self, room_counter_id: str):
        known_receptacle_counter_id_list = self.seen_room2receptacles[room_counter_id]
        known_receptacle_type_set = set([self.get_type_from_id(receptacle_counter_id) for receptacle_counter_id in known_receptacle_counter_id_list])
        belief_receptacle_type_set = self.receptacle_type_set - known_receptacle_type_set
        belief_receptacle_counter_id_list = []
        for belief_receptacle_type in belief_receptacle_type_set:
            belief_receptacle_counter_id = self.get_belief_receptacle_counter_id(belief_receptacle_type, room_counter_id)
            if belief_receptacle_counter_id not in self.full_explored_receptacle_counter_id_set:
                belief_receptacle_counter_id_list.append(belief_receptacle_counter_id)
        return belief_receptacle_counter_id_list
    
    def set_explored_receptacle(self, receptacle_counter_id: str):
        self.full_explored_receptacle_counter_id_set.add(receptacle_counter_id)
    
    def check_belief_counter_id(self, counter_id: str):
        return counter_id.count("|") > 1

    def get_room_counter_id_from_object_type_in_it(self, object_type: str):
        for room_counter_id in self.room_counter_id_list:
            tiny_objects = self.seen_room2objects[room_counter_id]
            for tiny_object in tiny_objects:
                if object_type == self.get_type_from_id(tiny_object):
                    return room_counter_id
        return None
                
    def get_receptacle_counter_id_from_object_type_in_it(self, object_type: str):
        # travel each receptacle to check the tiny objects in it
        for receptacle_counter_id in self.seen_receptacle_counter_id_list:
            tiny_objects = self.seen_receptacle2objects[receptacle_counter_id]
            for tiny_object in tiny_objects:
                if object_type == self.get_type_from_id(tiny_object):
                    return receptacle_counter_id
        return None
    
    def check_receptacle_maybe_in_room(self, room_counter_id: str, receptacle_counter_id: str) -> bool:
        # room_idx = self.room_counter_id_list.index(room_counter_id)
        # receptacle_idx_list = [receptacle_idx for room_idx, receptacle_idx in self.expolred_room_receptacle_relations if room_idx == room_idx]
        # return [self.receptacle_counter_id_list[receptacle_idx] for receptacle_idx in receptacle_idx_list]
        if self.check_belief_counter_id(receptacle_counter_id):
            _, belief_room_counter_id = self.get_belief_receptacle_type_and_room_counter_id(receptacle_counter_id)
            if belief_room_counter_id != room_counter_id:
                return False
            return self.check_standard_belief_counter_id(receptacle_counter_id)
        else:
            return receptacle_counter_id in self.seen_room2receptacles[room_counter_id]
    
    def check_receptacle_sure_in_room(self, room_counter_id: str, receptacle_counter_id: str):
        if room_counter_id not in self.room_counter_id_list:
            return False
        return receptacle_counter_id in self.seen_room2receptacles[room_counter_id]
    
    def check_receptacle_maybe_in_house(self, receptacle_counter_id: str):
        if self.check_belief_counter_id(receptacle_counter_id):
            return self.check_standard_belief_counter_id(receptacle_counter_id)
        else:
            return receptacle_counter_id in self.seen_receptacle_counter_id_list
    
    def check_receptacle_sure_in_house(self, receptacle_counter_id: str):
        return receptacle_counter_id in self.seen_receptacle_counter_id_list
    
    def get_receptacle_scene_id_from_receptacle_counter_id(self, receptacle_counter_id: str):
        receptacle_idx = self.seen_receptacle_counter_id_list.index(receptacle_counter_id)
        return self.seen_receptacle_scene_id_list[receptacle_idx]
    
    def get_receptacle_counter_id_from_receptacle_scene_id(self, receptacle_scene_id: str):
        receptacle_idx = self.seen_receptacle_scene_id_list.index(receptacle_scene_id)
        return self.seen_receptacle_counter_id_list[receptacle_idx]
    
    def get_object_scene_id_from_object_counter_id(self, object_counter_id: str):
        object_idx = self.seen_object_counter_id_list.index(object_counter_id)
        return self.seen_object_scene_id_list[object_idx]
    
    def get_object_counter_id_from_object_scene_id(self, object_scene_id: str):
        object_idx = self.seen_object_scene_id_list.index(object_scene_id)
        return self.seen_object_counter_id_list[object_idx]
    
    def get_room_counter_id_from_receptacle_counter_id(self, receptacle_counter_id: str):
        if self.check_belief_counter_id(receptacle_counter_id):
            return self.get_belief_receptacle_type_and_room_counter_id(receptacle_counter_id)[1]
        elif receptacle_counter_id in self.seen_receptacle2room:
            return self.seen_receptacle2room[receptacle_counter_id]
        raise ValueError(f"receptacle_counter_id {receptacle_counter_id} is not in any room")

    def get_object_in_room_not_in_receptacle(self, room_counter_id: str):
        object_in_room = set(self.seen_room2objects[room_counter_id])
        object_in_receptacle = set(self.seen_object2receptacle.keys())
        return list(object_in_room - object_in_receptacle)

    def check_standard_belief_counter_id(self, receptacle_counter_id: str) -> bool:
        if not self.check_belief_counter_id(receptacle_counter_id):
            return False
        belief_receptacle_type, room_counter_id = self.get_belief_receptacle_type_and_room_counter_id(receptacle_counter_id)
        if belief_receptacle_type not in self.receptacle_type_set or room_counter_id not in self.room_counter_id_list:
            return False
        standard_receptacle_counter_id = self.get_belief_receptacle_counter_id(belief_receptacle_type, room_counter_id)
        if standard_receptacle_counter_id != receptacle_counter_id:
            return False
        return standard_receptacle_counter_id in self.get_belief_receptacles_in_room(room_counter_id)
        
    def ground_receptacle_counter_id(self, receptacle_counter_id: str) -> list[str]:
        """
        if the receptacle counter id is belief id, then we check the seen receptacles in the room, if there is the same type seen receptacle, we return the seen receptacle, else we return the belief receptacle
        """
        if self.check_belief_counter_id(receptacle_counter_id):
            # make sure the receptacle_counter_id is a right format
            belief_receptacle_type, room_counter_id = self.get_belief_receptacle_type_and_room_counter_id(receptacle_counter_id)
            if belief_receptacle_type not in self.receptacle_type_set or room_counter_id not in self.room_counter_id_list:
                return []
            standard_receptacle_counter_id = self.get_belief_receptacle_counter_id(belief_receptacle_type, room_counter_id)
            if standard_receptacle_counter_id != receptacle_counter_id:
                return []
            if standard_receptacle_counter_id in self.full_explored_receptacle_counter_id_set:
                return []
            # we check that if there is the same type seen receptacle in the room.
            seen_receptacles = self.seen_room2receptacles[room_counter_id]
            seen_receptacles_with_same_type = []
            for seen_receptacle_counter_id in seen_receptacles:
                if self.get_type_from_id(seen_receptacle_counter_id) == belief_receptacle_type:
                    seen_receptacles_with_same_type.append(seen_receptacle_counter_id)
            if len(seen_receptacles_with_same_type) != 0:
                return seen_receptacles_with_same_type
            else:
                return [receptacle_counter_id]
        else:
            # if the receptacle_counter_id is a seen receptacle in room, then just return it, else return []
            if receptacle_counter_id in self.seen_receptacle_counter_id_list:
                return [receptacle_counter_id]
            else:
                return []
    
    def check_equal_room(self, room_counter_id1: str, room_counter_id2: str) -> bool:
        """
        check if the two rooms can be merged(only if the same room type, the seen receptacle type is empty and the tiny objects are empty)
        """
        room_type1 = self.get_type_from_id(room_counter_id1)
        room_type2 = self.get_type_from_id(room_counter_id2)
        if room_type1 != room_type2:
            return False
        if len(self.seen_room2receptacles[room_counter_id1]) > 0 or len(self.seen_room2receptacles[room_counter_id2]) > 0:
            return False
        if len(self.seen_room2objects[room_counter_id1]) > 0 or len(self.seen_room2objects[room_counter_id2]) > 0:
            return False
        return True
    
    def get_room_split_list(self) -> list[list[str]]:
        room_split_list = []
        merge_room_set = set()
        for room_counter_id1 in self.room_counter_id_list:
            if room_counter_id1 in merge_room_set:
                continue
            merge_room = [room_counter_id1]
            merge_room_set.add(room_counter_id1)
            for room_counter_id2 in self.room_counter_id_list:
                if room_counter_id2 in merge_room_set:
                    continue
                if self.check_equal_room(room_counter_id1, room_counter_id2):
                    merge_room.append(room_counter_id2)
                    merge_room_set.add(room_counter_id2)
            room_split_list.append(merge_room)
        return room_split_list
    
    def get_room_counter_id2merge_room_counter_id(self) -> dict[str, str]:
        room_split_list = self.get_room_split_list()
        first_room_counter_id_list = self.get_merge_room_counter_id_list()
        merge_room2split = {}
        for room_split, first_room_counter_id in zip(room_split_list, first_room_counter_id_list):
            for room_counter_id in room_split:
                merge_room2split[room_counter_id] = first_room_counter_id
        return merge_room2split
    
    def get_merge_room_counter_id_list(self) -> list[str]:
        room_split_list = self.get_room_split_list()
        merge_room_counter_id = []
        for room_split in room_split_list:
            split_idx_list = [self.get_counter_idx_from_id(room_counter_id) for room_counter_id in room_split]
            # get room_counter_id with min idx
            min_idx = min(split_idx_list)
            min_room_counter_id = room_split[split_idx_list.index(min_idx)]
            merge_room_counter_id.append(min_room_counter_id)
        return merge_room_counter_id
    
    def get_merge_room_counter_id2split(self) -> dict[str, list[str]]:
        room_split_list = self.get_room_split_list()
        merge_room_counter_id_list = self.get_merge_room_counter_id_list()
        return {
            merge_room_counter_id: room_split
            for merge_room_counter_id, room_split in zip(merge_room_counter_id_list, room_split_list)
        }        
    
    def get_room_counter_id2split_dict(self) -> dict[str, list[str]]:
        room_split_list = self.get_room_split_list()
        room_counter_id2split = {}
        for room_split in room_split_list:
            for room_counter_id in room_split:
                room_counter_id2split[room_counter_id] = room_split
        return room_counter_id2split    
    
    def get_scene_graph_prompt(self):
        return self.get_scene_graph_prompt_with_belief_list()
    
    def get_scene_graph_prompt_with_belief_list(self, belief_list: list[str]=None):
        introduction = (
            f"The house consists of {len(self.room_counter_id_list)} rooms and some receptacles, which are as follows:\n"
            f"This is a scene graph of the house, which is a dict with two keys: rooms and receptacles.\n"
            f"The 'rooms' key is a list of room metadata dict. The room dict has four attributes: 'id', 'receptacles', 'belief_receptacles' and 'objects'.\n"
            f"'id' is the room id, 'receptacles' is a list contains all the receptacles in the room as we know, 'belief_receptacles' is a list contains all the receptacles that may be in the room, 'objects' is a list contains the objects in the room but not in any receptacle.\n"
            f"The 'receptacles' key is a list of receptacle metadata dict. The receptacle dict has two attributes: 'id' and 'objects'.\n"
            f"'id' is the receptacle id, 'objects' is a list contains all the objects in the receptacle as we know.\n"
        )
        data = {
            "rooms": [],
            "receptacles": [],
        }
        merge_room_counter_ids = self.get_merge_room_counter_id_list()
        # keep sorted and we can get the same scene graph of the same house to best use the llm cache
        merge_room_counter_ids = sorted(merge_room_counter_ids)
        for room_counter_id in merge_room_counter_ids:
            seen_receptacles = self.seen_room2receptacles[room_counter_id]
            seen_objects = self.seen_room2objects[room_counter_id]
            belief_receptacles = self.get_belief_receptacles_in_room(room_counter_id)
            if belief_list is not None:
                belief_receptacles = [belief_receptacle for belief_receptacle in belief_receptacles if belief_receptacle in belief_list]
            seen_receptacles = sorted(seen_receptacles)
            seen_objects = sorted(seen_objects)
            belief_receptacles = sorted(belief_receptacles)
            data['rooms'].append({
                "id": room_counter_id,
                "receptacles": seen_receptacles,
                "belief_receptacles": belief_receptacles,
                "objects": seen_objects
            })
            
        for receptacle_counter_id in self.seen_receptacle_counter_id_list:
            if self.check_belief_counter_id(receptacle_counter_id):
                continue
            seen_objects = self.seen_receptacle2objects[receptacle_counter_id]
            seen_objects = sorted(seen_objects)
            data['receptacles'].append({
                "id": receptacle_counter_id,
                "objects": seen_objects,
                "target_object": "Not Found here" if receptacle_counter_id in self.full_explored_receptacle_counter_id_set else "Maybe here"
            })
        return introduction + self.json2str(data)
