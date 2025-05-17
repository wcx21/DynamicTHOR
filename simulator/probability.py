from collections import Counter
import pandas as pd
import uuid
from simulator.scene_graph import Semantic_Map, Fully_Known_Semantic_Map, Partly_Known_Semantic_Map

class Probablity:
    def __init__(self) -> None:
        """
        memry: dict, 
            key is the uuid, 
            value is
                a dict, key is the room_id, value is list with 3 element, [float, float, dict], 
                the first float is the probability that this event happend in the room, 
                the second float is the probability that this event happend in the room and the person move the object to the room
                the third element is a dict, key is the receptacle_id, value is the probability that the person move the object to this (room_id, receptacle_id)
            if the key is the none key, the value is a dict:
                key is the receptacle_id, value is the probability that the target object in the receptacle
        """
        self.none_key = str(uuid.uuid4())
        self.memory: dict[str, dict] = {
            self.none_key: {}
        }
    
    def insert_data(self, data: dict=None) -> str:
        if data is None:
            data = {}
        key = str(uuid.uuid4())
        self.memory[key] = data
        return key
    
    def get_data(self, key: str) -> dict:
        return self.memory[key]
    
    def check_room_in_data(self, room_id: str, key: str) -> bool:
        return room_id in self.get_data(key)
    
    def set_oracle_room(self, room_id: str, key: str, move_to_room_probability: float|None=None):
        data = {
            room_id: [1.0, move_to_room_probability, {}]
        }
        if self.check_room_in_data(room_id, key):
            old_data = self.memory[key][room_id]
            if move_to_room_probability is None:
                data[room_id][1] = old_data[1]
            data[room_id][2] = old_data[2]
        self.memory[key] = data
        
    def set_oracle_move_to_room_probability(self, key: str, move_to_room_prob: float):
        for room_id in self.get_data(key):
            self.memory[key][room_id][1] = move_to_room_prob
            
    def set_room_analysis(self, key: str, room_analysis: dict[str, float], move_to_room_prob: float|None=None):
        """
        args:
            room_analysis: the key is the room_id, and the value is the probability that the event happend in the room
        """
        data = {
            room_id: [room_analysis[room_id], move_to_room_prob, {}]
            for room_id in room_analysis
        }
        for room_id in data:
            if self.check_room_in_data(room_id, key):
                old_data = self.memory[key][room_id]
                data[room_id][1] = move_to_room_prob or old_data[1]
                data[room_id][2] = old_data[2]
        self.memory[key] = data
    
    def set_event_analysis(self, key: str, room_id: str, move_to_room_probability: float) -> bool:
        if self.check_room_in_data(room_id, key):
            self.memory[key][room_id][1] = move_to_room_probability
            return True
        return False
    
    def set_receptacle_analysis(self, key: str, room_id: str, receptacles: dict) -> bool:
        if self.check_room_in_data(room_id, key):
            self.memory[key][room_id][2] = receptacles
            return True
        return False
    
    def get_candidate_rooms(self, key: str) -> list[str]:
        return list(self.get_data(key).keys())
    
    def set_none_value(self, value: dict):
        self.memory[self.none_key] = value
    
    def get_none_value(self) -> dict:
        # normalize
        none_probability = sum(self.memory[self.none_key].values())
        if none_probability == 0:
            return {}
        else:
            for receptacle_id in self.memory[self.none_key]:
                self.memory[self.none_key][receptacle_id] /= none_probability
        return self.memory[self.none_key]
    
    def get_probability(self, key: str) -> tuple[float, dict]:
        """
        return:
            move probability: float, the probability that the event moved the object.
            receptacle probability: dict, the probability that the event moved the object to the receptacle.
        """
        data = self.get_data(key)
        # happend in each room probability need to normalize
        happend_in_room_probability = sum([data[room_id][0] for room_id in data])
        for room_id in data:
            data[room_id][0] /= happend_in_room_probability
        move_probability = 0.0
        for room_id in data:
            move_probability += data[room_id][0] * (data[room_id][1] or 0.0)
        receptacle_probability = {}
        for room_id in data:
            move_to_room = data[room_id][0] * (data[room_id][1] or 0.0)
            # take to each receptacle probability need to normalize
            object_in_receptacle_probability = sum([data[room_id][2][receptacle_id] for receptacle_id in data[room_id][2]])
            for receptacle_id in data[room_id][2]:
                data[room_id][2][receptacle_id] /= object_in_receptacle_probability
            for receptacle_id in data[room_id][2]:
                receptacle_probability[receptacle_id] = move_to_room * data[room_id][2][receptacle_id]
        return move_probability, receptacle_probability

    def update_scene_graph(self, scene_graph: Semantic_Map):
        if isinstance(scene_graph, Fully_Known_Semantic_Map):
            # because the receptacle in Fully_known_Semantic_Map is not change, so we don't need to update the memory
            pass
        elif isinstance(scene_graph, Partly_Known_Semantic_Map):
            self.update_partly_scene_graph(scene_graph)
        else:
            raise TypeError(f"scene_graph should be Fully_Known_Semantic_Map or Partly_Known_Semantic_Map, not {type(scene_graph)}")
    
    def update_receptacle_probability(self, scene_graph: Partly_Known_Semantic_Map, receptacles: dict[str, float]) -> dict:
        new_receptacles = Counter()
        for receptaclce_id, probability in receptacles.items():
            new_receptacle_id_list = scene_graph.ground_receptacle_counter_id(receptaclce_id)
            if len(new_receptacle_id_list) == 0:
                continue
            else:
                receptacle_num = len(new_receptacle_id_list)
                for new_rec_id in new_receptacle_id_list:
                    new_receptacles[new_rec_id] += probability / receptacle_num
        # normalize
        receptacle_probability = sum(new_receptacles.values())
        if receptacle_probability == 0:
            return {}
        else:
            for receptacle_id in new_receptacles:
                new_receptacles[receptacle_id] /= receptacle_probability
            return dict(new_receptacles)
    
    def update_partly_scene_graph(self, scene_graph: Partly_Known_Semantic_Map):
        for key in self.memory:
            if key == self.none_key:
                continue
            data = self.memory[key]
            for room_id in data:
                data[room_id][2] = self.update_receptacle_probability(scene_graph, data[room_id][2])
        self.memory[self.none_key] = self.update_receptacle_probability(scene_graph, self.memory[self.none_key])
        