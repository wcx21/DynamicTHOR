from dataclasses import dataclass, field
from datetime import datetime
from models.llm.gpt import GPT_Model
import json
from simulator.depth_utils import Point
from simulator.scene_graph import Semantic_Map
from simulator.constants import ACTIVITY_LIST, ACTIVITY_DESCRIPTION_LIST
import random
import torch
import numpy as np


class Character:
    def __init__(self, json_data: dict):
        self.name = json_data.get('name', '')
        self.age = json_data.get('age', '')
        self.gender = json_data.get('gender', 'Female')  # Assuming default value is 'Female'
        self.family = json_data.get('family', '')
        self.profession = json_data.get('profession', '')
        self.json_data = json_data
        self.person_description = self.describe()

    def describe(self, gpt_agent: GPT_Model=None):
        # Use the correct pronoun for the gender
        gpt_agent = gpt_agent or GPT_Model()
        return gpt_agent.user_query(
            f"{json.dumps(self.json_data, indent=4)}\nPlease use the information provided to generate a self-introduction for this person. Don't give any extra infomation, just give the self-introduction."
        )

@dataclass
class Task:
    persons: list[Character]
    belongs: list[str] # changable
    target_object: str # changable
    current_time: datetime # changable
    semantic_map: Semantic_Map # changable
    target_object_num: int
    object_poses: dict[str, Point] = field(default_factory=dict)
    
    @property
    def object_poses_file_name(self):
        return f"{self.current_time.strftime('%Y-%m-%d_%H-%M-%S')}_object_poses.pkl"
    
    def set_seed(self):
        # we use timestamp to set the seed
        # TODO if you want to reproduce the result, you can set the seed here
        timestamp = int(self.current_time.timestamp())
        random.seed(timestamp)
        np.random.seed(timestamp)
        torch.manual_seed(timestamp)