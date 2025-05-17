import pickle
import json
import os
import numpy as np
import random
import copy
import datetime

from simulator.constants import PRIVATE_OBJECT_LIST,COMMONLY_USED_OBJECT_LIST
from simulator.task import Task

#sample task: time + object

def get_obj_type(obj_id):
    return obj_id.split('|')[0]

def get_obj_belong_type(obj_id):
    obj_type = get_obj_type(obj_id)
    belong_type = "public"
    if obj_type in PRIVATE_OBJECT_LIST:
        belong_type = "private"
    elif obj_type in COMMONLY_USED_OBJECT_LIST:
        belong_type = "part_public"
    return belong_type

filter_type = [
                "AlarmClock",
                "Apple",
                "BaseballBat",
                "BasketBall",
                "Bowl",
                "Book"
                "CellPhone",
                "Cloth",
                "CreditCard",
                'Cup',
                'DishSponge',
                'Fork',
                'KeyChain',
                "Laptop",
                'Knife',
                "Mug",
                "Newspaper",
                'Pen', 
                'Pencil',
                'Pillow',
                'Plate',
                'Spoon', 
                'SprayBottle',
                'Vase', 
                'Watch',
            ]

class Sampler():
    def __init__(self):
        self.schedules = []
        self.activity_base = {}
        self.character = []
        self.character_specific = {}
        self.scene_layout = {}

    def clear_character_specific(self):
        self.character_specific = {}
        for c in self.character:
            self.character_specific[c] = {}
            self.character_specific[c]['room'] = {}
            self.character_specific[c]['related_object'] = []


    def set_activity(self, character_info, activity_base, schedules):
        self.activity_base = activity_base
        self.character_info = character_info
        self.character = [c['name'] for c in self.character_info]
        self.schedules = schedules
        self.clear_character_specific()
        
            
  
    def set_scene_layout(self, scene_layout):
        self.scene_layout = scene_layout
        self.clear_character_specific()



    def get_character_group(self):
        #给人物分组分配房间，例如夫妻,看接口定
        result = [[c] for c in self.character]
        return result
        

    def object_allocation(self):
        #room 
        rooms = self.scene_layout['room']
        bedrooms = [room for room in rooms if room['roomType'] == 'Bedroom']
        bathrooms = [room for room in rooms if room['roomType'] == 'Bathroom']
        character_group = self.get_character_group()
        for cg in character_group:
            bedroom_stack = copy.deepcopy(bedrooms)
            bathroom_stack = copy.deepcopy(bathrooms)
            if len(bedrooms) > 0:
                cd_bedroom = bedroom_stack[-1]
                bedroom_stack.pop(-1)
            else:
                cd_bedroom = bedrooms[0]
            if len(bathrooms) > 0:
                cd_bathroom = bathroom_stack[-1]
                bathroom_stack.pop(-1)
            else:
                cd_bathroom = bathrooms[0]

            for c in cg:
                self.character_specific[c]['room'] = {
                    'Bedroom':cd_bedroom['roomId'],
                    'Bathroom':cd_bathroom['roomId']
                }
        
        #object
        pickup_objects = self.scene_layout['pickupable object']
        for pobj in pickup_objects:
            pobj_type = get_obj_type(pobj)
            #private
            if pobj_type in PRIVATE_OBJECT_LIST:
                for character in self.character_specific:
                    private_objects = self.character_specific[character]['related_object']
                    exist = False
                    for o in private_objects:
                        if get_obj_type(o) == pobj_type:
                            exist = True
                            break
                    if not exist:
                        self.character_specific[character]['related_object'].append(pobj)

            #common used (not all characters)
            elif pobj_type in COMMONLY_USED_OBJECT_LIST:
                if len(self.character) >= 2:
                    selected_characters = random.sample(self.character, 2)
                    for sc in selected_characters:
                        self.character_specific[sc]['related_object'].append(pobj)
                

    def sample_schedules(self, id=0):
        
        character = self.character[id]
        schedules = self.schedules[id]
        activity_base = self.activity_base[id]
        results = {}
        for day in schedules:
            day_schedule = []
            for activity in schedules[day]:
                
                
                activity_name = activity['activity']
                #character = activity['character']
                if activity_name not in activity_base:
                    continue
                activity_content = activity_base[activity_name]

                #sample room
                rooms = list(activity_content.keys())
                room_probs = np.array([activity_content[room]['room_prob'] for room in activity_content])
                if not room_probs.sum() == 1:
                    room_probs = room_probs / room_probs.sum()

                choose_ids = [i for i in range(len(rooms))]
                choose_id = np.random.choice(choose_ids, p=room_probs.ravel())
                room = rooms[choose_id]
       
                #room ground to scene
                final_rooms = []
                if (room == 'Bedroom' or room == 'Bathroom'):
                    
                    final_room_id = self.character_specific[character]['room'][room]
                    for r in self.scene_layout['room']:
                        if r['roomId'] == final_room_id:
                            final_room = r
                            break
                else:
                    for scene_room in self.scene_layout['room']:
                        if scene_room['roomType'] == room:
                            final_rooms.append(scene_room)
                    
                    final_room = final_rooms[0]
         
                #sample object
                object_effect = activity_content[room]['object_effect']
                sample_objects = []

                
                for obj in object_effect:
                    object_prob = float(object_effect[obj]['object_prob'].strip(')'))
                    random_num = random.random()
                    if random_num < object_prob:
                        sample_objects.append(obj)

                
                #sample receptacle
                sample_receptacles = []
                for obj in sample_objects:
                    #除去房间中没有的receptacle，并重新归一化
                    possible_receptacles_init = object_effect[obj]['receptacles']
                    possible_receptacles = []
                    for pr in possible_receptacles_init:
                        for child in final_room['children']:
                            if pr[0] == child.split('|')[0]:
                                possible_receptacles.append(pr)
                                break
                    if len(possible_receptacles) == 0:
                        continue
                    prob_sum = 0
                    for pr in possible_receptacles:
                        prob_sum += pr[1]
                    for i in range(len(possible_receptacles)):
                        possible_receptacles[i][1] = possible_receptacles[i][1] / prob_sum
                    
                    receptacle_probs = np.array([recep[1] for recep in possible_receptacles])
                    choose_ids = [i for i in range(len(receptacle_probs))]
                    choose_id = np.random.choice(choose_ids, p=receptacle_probs.ravel())
                    receptacle = possible_receptacles[choose_id][0]
                    sample_receptacles.append([obj, receptacle])
                
                final_sample = []
                for type_sample in sample_receptacles:
                    obj_type = type_sample[0]
                    receptacle_type = type_sample[1]
                    final_obj = ""
                    final_recep = ""

                    related_objs = self.character_specific[character]['related_object']
                    if obj_type in PRIVATE_OBJECT_LIST:
                        
                        for ro in related_objs:
                            if get_obj_type(ro) == obj_type:
                                final_obj = ro
                                break
                    
                    else:
                        exist = False
                        for ro in related_objs:
                            if get_obj_type(ro) == obj_type:
                                exist = True
                                final_obj = ro
                                break
                        if not exist:
                            #从全部找
                            for pick_obj in self.scene_layout['pickupable object']:
                                if pick_obj.split("|")[0] == obj_type:
                                    final_obj = pick_obj
                                    break
                    
                    for recep in final_room['children']:
                        if recep.split('|')[0] == receptacle_type:
                            final_recep = recep
                            break
                    
                    if len(final_obj)>0 and len(final_recep) > 0:
                        final_sample.append({
                            'object':final_obj,
                            'receptacle':final_recep
                        })
                
                day_schedule.append({
                    'activity':activity_name,
                    'character':character,
                    'start_time':activity['start_time'],
                    'end_time':activity['end_time'],
                    'room':final_room['roomId'],
                    'content':final_sample,
                })
                    
            results[day] = day_schedule      

        return results     
                    

    def sample_task(self):
        #time
        candidate_time = [9,15,21]
        candidate_day = list(self.schedules[0].keys())[3:]
        select_day = random.sample(candidate_day, 1)[0]
        select_time = random.sample(candidate_time, 1)[0]
        task_time = select_day.replace(hour=select_time)

        #target
        select_character = random.sample(self.character, 1)[0]
        related_object = self.character_specific[select_character]['related_object']
        public_object = [obj for obj in self.scene_layout['pickupable object'] if get_obj_type(obj) not in PRIVATE_OBJECT_LIST and get_obj_type(obj) not in COMMONLY_USED_OBJECT_LIST]
        candidate_object = []
        candidate_object.extend(related_object)
        candidate_object.extend(public_object)
        final_candidate_obj = []
        for obj in candidate_object:
            if get_obj_type(obj) not in filter_type:
                continue
            else:
                final_candidate_obj.append(obj)
        
        target_object = random.sample(final_candidate_obj, 1)[0]


        #task
        person = self.character_info
        belong = []
        belong_type = get_obj_belong_type(target_object)
        if belong_type == 'public':
            for c in self.character:
                belong.append(c)
        elif belong_type == "part_public":
            for c in self.character_specific:
                related_objects = self.character_specific[c]['related_object']
                if target_object in related_objects:
                    belong.append(c)
        else:
            belong.append(select_character)

        #time = datetime.datetime.strptime("{} {}".format(select_day, select_time), "%Y-%m-%d %H:%M:%S")
        #task = Task(person, target_object, belong, time, start_location)
        
        task = {
            "time":task_time,
            'target_object':target_object,
            'person':select_character,
            'belong':belong
        }
        
        return task
        

    def sample(self):
        self.object_allocation()
        sample_shedules = {}
        keys = list(self.schedules[0].keys())
        for key in keys:
            sample_shedules[key] = []
        for i in range(len(self.character)):
            schedule = self.sample_schedules(i)
            for key in keys:
                sample_shedules[key].extend(schedule[key])
        
        def compare_time(item):
            return item['start_time']
        
        for key in keys:
            sample_shedules[key] = sorted(sample_shedules[key], key=compare_time)
        

        task = self.sample_task()
        return sample_shedules, task
    















