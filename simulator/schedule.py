import copy
import json
from pathlib import Path
import pickle
from typing import Union
from simulator.scene_graph import Semantic_Map
from simulator.utils import SortedDictDefaultList
from datetime import datetime, timedelta, time
import pandas as pd
from simulator.constants import ACTIVITY_LIST, ACTIVITY_DESCRIPTION_LIST
from simulator.utils import translate_participle

class Schedule_Table:
    def __init__(self, scene_graph: Semantic_Map, person_name: str=None, path: Union[str, Path]="data/data_version/data_0/schedules.pkl"):
        # TODO Notice that now we just use the start time as the time of the activity in the schedule to be executed 
        self.daily_schedules = pd.DataFrame(columns=['person_name', 'date', 'start_time', 'end_time', 'activity_name', 'room', 'tiny_objects'])
        self.dynamic_info_schedule = SortedDictDefaultList()
        self.scene_graph = scene_graph
        # only keep the first name
        self.person_name = person_name
        self.reset_schedule(scene_graph, path)
    
    @staticmethod
    def get_time_from_str(date_str: str=None, time_str: str=None, datetime_str: str=None):
        """
        args:
            date_str: str, like "2023-02-13"
            time_str: str, like "06:00"
        """
        assert (date_str is not None and time_str is not None) or datetime_str is not None
        datetime_str = datetime_str or (date_str + " " + time_str)
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")

    def get_start_datetime(self) -> datetime:
        return self.daily_schedules.iloc[0]['date']
    
    def get_end_datetime(self) -> datetime:
        # the last day is the schedules of the last day + 1day
        return self.daily_schedules.iloc[-1]['date'] + timedelta(days=1)
    
    @staticmethod
    def datetime_obj2str(datetime_obj: datetime):
        return datetime_obj.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def datetime_str2obj(datetime_str: str):
        return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")

    @staticmethod
    def date_obj2str(date_obj: datetime) -> str:
        return date_obj.strftime("%Y-%m-%d")
    
    @staticmethod
    def date_str2obj(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%d")
    
    @staticmethod
    def timedelta_obj2str(timedelta_obj: timedelta) -> str:
        """
        return string like 06:00, only contains hour and minute
        """
        timestamp = pd.Timestamp('1970-01-01') + timedelta_obj
        time_str = timestamp.strftime('%H:%M')
        return time_str

    @staticmethod
    def timedelta_str2obj(timedelta_str: str) -> timedelta:
        """
        args:
            timedelta_str: string like 06:00, only contains hour and minute
        """
        datetime_obj = datetime.strptime(timedelta_str, '%H:%M')
        timedelta_obj = datetime_obj - datetime.strptime('00:00', '%H:%M')
        return timedelta_obj
    
    @property
    def current_date(self) -> datetime:
        # mask the time in "00:00"
        return self.current_time.replace(hour=0, minute=0, second=0, microsecond=0)

    def clear_schedule(self):
        self.dynamic_info_schedule.clear()
        self.daily_schedules.drop(self.daily_schedules.index, inplace=True)

    def reset_schedule(self, scene_graph, schedules_or_path):
        self.clear_schedule()
        self.add_schedules(schedules_or_path)
        self.reset_current_time()
        self.scene_graph = scene_graph

    def reset_current_time(self):
        self.current_time = self.get_start_datetime()
        
    # @staticmethod
    # def transfer_activity_description(activity_name: str):
    #     # we need to transfer the string like 'watch tv' -> 'watching tv'
    #     action = activity_name.split(" ")[0]
    #     if action.endswith("ing"):
    #         return activity_name
    #     else:
    #         return activity_name.replace(action, action + "ing")
       
    # TODO this function is right, but we have use the above function to generate some data
    # to not waste these data, we use the above function for now, but after the data update, we need to use this function
    @staticmethod
    def transfer_activity_description(activity_name: str):
        # we need to transfer the string like 'watch tv' -> 'watching tv'
        action = activity_name.split(" ")[0]
        # new_action = conjugate(action, tense=PARTICIPLE, parse=True)
        new_action = translate_participle(action)
        return activity_name.replace(action, new_action)
        
    def add_schedules(self, schedules_or_path):
        """
       {
            'activity': 'doing morning exercise',
            'character': 'Michael Thompson',
            'start_time': datetime.time(5, 30),
            'end_time': datetime.time(5, 50),
            'room': 'room|3',
            'content': [{'object': 'Laptop|surface|5|44', 'receptacle': 'Sofa|3|3'}]
        }
        """
        if isinstance(schedules_or_path, str):
            with open(schedules_or_path, 'rb') as file:
                schedules = pickle.load(file)
        elif isinstance(schedules_or_path, dict):
            schedules = schedules_or_path
        else:
            raise Exception("schedules_or_path must be str or list")
        
        # transfer the json to pd.DataFrame
        for date_obj, activities in schedules.items():
            for activity in activities:
                if self.person_name is not None and activity['character'] != self.person_name:
                    continue
                person_name = self.person_name or activity['character']
                start_time: time = activity['start_time']
                end_time: time = activity['end_time']
                start_time_delta = timedelta(hours=start_time.hour, minutes=start_time.minute)
                end_time_delta = timedelta(hours=end_time.hour, minutes=end_time.minute)
                activity_name = activity['activity']
                activity_name = self.transfer_activity_description(activity_name)
                action_list = activity['content']
                room_scene_id = activity['room']
                room_counter_id = self.scene_graph.get_room_counter_id_from_room_scene_id(room_scene_id)
                object_type_list = [self.scene_graph.get_type_from_id(action['object']) for action in action_list]
                tiny_objects = ",".join(object_type_list)
                self.daily_schedules.loc[len(self.daily_schedules)] = [person_name, date_obj, start_time_delta, end_time_delta, activity_name, room_counter_id, tiny_objects]
        
        for date_obj, date_activity_list in schedules.items():
            for activity in date_activity_list:
                if self.person_name is not None and activity['character'] != self.person_name:
                    continue
                start_time = activity['start_time']
                start_time_delta = timedelta(hours=start_time.hour, minutes=start_time.minute)
                datetime_obj = date_obj + start_time_delta
                action_list = activity['content']
                self.dynamic_info_schedule[datetime_obj].extend(copy.deepcopy(action_list))
    
    @property
    def activity_list(self):
        return self.daily_schedules['activity_name'].tolist()
    
    def filter_events(
        self, person_name=None, activity_name: str=None, date: datetime=None, 
        start_time: timedelta=None, end_time: timedelta=None, **args
    ):
        # if the start_time or the end_time is not None, then the date must not be None
        if start_time is None and end_time is None and date is None:
            schedule_df = self.filter_schedules_by_period(end_time=self.current_time)
        else:
            schedule_df = self.daily_schedules
        schedule_df = self.filter_schedule_df(
            schedule_df=schedule_df, person_name=person_name, activity_name=activity_name,
            date=date, start_time=start_time, end_time=end_time, **args
        )
        return schedule_df
    
    @staticmethod
    def filter_schedule_df(
        schedule_df: pd.DataFrame, person_name: str=None, activity_name: str=None,
        date: datetime=None, start_time: timedelta=None, end_time: timedelta=None, **args 
    ) -> pd.DataFrame:
        # room and tiny_objects value maybe None, so we should use mask_room and mask_tiny_objects to control whether to mask the room and tiny_objects
        mask_room = 'room' not in args
        mask_tiny_objects = 'tiny_objects' not in args
        room = args.get('room', None)
        tiny_objects = args.get('tiny_objects', None)
        assert (start_time is None and end_time is None) or date is not None
        if person_name is not None:
            schedule_df = schedule_df[schedule_df['person_name'] == person_name]
        if activity_name is not None:
            schedule_df = schedule_df[schedule_df['activity_name'] == activity_name]
        if not mask_room:
            if room is None:
                schedule_df = schedule_df[schedule_df['room'].isnull()]
            else:
                schedule_df = schedule_df[schedule_df['room'] == room]
        if not mask_tiny_objects:
            if tiny_objects is None:
                schedule_df = schedule_df[schedule_df['tiny_objects'].isnull()]
            else:
                schedule_df = schedule_df[schedule_df['tiny_objects'] == tiny_objects]
        if date is not None:
            schedule_df = schedule_df[schedule_df['date'] == date]
        if start_time is not None:
            schedule_df = schedule_df[schedule_df['start_time'] == start_time]
        if end_time is not None:
            schedule_df = schedule_df[schedule_df['end_time'] == end_time]
        return schedule_df
    
    # @staticmethod
    # def set_schedule_df(
    #     schedule_df: pd.DataFrame, set_value_function: callable, person_name: str=None, activity_name: str=None, 
    #     room: str=None, date: datetime=None, start_time: timedelta=None, end_time: timedelta=None,
    # ):
    #     # 假设原始的DataFrame是 original_df

    #     if person_name is not None:
    #         mask &= (original_df['person_name'] == person_name)
    #     if activity_name is not None:
    #         mask &= (original_df['activity_name'] == activity_name)
    #     # 以此类推，为其他条件添加布尔索引逻辑

    #     # 使用 loc 进行赋值操作，确保修改反映到原始 DataFrame
    #     original_df.loc[mask, 'new_column'] = new_value

        
    def get_schedule_by_time(self, time: datetime) -> pd.Series:
        # we should make sure that only one schedule match the time
        # the start_time < time <= end_time
        schedule_df = self.daily_schedules[self.daily_schedules['start_time'] < time and self.daily_schedules['end_time'] >= time]
        assert len(schedule_df) == 1
        return schedule_df.iloc[0]
    
    def filter_schedules_by_period(self, start_time: datetime=None, end_time: datetime=None):
        schedule_df = self.daily_schedules
        if start_time is not None:
            schedule_df = schedule_df[self.get_execute_time(schedule_df) >= start_time]
        if end_time is not None:
            schedule_df = schedule_df[self.get_execute_time(schedule_df) < end_time]
        return schedule_df
    
    def get_execute_time(self, activity):
        if activity is None:
            activity = self.daily_schedules
        return activity['date'] + activity['start_time']

    @staticmethod
    def event2str(schedule: pd.Series, mask_tiny_objects: bool=False):
        # activity name, start time, end time, room, tiny objects
        activity_name = schedule['activity_name']
        date = schedule['date']
        start_time = schedule['start_time']
        end_time = schedule['end_time']
        room = schedule['room']
        tiny_objects = schedule['tiny_objects']
        person_name = schedule['person_name']
        prefix = f"{Schedule_Table.date_obj2str(date)} {Schedule_Table.timedelta_obj2str(start_time)}-{Schedule_Table.timedelta_obj2str(end_time)}: {person_name} was {activity_name}"
        room_suffix = f" in {room}" if room is not None else ""
        if tiny_objects is None or mask_tiny_objects:
            tiny_objects_suffix = ""
        elif len(tiny_objects) == 0:
            tiny_objects_suffix = " with nothing"
        else:
            tiny_objects_suffix = f" with {tiny_objects}"
        return prefix + room_suffix + tiny_objects_suffix
        
    @staticmethod
    def str2event(schedule_str: str):
        # get start time, end time, person_name, activity_name, room, tiny_objects
        schedule_str = schedule_str.strip()
        date_str = schedule_str[:10]
        date_obj = Schedule_Table.date_str2obj(date_str)
        start_time_str = schedule_str[11:16]
        start_time = Schedule_Table.timedelta_str2obj(start_time_str)
        end_time_str = schedule_str[17:22]
        end_time = Schedule_Table.timedelta_str2obj(end_time_str)
        person_name = schedule_str[24:schedule_str.find(" was")]
        activity_name = schedule_str[28+len(person_name)+1:]
        # activity_name_end_index = activity_name.find(" in ") or activity_name.find(" with ") or len(activity_name)
        if " in " in activity_name:
            activity_name_end_index = activity_name.find(" in ")
        elif " with " in activity_name:
            activity_name_end_index = activity_name.find(" with ")
        else:
            activity_name_end_index = len(activity_name)
        activity_name = activity_name[:activity_name_end_index]
        activity_name = activity_name.strip()
        room = None
        tiny_objects = None
        if " in " in schedule_str:
            room = schedule_str[schedule_str.find(" in ")+4:].split()[0]
        if " with " in schedule_str:
            tiny_objects = schedule_str[schedule_str.find(" with ")+6:]
            if "nothing" in tiny_objects:
                tiny_objects = ""
            else:
                tiny_objects = tiny_objects.strip()
        return date_obj, start_time, end_time, person_name, activity_name, room, tiny_objects

    @staticmethod
    def dict2str(data: dict) -> dict:
        """
        data:
            {
                "date": datetime | None,
                "start_time": timedelta | None,
                "end_time": timedelta | None,
                "person_name": str,
                "activity_name": str,
                "room": str | None,
                "tiny_objects": str | None,
                "rooms": list[str], optional, it can not exist at the same time with the room
            }
        """
        # the date, start_time, end_time must all not be None or all None, the person_name and the activity_name must not be None
        assert ('date' in data and 'start_time' in data and 'end_time' in data) or ('date' not in data and 'start_time' not in data and 'end_time' not in data)
        assert 'person_name' in data and 'activity_name' in data
        person_name, activity_name = data['person_name'], data['activity_name']
        date, start_time, end_time = data.get('date', None), data.get('start_time', None), data.get('end_time', None)
        room = data.get('room', None)
        tiny_objects = data.get('tiny_objects', None)
        description = ""
        if date is not None:
            description += f"{Schedule_Table.date_obj2str(date)} {Schedule_Table.timedelta_obj2str(start_time)}-{Schedule_Table.timedelta_obj2str(end_time)}: "
        description += f"{person_name} was {activity_name}"
        if room is not None:
            description += f" in {room}"
        if tiny_objects is not None:
            if len(tiny_objects) == 0:
                description += " with nothing"
            else:
                description += f" with {tiny_objects}"
        if 'rooms' in data:
            description += "(Note that this event has been executed in {rooms})".format(rooms=", ".join(data['rooms']))
        return description
    
    @staticmethod
    def str2dict(schedule_str: str) -> dict:
        """
        return dict:
            key: date, start_time, end_time, person_name, activity_name, room, tiny_objects 
        """
        # the schedule_str must be the output of the function dict2str
        if '(Note that this event has been executed in ' in schedule_str:
            # remove this note
            schedule_str = schedule_str[:schedule_str.find('(Note that this event has been executed in ')]
        if ':' in schedule_str:
            args = Schedule_Table.str2event(schedule_str)  
            date, start_time, end_time, person_name, activity_name, room, tiny_objects = args
            return {
                "date": date,
                "start_time": start_time,
                "end_time": end_time,
                "person_name": person_name,
                "activity_name": activity_name,
                "room": room,
                "tiny_objects": tiny_objects
            }
        else:
            schedule_str = "2023-02-13 06:00-06:10: " + schedule_str
            args = Schedule_Table.str2event(schedule_str)
            _, _, _, person_name, activity_name, room, tiny_objects = args
            return {
                "person_name": person_name,
                "activity_name": activity_name,
                "room": room,
                "tiny_objects": tiny_objects
            }        
    
    def set_current_time(self, time: datetime):
        self.current_time = time
    
    def pass_time(self, delta_time: timedelta=None, new_time: datetime=None) -> list[dict]:
        """
        get the actions in the time interval [self.current_time, new_time)
        each action is like {"object": "Cloth|surface|3|11", "receptacle": "Dresser|5|1"}
        """
        # make sure that the delta_time and new_time only enable one
        assert (delta_time is None) or (new_time is None)
        if delta_time is not None:
            new_time += self.current_time + delta_time
        elif new_time is None:
            raise Exception("delta_time and new_time can not be both None")
        last_time = self.current_time
        self.current_time = new_time
        
        action_list = []
        for schedule_time in self.dynamic_info_schedule:
            if schedule_time >= last_time and schedule_time < new_time:
                action_list.extend(self.dynamic_info_schedule[schedule_time])
            elif schedule_time >= new_time:
                break
        return action_list
    
    def get_full_schedules(self, days: int) -> pd.DataFrame:
        """
        return:
            a pd.DataFrame contains the schedule from the current time - days to the current time
            and the 'room' of the schedule should be masked
        """
        schedules_in_days = self.filter_schedules_by_period(start_time=self.current_date - timedelta(days=days), end_time=self.current_time).copy()
        # mask the room
        schedules_in_days['room'] = schedules_in_days['room'].apply(lambda _: None)
        # add a column to store the tiny objects
        schedules_in_days['tiny_objects'] = schedules_in_days['tiny_objects'].apply(lambda _: None)
        # add a column to store the person name
        return schedules_in_days