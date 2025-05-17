from collections import Counter, defaultdict
import json
import pandas as pd
import numpy as np
from simulator.probability import Probablity
from simulator.constants import ACTIVITY_LIST
from simulator.scene_graph import Semantic_Map
from simulator.schedule import Schedule_Table
from sentence_transformers import SentenceTransformer
from abc import ABCMeta, abstractmethod

from simulator.task import Task
from simulator.utils import unique_list

class Person:
    def __init__(self, schedule: Schedule_Table, task: Task,  probability: Probablity, full_days: int=2) -> None:
        # share the schedule with the dynamic_thor, and here we won't change the schedule
        self.schedule_table = schedule
        self.task = task
        self.bad_memory = []
        self.qa_memory = []
        self.full_days = full_days
        self.probability = probability
        self.full_schedule: pd.DataFrame = self.get_full_schedule()
        # it is a list of tuple, the first element is the query type, the second element is the schedule inference the full schedule
        self.query_type_list = ['room', 'tiny objects']

    @property
    def known_schedule(self):
        return self.full_schedule[~self.full_schedule['mask']]
    
    def get_full_schedule(self) -> pd.DataFrame:
        """
        probability: a str, inference to the Probability class a key
        worth_ask: bool, if the person should ask the room where it will happen
        room: str|None, if None it means the room is unknown, else it means the room is known
        mask: bool, if True, it means the schedule is not known
        tiny_objects: list[str]|None, if None it means the tiny_objects is unknown, else it means the tiny_objects is known
        """
        full_schedules = self.schedule_table.get_full_schedules(self.full_days)
        # add a column named 'receptacles'
        full_schedules['probability'] = full_schedules['activity_name'].apply(lambda _: self.probability.insert_data())
        full_schedules['worth_ask'] = full_schedules['activity_name'].apply(lambda _: True)
        full_schedules['mask'] = full_schedules['activity_name'].apply(lambda _: False)
        full_schedules['worth_analysis'] = full_schedules['activity_name'].apply(lambda _: None)
        return full_schedules

    def set_full_schedule_value(self, index: int, key: str, value: any):
        self.full_schedule.loc[index, key] = value
    
    def get_locations_of_activity(self, activity_name: str) -> list[str|None]:
        """
        return
            1. []: it means that the activity_name is not in the schedule
            2. [None]*n: it means that the activity_name is in the schedule, but the room is unknown
            3. [str]*n: it means that all the room are known
        """
        # mask is False and the activity_name is the same
        rooms: pd.Series = self.known_schedule[self.known_schedule['activity_name'] == activity_name]['room']
        # return all the rooms that this activity may happen, keep it unique, 
        return rooms.unique().tolist()

    def set_full_schedule_unknown(self):
        self.full_schedule['mask'] = True

    def filter_full_schedule(self, events: list[tuple], ignore_error: bool=False) -> bool:
        # if target object in full_schedule's tiny objects, then set the mask to False
        self.full_schedule['mask'] = self.full_schedule.apply(lambda event: False if event['tiny_objects'] is not None and self.task.target_object in event['tiny_objects'] else event['mask'], axis=1)
        # add the filter events
        for date, start_time, end_time, _, activity_name, room, tiny_objects in events:
            schedule_df = Schedule_Table.filter_schedule_df(schedule_df=self.full_schedule, activity_name=activity_name, date=date, start_time=start_time, end_time=end_time)
            if len(schedule_df) != 1 and not ignore_error:
                return False
            event = schedule_df.iloc[0]
            index = schedule_df.index[0]
            if (event['room'] != room or event['tiny_objects'] != tiny_objects) and not ignore_error:
                return False
            if tiny_objects is not None:
                continue
            self.set_full_schedule_value(index, 'mask', False)
        return True
        
    def get_choice_list(self, query_type):
        if query_type == 'room':
            # if the schedule in self.full_schedules and the schedule's room is None, the it can be asked, and should be added to the choice list
            # if the room is an empty dict, it means the room is unknown
            return self.known_schedule[self.known_schedule['room'].isna()]
        else:
            # if the schedule in self.full_schedules and the schedule's tiny_objects is None, the it can be asked, and should be added to the choice list
            return self.known_schedule[self.known_schedule['tiny_objects'].isna()]
    
    def get_bad_memory(self) -> str | None:
        if len(self.bad_memory) == 0:
            return None
        bad_memory_description = ""
        for query_type, query_args, reason in self.bad_memory:
            if query_type == 'room':
                date, start_time, end_time, _, activity_name, _, _ = query_args
                bad_memory_description += f"Error: You asked 'In which room was {self.person_name} {activity_name} from {start_time} to {end_time} on {date}?' but {reason}\n"
            else:
                date, start_time, end_time, _, activity_name, _, _ = query_args
                bad_memory_description += f"Error: You asked 'What tiny objects did {self.person_name} move while {activity_name} from {start_time} to {end_time} on {date}?' but {reason}\n"
    
    def get_known_schedule(self) -> pd.DataFrame:
        return self.known_schedule
    
    def get_answer(self, query_type: str, query_args: tuple) -> bool:
        if query_type not in self.query_type_list:
            return False
        date_obj, start_time, end_time, _, activity_name, room, tiny_objects = query_args
        if query_type == 'room' and room is not None:
            self.bad_memory.append((query_type, query_args, "the room is already known"))
            return False
        elif query_type == 'tiny objects' and tiny_objects is not None:
            self.bad_memory.append((query_type, query_args, "the tiny_objects is already known"))
            return False
        oracle_schedule_df = self.schedule_table.filter_events(activity_name=activity_name, date=date_obj, start_time=start_time, end_time=end_time)
        local_schedule_df = Schedule_Table.filter_schedule_df(self.known_schedule, date=date_obj, activity_name=activity_name, start_time=start_time, end_time=end_time)
        if len(oracle_schedule_df) == 0 or len(local_schedule_df) == 0:
            self.bad_memory.append((query_type, query_args, "no such schedule"))
            return False
        if len(oracle_schedule_df) != 1 or len(local_schedule_df) != 1:
            raise ValueError("multiple schedules")

        oracle_event = oracle_schedule_df.iloc[0]
        local_event = local_schedule_df.iloc[0]
        local_index = local_schedule_df.index[0]
        if query_type == 'room':
            if not local_event['worth_ask']:
                self.bad_memory.append((query_type, query_args, f"it is not worth asking, {local_event['worth_analysis']}"))
                return False
            oracle_room = oracle_event['room']
            if oracle_room is None:
                raise ValueError("here is a schedule without room")
            elif local_event['room'] is None:
                self.qa_memory.append((query_type, query_args, oracle_room))
                self.update_query(local_event, local_index, query_type, oracle_room)
                return True
            else:
                self.bad_memory.append((query_type, query_args, "the room is already known"))
                return False
        else:
            tiny_objects = oracle_event['tiny_objects']
            # oracle tiny objects can be [] but not None
            if tiny_objects is None:
                raise ValueError("here is a schedule without tiny_objects")
            elif local_event['tiny_objects'] is None:
                self.qa_memory.append((query_type, query_args, tiny_objects))
                self.update_query(local_event, local_index, query_type, tiny_objects)
                return True
            else:
                self.bad_memory.append((query_type, query_args, "the tiny_objects is already known"))
                return False
            
    def get_query_and_answer_memory(self) -> str:
        """
        return a string describe the query and answer memory
        """
        if len(self.qa_memory) == 0:
            return None
        qa_memory_description = ""
        for query_type, query_args, answer in self.qa_memory:
            if query_type == 'room':
                date, start_time, end_time, _, activity_name, _, _ = query_args
                qa_memory_description += f"Q: Which room did {self.person_name} visit when {self.person_name} was {activity_name} from {start_time} to {end_time} on {date}?\n"
                qa_memory_description += f"A: {answer}\n"
            else:
                date, start_time, end_time, _, activity_name, _, _ = query_args
                qa_memory_description += f"Q: Which tiny objects did {self.person_name} move when {self.person_name} was {activity_name} from {start_time} to {end_time} on {date}?\n"
                qa_memory_description += f"A: {answer}\n" if len(answer) > 0 else "A: Nothing\n"
        return qa_memory_description
    
    def get_move_prob(self, event: pd.Series)-> float|None:
        if event['tiny_objects'] is None:
            return None
        if self.task.target_object not in event['tiny_objects']:
            move_prob = 0.0
        else:
            move_prob = 1.0
        return move_prob
    
    def update_query(self, local_event: pd.Series, local_index: int, query_type: str, answer: str | list[str]):
        if query_type == 'room':
            oracle_room = answer
            self.set_full_schedule_value(local_index, 'room', oracle_room)
            self.probability.set_oracle_room(oracle_room, local_event['probability'], self.get_move_prob(local_event))
        else:
            oracle_tiny_objects = answer
            self.set_full_schedule_value(local_index, 'tiny_objects', oracle_tiny_objects)
            # if the target object in the tiny objects
            move_to_room_prob = 1.0 if self.task.target_object in oracle_tiny_objects else 0.0
            self.probability.set_oracle_move_to_room_probability(local_event['probability'], move_to_room_prob)
                
    def update_room_analysis(self, analysis_args: tuple, event_args: dict) -> bool:
        """
        analysis_args: (worth_ask: bool, predictions: dict[str, float], analysis:str)
        """
        worth_ask, predictions, analysis = analysis_args
        update_events = Schedule_Table.filter_schedule_df(self.known_schedule, **event_args)
        have_set = False
        for index, update_event in update_events.iterrows():
            if update_event['room'] is not None:
                continue
            self.set_full_schedule_value(index, 'worth_ask', worth_ask)
            self.set_full_schedule_value(index, 'worth_analysis', analysis)
            self.probability.set_room_analysis(update_event['probability'], predictions, self.get_move_prob(update_event))
            have_set = True
        return have_set
    
    def update_event_analysis(self, analysis_args: tuple, event_args: dict) -> bool:
        """
        event_args:
            room: str; the room where the event happend
            activity_name: str; the activity name of the event
            date[optional]: datetime; the date of the event
            start_time[optional]: timedelta; the start time of the event
            end_time[optional]: timedelta; the end time of the event
        analysis_args:
            probability: float; the probability that the room is the room where the event happend
        """
        probability, = analysis_args
        update_room = event_args['room']
        event_args.pop('room')
        update_events = Schedule_Table.filter_schedule_df(self.known_schedule, **event_args)
        have_set = False
        for _, update_event in update_events.iterrows():
            if update_event['room'] is not None and update_event['room'] != update_room:
                continue
            else:
                have_set = self.probability.set_event_analysis(update_event['probability'], update_room, probability) or have_set
        return have_set
    
    def update_receptacle_analysis(self, analysis_args: tuple, event_args: dict):
        """
            analysis_args:
                receptacle_probability: dict[str, float], the key is the receptacle id, and the value is the probability that the receptacle is the receptacle where the event happend
        """
        receptacle_predictions, = analysis_args
        update_room = event_args['room']
        event_args.pop('room')
        update_events = Schedule_Table.filter_schedule_df(self.known_schedule, **event_args)
        have_set = False
        for _, update_event in update_events.iterrows():
            if update_event['room'] is not None and update_event['room'] != update_room:
                continue
            else:
                have_set = self.probability.set_receptacle_analysis(update_event['probability'], update_room, receptacle_predictions) or have_set
        return have_set

    @abstractmethod
    def clear_query_memory(self):
        self.bad_memory.clear()
        self.qa_memory.clear()
        self.full_schedule = self.schedule_table.get_full_schedules(self.full_days)

    @abstractmethod
    def reset(self, schedule: Schedule_Table):
        self.clear_query_memory()
        self.schedule_table = schedule

    @property
    def person_name(self):
        return self.schedule_table.person_name
    
    
class Persons(metaclass=ABCMeta):
    def __init__(self, schedules: list[Schedule_Table], task: Task) -> None:
        self.schedule_table_list = schedules
        self.task = task # read only
        # self.reset_task()
        
    @abstractmethod
    def reset_task(self):
        for schedule_table in self.schedule_table_list:
            schedule_table.set_current_time(self.task.current_time)
        self.probability = Probablity()
        persons = [Person(schedule=schedule, task=self.task, probability=self.probability) for schedule in self.schedule_table_list]
        self.person_dict = {person.person_name: person for person in persons}
        self.bad_memory = []
        self.query_type_list = ['room', 'tiny objects']

    @abstractmethod
    def clear_query_memory(self):
        pass

    @abstractmethod
    def get_receptacle_in_room_probability(self, receptacle_id: str) -> float:
        return 1.0
    
    @abstractmethod
    def update_receptacle_in_room_probability(self, receptacle_id: str, probability: float):
        pass

    @abstractmethod
    def update_scene_graph(self):
        self.probability.update_scene_graph(self.scene_graph)
        
    @property
    def scene_graph(self) -> Semantic_Map:
        return self.task.semantic_map
    
    def set_time(self, time):
        for person_name in self.person_dict:
            person = self.person_dict[person_name]
            person.schedule_table.set_current_time(time)

    def update_room_analysis(self, analysis_args: tuple, person_name: str, event_args: dict[str, any]) -> bool:
        if person_name not in self.person_name_list:
            return False
        answer_person = self.get_answer_person(person_name)
        return answer_person.update_room_analysis(analysis_args, event_args)

    def update_event_analysis(self, analysis_args: tuple, person_name: str, event_args: dict[str, any]) -> bool:
        if person_name not in self.person_name_list:
            return False
        answer_person = self.get_answer_person(person_name)
        return answer_person.update_event_analysis(analysis_args, event_args)
    
    def update_receptacle_analysis(self, analysis_args: tuple, person_name: str, event_args: dict[str, any]) -> bool:
        if person_name not in self.person_name_list:
            return False
        answer_person = self.get_answer_person(person_name)
        return answer_person.update_receptacle_analysis(analysis_args, event_args)

    def get_person_events(self, mask_time: bool, mask_room: bool, mask_tiny_objects: bool, person_name: str= None, event_args: dict = None) -> list[dict[str, any]]:
        """
        args:
            mask_room: 
                true, the event['room'] must be None and don't add the event['room'] in the return data; 
                false, we will add the event['room'] in the return data if the event['room'] is not None, or we will add all the key in event['rooms'] in the return data
            mask_time: true then we add the date, start_time, end_time in the return data, else don't
            mask_tiny_objects: true then we add the tiny_objects in the return data, else don't
        return:
            a list of dict, each dict is a event, the key is the name of the event, and the value is the value of the event
        """
        events = []
        answer_persons = [self.get_answer_person(person_name)] if person_name is not None else self.answers
        for answer_person in answer_persons:
            known_schedule = answer_person.get_known_schedule()
            if event_args is not None:
                known_schedule = Schedule_Table.filter_schedule_df(known_schedule, **event_args)
            for _, event in known_schedule.iterrows():
                data = dict(person_name=answer_person.person_name, activity_name=event['activity_name'])
                if not mask_time:
                    data.update(date=event['date'], start_time=event['start_time'], end_time=event['end_time'])
                if not mask_tiny_objects:
                    data.update(tiny_objects=event['tiny_objects'])
                if not mask_room:
                    if event['room'] is not None:
                        data.update(room=event['room'])
                        events.append(data)
                    else:
                        datas = []
                        for candidate_room in answer_person.probability.get_candidate_rooms(event['probability']):
                            data.update(room=candidate_room)
                            datas.append(data.copy())
                        events.extend(datas)
                elif event['room'] is not None:
                    continue
                else:
                    rooms = answer_person.get_locations_of_activity(event['activity_name'])
                    rooms = [room for room in rooms if room is not None]
                    if len(rooms) > 0:
                        data.update(rooms=rooms)
                    events.append(data)
        # remove the duplicate events
        return unique_list(events)
    
    def get_choice_list(self, query_type: str):
        return pd.concat([answer.get_choice_list(query_type) for answer in self.answers])
     
    @property
    def answers(self) -> list[Person]:
        # the person who's name in self.task.belongs
        return [self.person_dict[person_name] for person_name in self.task.belongs]
        
    def filter_full_schedule(self, events: list[str], ignore_error: bool=False) -> bool:
        person2events = defaultdict(list)
        for event in events:
            try:
                event = Schedule_Table.str2event(event)
                person_name = event[3]
                person2events[person_name].append(event)
            except:
                if ignore_error:
                    continue
                else:
                    return False
        for answer in self.answers:
            answer.set_full_schedule_unknown()
        for person_name, events in person2events.items():
            answer_person = self.get_answer_person(person_name)
            if answer_person is None:
                if ignore_error:
                    continue
                else:
                    return False
            if not answer_person.filter_full_schedule(events, ignore_error):
                return False
        return True
    
    def get_answer(self, query_type: str, query_args: tuple) -> bool:
        schedule_str, = query_args
        try:
            if query_type not in self.query_type_list:
                self.bad_memory.append((query_type, None))
            query_args = Schedule_Table.str2event(schedule_str)
        except:
            self.bad_memory.append((None, schedule_str))
            return False

        person_name = query_args[3]
        answer_person = self.get_answer_person(person_name)
        if answer_person is None:
            self.bad_memory.append((None, schedule_str))
            return False
        return answer_person.get_answer(query_type, query_args)
    
    def get_query_and_answer_memory(self) -> str:
        qa_memory_description = "Here are the Q&A records:\n"
        all_none = True
        for answer in self.answers:
            qa = answer.get_query_and_answer_memory()
            if qa is not None:
                all_none = False
                qa_memory_description += qa
        if all_none:
            return ""
        else:
            return qa_memory_description + "\n"
    
    def get_bad_memory(self) -> str:
        bad_memory_description = "Warning: You have asked some wrong questions as follows, you should avoid asking them again:\n"
        all_none = True
        for answer in self.answers:
            bad_memory = answer.get_bad_memory()
            if bad_memory is not None:
                all_none = False
                bad_memory_description += bad_memory
        for bad_memory in self.bad_memory:
            all_none = False
            if bad_memory[0] is not None and bad_memory[1] is None:
                bad_memory_description += f"Error: You asked a query type: {bad_memory[0]}, but no such query type\n"
            elif bad_memory[0] is None and bad_memory[1] is not None:
                bad_memory_description += f"Error: You asked a schedule: {bad_memory[1]}, but the schedule format is wrong\n"
        if all_none:
            return ""
        else:
            return bad_memory_description + "\n"
    
    def get_known_events_description(self):
        # concat all the known schedules
        all_schedules = pd.concat([answer.get_known_schedule() for answer in self.answers])
        # sorted by the date + start_time
        if len(all_schedules) == 0:
            return []
        all_schedules.sort_values(by=['date', 'start_time'], inplace=True)
        known_events = []
        # 遍历所有的 schedule, 加上 schedule 的 string 格式
        for _, schedule in all_schedules.iterrows():
            known_events.append(Schedule_Table.event2str(schedule, False))
        return known_events

    def get_full_events_description(self):
        all_schedules = pd.concat([answer.full_schedule for answer in self.answers])
        # sorted by the date + start_time
        all_schedules.sort_values(by=['date', 'start_time'], inplace=True)
        if len(all_schedules) == 0:
            return []
        # 遍历所有的 schedule, 加上 schedule 的 string 格式
        full_events = []
        for _, schedule in all_schedules.iterrows():
            # delete that which has known tiny objects
            if schedule['tiny_objects'] is not None:
                continue
            full_events.append(Schedule_Table.event2str(schedule, False))
        return full_events
    
    def update_object_in_receptacles_without_schedule(self, analysis_receptacles: dict[str, float]):
        return self.probability.set_none_value(analysis_receptacles)
    
    def update_object_in_receptacles_without_cot(self, analysis_receptacles: dict[str, float]):
        """
        this is without chain of thought, directly output receptacle and the probability
        """
        return self.probability.set_none_value(analysis_receptacles)

    def get_answer_person(self, person_name: str) -> Person | None:
        return self.person_dict.get(person_name, None)
    
    def get_receptacle_probability_table(self) -> pd.DataFrame:
        """
        1. for each event, we sorted them by the time.
        2. then we use the probability to get the probability that it move the target object to each receptacle, and it don't move the receptacle
        3. 
        """
        # if find the target object in receptacle of the scene graph, just return this receptacle
        target_receptacle_counter_id = self.scene_graph.get_receptacle_counter_id_from_object_type_in_it(self.task.target_object)
        if target_receptacle_counter_id is not None:
            return pd.DataFrame.from_dict({target_receptacle_counter_id: [1.0]}, orient='index', columns=['probability'])
        known_schedule = pd.concat([answer.get_known_schedule() for answer in self.answers])
        self.update_scene_graph()
        # sorted it from the late to the early
        known_schedule.sort_values(by=['date', 'start_time'], ascending=False, inplace=True)
        last_not_move_prob = 1.0
        receptacle_prob_dict = Counter()
        for _, event in known_schedule.iterrows():
            move_prob, receptacle_probs = self.probability.get_probability(event['probability'])        
            for receptacle_id, receptacle_prob in receptacle_probs.items():
                receptacle_prob_dict[receptacle_id] += receptacle_prob * self.get_receptacle_in_room_probability(receptacle_id) * last_not_move_prob
            last_not_move_prob *= (1 - move_prob / self.task.target_object_num)
        for receptacle_id, receptacle_prob in self.probability.get_none_value().items():
            receptacle_prob_dict[receptacle_id] += receptacle_prob * last_not_move_prob * self.task.target_object_num
        # delete the receptacle that is explored in scene graph
        delete_receptacles = self.task.semantic_map.get_explored_receptacle_counter_id_list()
        for delete_receptacle in delete_receptacles:
            if delete_receptacle in receptacle_prob_dict:
                del receptacle_prob_dict[delete_receptacle]
        # if the target object in room of the scene graph, we just remain the receptacle in the target room
        target_room_counter_id = self.scene_graph.get_room_counter_id_from_object_type_in_it(self.task.target_object)
        if target_room_counter_id is not None:
            receptacle_prob_dict = {receptacle_id: receptacle_prob for receptacle_id, receptacle_prob in receptacle_prob_dict.items() if self.scene_graph.check_receptacle_maybe_in_room(receptacle_id, target_room_counter_id)}
        # normalize the probability
        total_prob = sum(receptacle_prob_dict.values())
        if total_prob == 0:
            return pd.DataFrame(columns=['probability'])
        for receptacle_id, receptacle_prob in receptacle_prob_dict.items():
            receptacle_prob_dict[receptacle_id] = receptacle_prob / total_prob
        # return the probability table, with two columns, the first column is the 'receptacle_id', and the second column is the 'probability'
        df = pd.DataFrame.from_dict(receptacle_prob_dict, orient='index', columns=['probability'])
        # sorted it by the probability
        df.sort_values(by=['probability'], ascending=False, inplace=True)
        return df

    def get_room_probability_table(self) -> pd.DataFrame:
        receptacle_probability_table = self.get_receptacle_probability_table()
        # we add a column named 'room' in the receptacle_probability_table
        receptacle_probability_table['room'] = receptacle_probability_table.index.map(lambda receptacle_id: self.scene_graph.get_room_counter_id_from_receptacle_counter_id(receptacle_id))
        # we need to group by the room and sum the probability
        room_probability_dict = receptacle_probability_table.groupby('room').sum().to_dict()['probability']
        # we don't need to normalize the probability, because the receptacle_probability_table is normalized
        return pd.DataFrame.from_dict(room_probability_dict, orient='index', columns=['probability'])
        
    @property
    def person_name_list(self):
        return list(self.person_dict.keys())
    

class Persons_With_Full_Scene(Persons):
    def __init__(self, schedules: list[Schedule_Table], task: Task) -> None:
        super().__init__(schedules, task)
    
    def reset_task(self):
        return super().reset_task()
    
    def clear_query_memory(self):
        pass

    def get_receptacle_in_room_probability(self, receptacle_id: str) -> float:
        return super().get_receptacle_in_room_probability(receptacle_id)

    def update_receptacle_in_room_probability(self, receptacle_id: str, probability: float):
        pass
    
    def update_scene_graph(self):
        return super().update_scene_graph()

class Persons_With_Partly_Scene(Persons):
    def __init__(self, schedules: list[Schedule_Table], task: Task) -> None:
        super().__init__(schedules, task)
    
    def reset_task(self):
        self.receptacle_in_room_probability_table = pd.DataFrame(columns=['receptacle', 'probability'])
        return super().reset_task()
    
    def update_receptacle_in_room_probability(self, receptacle: str, probability: float):
        # if the room and receptacle row is in the table, then just update the probability
        update_row = self.receptacle_in_room_probability_table[self.receptacle_in_room_probability_table['receptacle'] == receptacle]
        if len(update_row) > 0:
            self.receptacle_in_room_probability_table.loc[update_row.index, 'probability'] = probability
        else:
            # add a line in it
            self.receptacle_in_room_probability_table.loc[len(self.receptacle_in_room_probability_table)] = [receptacle, probability]
        
    def get_receptacle_in_room_probability(self, receptacle_id: str) -> float:
        # if the receptacle_id is in the table, then return the probability
        ground_receptacle_id_list = self.scene_graph.ground_receptacle_counter_id(receptacle_id)
        if len(ground_receptacle_id_list) != 1 or ground_receptacle_id_list[0] != receptacle_id:
            return 0
        row = self.receptacle_in_room_probability_table[self.receptacle_in_room_probability_table['receptacle'] == receptacle_id]
        if len(row) > 0:
            return row.iloc[0]['probability']
        else:
            # it means that this receptacle is seen receptacle in the scene graph
            return 1
        
    def update_scene_graph(self):
        super().update_scene_graph()
        # we need to update the receptacle_in_room_probability_table, we first transfer it to a dict
        # group by the receptacle and then sum the probability
        receptacle_in_room_probability_dict: dict = self.receptacle_in_room_probability_table.groupby('receptacle').sum().to_dict()['probability']
        new_receptacle_in_room_probability_dict = {}
        for receptacle_id, receptacle_in_room_prob in receptacle_in_room_probability_dict.items():
            if self.scene_graph.check_receptacle_sure_in_house(receptacle_id):
                continue
            ground_receptacle_id_list = self.scene_graph.ground_receptacle_counter_id(receptacle_id)
            if len(ground_receptacle_id_list) == 1 and ground_receptacle_id_list[0] == receptacle_id:
                new_receptacle_in_room_probability_dict[receptacle_id] = receptacle_in_room_prob
        # transfer the new dict to the table, the key must to be the 'receptacle' column, not the index
        self.receptacle_in_room_probability_table = pd.DataFrame.from_dict(new_receptacle_in_room_probability_dict, orient='index', columns=['probability']).reset_index().rename(columns={'index': 'receptacle'})
        
    def clear_query_memory(self):
        pass
    
    def reset(self, schedules: list[Schedule_Table]):
        return super().reset(schedules)