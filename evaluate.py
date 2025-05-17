import json
import pickle
import copy
from ai2thor.util.metrics import compute_spl, get_shortest_path_to_object
import os
from tqdm import tqdm
import numpy as np


import sys
sys.path.append('/data/wdz/Dynamic_Scene2')
from simulator.dynamic_thor import Dynamic_Thor


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

def get_metadata_by_objectid(event, obj_id):
    for obj in event.metadata['objects']:
        if obj['objectId'] == obj_id:
            return obj
    return None


def compute_SR(episode_info_list):
    success_list = [episode_info['success'] for episode_info in episode_info_list]
    return np.mean(success_list)
    
def compute_SPL(episode_info_list):

    # for episode_info in episode_info_list:
    #     episode = {
    #         "path": episode_info['path'],
    #         "shortest_path": episode_info['shortest_path'],
    #         "success": episode_info['success']
    #     }
    #     episode_list.append(episode)
    return compute_spl(episode_info_list)

def get_shortest_path_to_point(env, initial_position, target_position):
    kwargs = dict(
        action="GetShortestPathToPoint",
        position=initial_position,
        target=target_position
    )
    event = env.step(kwargs)
    if event.metadata["lastActionSuccess"]:
        return event.metadata["actionReturn"]["corners"]
    else:
        raise ValueError(
            "Unable to find shortest path to point '{}'  due to error '{}'.".format(
                target_position, event.metadata["errorMessage"]
            )
        )
    

def evaluate_single_episode(episode_info, env, max_steps=[300, 400, 500]):
    #SR, SPL, NE, Episode length average
    episode_id = episode_info['task_info']['id']
    task_info = episode_info['task_info']
    
    results = [{'max_steps':ms,} for ms in max_steps]
    
    
    episode_length = episode_info['ep_length']
    success = episode_info['oracle_success']
    success_steps = episode_info['oracle_success_steps']
    
    target_metadata = get_metadata_by_objectid(env.last_event, task_info['target_object_ids'][0])
    target_pos = target_metadata['position']
    target_object = task_info['target_object_ids'][0]
    # print(episode_id)
    # print(episode_steps)
    # print(len(task_info['followed_path']))

        # shortest_path = get_shortest_path_to_point(env=env, 
        #                                            initial_position=path[0], 
        #                                            target_position=target_pos)
    try:
        shortest_path = get_shortest_path_to_object(controller=env,
                                                    object_id=target_object,
                                                    initial_position=task_info['followed_path'][0])
    except:
        shortest_path = []
    
    last_agent_position = None
    
    for i, res in enumerate(results):
        max_steps = res['max_steps']
        #success
        if success_steps < 0:
            results[i]['success'] = False
            results[i]['episode_steps'] = min(max_steps,episode_length)
        else:
            if max_steps >= success_steps:
                results[i]['success'] = True
                results[i]['episode_steps'] = success_steps
            else:
                results[i]['success'] = False
                results[i]['episode_steps'] = max_steps
                
        epstep = results[i]['episode_steps']
        
        path = task_info['followed_path'][:epstep]
        results[i]['path']=path
        results[i]['shortest_path'] = shortest_path
        
        if last_agent_position is None or last_agent_position != path[-1]:
            env.step(
                action = 'Teleport',
                position=path[-1]
            )
        
            if not env.last_event.metadata['lastActionSuccess']:
                print(episode_id, "teleport failed")
                print(env.last_event.metadata['errorMessage'])
            else:
                last_agent_position = path[-1]
        
        target_metadata = get_metadata_by_objectid(env.last_event, target_object)
        dist = target_metadata['distance']
        results[i]['distance'] = dist
            
    return results       
        


def record(result_dir, data_dir, random_pose_dir=None):
    output_dir = '/data/wdz/procthor-rl-dy/record/new2'
    test_file = os.path.join(output_dir, result_dir.split('/')[-1])
    if not os.path.exists(test_file):
        os.mkdir(test_file)
    
    
    
    result_files = os.listdir(result_dir)
    results = []
    for res_file in tqdm(result_files):
        episode_info_list = json.load(open(os.path.join(result_dir, res_file),'r',encoding='utf-8'))
        for episode_info in episode_info_list:
            episode_id = episode_info['task_info']['id']
            data_id = int(episode_id.split('_')[1])
            task_id = int(episode_id.split('_')[-1])
            data_path = os.path.join(data_dir, 'data_{}'.format(data_id))
            schedules, tasks, house_info, object_poses_dict = read_data(data_path)  
            env = Dynamic_Thor(house_info=house_info, fully_scene=True, schedule_path=schedules, object_poses_dict=object_poses_dict)
            task = tasks[task_id]
            if random_pose_dir is not None:
                random_pose_path = os.path.join(random_pose_dir, f'data_{data_id}', 'random_put.json')
                random_poses = json.load(open(random_pose_path,'r'))
                event = env.set_task_setting(task, False, True, random_poses[task_id])
            else:
                env.set_task_setting(task)
            episode_results = evaluate_single_episode(episode_info, env)
            env.stop()
            # print(episode_results)
            results.append({
                'id':episode_id,
                'episode_info':episode_results
            })
            
    
    with open(os.path.join(test_file, 'results.json'),'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(len(results))        

def evaluate_ms(result_info_dir, max_step=500):
    result_info = json.load(open(os.path.join(result_info_dir, 'results.json'),'r',encoding='utf-8'))
    episode_info_list = []
    for episode in result_info:
        id = episode['id']
        episode_info = episode['episode_info']
        for ei in episode_info:
            if ei['max_steps'] == max_step:
                episode_info_list.append(ei)
    
    SR = compute_SR(episode_info_list)
    spl = compute_SPL(episode_info_list)
    ep_length_list = [ep['episode_steps'] for ep in episode_info_list]
    ave_ep_length = np.mean(ep_length_list)
    
    # noenoughsteps = [episode['episode_steps']<max_step for episode in episode_info_list]
    # nesr = np.mean(noenoughsteps)
    
    result= {
        'max_step':max_step,
        'success_rate':SR,
        'spl':spl,
        'average_ep_length':ave_ep_length
        # 'nesr':nesr,
    }
    return result
    

def evaluate():
    result_info_dirs = ['/data/wdz/procthor-rl-dy/record/new2/test_CLIP_easy_2025-03-31_03-51-05',
                        '/data/wdz/procthor-rl-dy/record/new2/test_CLIP_hard_2025-03-31_05-10-58',
                        '/data/wdz/procthor-rl-dy/record/new2/test_DINO_easy_2025-03-31_03-51-56',
                        '/data/wdz/procthor-rl-dy/record/new2/test_DINO_hard_2025-03-31_05-02-19']
    
    for result_info_dir in result_info_dirs:
        max_steps = [300, 400, 500]
        print(result_info_dir.split('/')[-1].split('_')[1:3])
        for ms in max_steps:
            res = evaluate_ms(result_info_dir, ms)
            print(res)
    
def evaluate_origin_results_info():
    result_info_dir = '/data/wdz/procthor-rl/procthor_objectnav/results/test_DINO_hard_2024-05-21_23-30-38'
    result_info_files = os.listdir(result_info_dir)
    episode_info_list = []
    for file in result_info_files:
        result_info = json.load(open(os.path.join(result_info_dir,file),'r',encoding='utf-8'))
        for ri in result_info:
            episode_info_list.append(ri)
    
    for max_step in [300, 400, 500]:
        sucess_list = [episode['oracle_success'] and episode['oracle_success_steps']<=max_step for episode in episode_info_list]
        print({
            'max_step':max_step,
            'sr':np.mean(sucess_list)
        })
        

def main():
    #compute SR, SPL, NE, Episode Length
    results_dir = 'procthor_objectnav/results'
    results_output_dir = 'record'
    
    random_pose_dirs = ['/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/random_put/sampled_scenes_2024_easy_no_entropy',
                        '/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/random_put/sampled_scenes_2024_no_entropy']
    
    result_dir_list = ['/data/wdz/procthor-rl/procthor_objectnav/results/test_DINO_easy_2024-05-21_13-50-31',
                        '/data/wdz/procthor-rl/procthor_objectnav/results/test_DINO_hard_2024-05-21_14-50-10',
                        '/data/wdz/procthor-rl/procthor_objectnav/results/test_CLIP_easy_2024-05-22_01-19-20',
                        '/data/wdz/procthor-rl/procthor_objectnav/results/test_CLIP_hard_2024-05-22_03-52-06']
    result_dir_list = ['/data/wdz/procthor-rl-dy/procthor_objectnav/results/test_CLIP_easy_2025-03-31_03-51-05',
                        '/data/wdz/procthor-rl-dy/procthor_objectnav/results/test_CLIP_hard_2025-03-31_05-10-58',
                        '/data/wdz/procthor-rl-dy/procthor_objectnav/results/test_DINO_easy_2025-03-31_03-51-56',
                        '/data/wdz/procthor-rl-dy/procthor_objectnav/results/test_DINO_hard_2025-03-31_05-02-19']
    data_dir = ['/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/sampled_scenes_2024_easy_no_entropy',
                '/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/sampled_scenes_2024_no_entropy']
    # record(result_dir_list[0], data_dir[0], random_pose_dirs[0])
    # record(result_dir_list[1], data_dir[1], random_pose_dirs[1])
    # record(result_dir_list[2], data_dir[0], random_pose_dirs[0])
    # record(result_dir_list[3], data_dir[1], random_pose_dirs[1])
    
    evaluate()
    # evaluate_origin_results_info()
    
    

    
if __name__=='__main__':
    main()   


