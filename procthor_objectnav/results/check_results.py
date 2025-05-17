import os
import json
import math
import sys
import pickle
sys.path.append('/data/wdz/Dynamic_Scene')
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


test_dir1 = './test1'
test_dir2 = './test2'

num1 = len(os.listdir(test_dir1))
num2 = len(os.listdir(test_dir2))

print(num1, num2, num1 + num2)
def vector_distance(v0, v1):
    dx = v0["x"] - v1["x"]
    dy = v0["y"] - v1["y"]
    dz = v0["z"] - v1["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def read_result(result_dirs):
    success_num = 0
    oracle_success_num = 0
    total_num = 0
    for res_dir in result_dirs:
        res_files = os.listdir(res_dir)
        for rf in res_files:
            episodes = json.load(open(os.path.join(res_dir,rf),'r',encoding='utf-8'))
            for res in episodes:
                success = res['success']
                oracle_success = res['oracle_success']
                success_num += success
                oracle_success_num += oracle_success
                total_num += 1
                
                task_info = res['task_info']
                id = task_info['id']
                if id != 'data_0_3':
                    continue
                ep_length = res['ep_length']
                path = task_info['followed_path']
                distance_to_target = res['dist_to_target']
                target = {
                            "x": 5.611671447753906,
                            "y": 0.8313186168670654,
                            "z": 25.69274139404297
                        }
                print(path[-1])
                compute_dis = vector_distance(path[-1], target)
                print(compute_dis)
                print(distance_to_target)
                
            
            
            
    return {
        'success_num':success_num,
        'success_rate':success_num / total_num,
        'oracle_success_num':oracle_success_num,
        'oracle_success_rate':oracle_success_num / total_num,
        'total_num':total_num    
    }
    

#res1 = read_result([test_dir1])
# res2 = read_result([test_dir1, test_dir2])
# print(res1)
# print(res2)
data_dir = '/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/sampled_scenes_2024_no_entropy'
test_dir = '/data/wdz/procthor-rl/procthor_objectnav/results/test_DINOCodeBook_hard_2024-05-19_13-43-18'
data_path = os.path.join(data_dir, 'data_0')

    
    
    

