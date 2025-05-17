
# DynamicThor Navigation Experiments

This repository provides the code and dataset for dynamic embodied navigation experiments, based on the DynamicThor framework, which extends the ProcTHOR dataset by introducing temporal dynamics.


## Environment Requirements

* Python version: `3.10`

## Creating a Conda Environment

1. Ensure that Anaconda or Miniconda is installed.

2. Create a new Conda environment with Python 3.10:

   ```bash
   conda create -n myenv python=3.10
   ```

3. Activate the new environment:

   ```bash
   conda activate myenv
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

   # Install CLIP module
   cd src/clip
   python setup.py install
   ```

## Loading the Dataset

To load a dynamic scene and its associated data, use the following function:

```python
import os
import pickle
import json

def read_data(data_path):
    with open(os.path.join(data_path, "schedules.pkl"), 'rb') as f:
        schedules = pickle.load(f)
    with open(os.path.join(data_path, "tasks.pkl"), 'rb') as f:
        tasks = pickle.load(f)
    with open(os.path.join(data_path, "house_info.json"), 'r', encoding='utf-8') as f:
        house_info = json.load(f)
    with open(os.path.join(data_path, "object_poses_dict.pkl"), 'rb') as f:
        object_poses_dict = pickle.load(f)
    
    return schedules, tasks, house_info, object_poses_dict
```

Then use it to initialize the `Dynamic_Thor` environment:

```python
from simulator.dynamic_thor import Dynamic_Thor

data_dir = 'your_data_directory/DynamicSceneDataset/dynamic_scene/sampled_scenes_hard'
i = 0  # index of the scene folder

schedules, tasks, house_info, object_poses_dict = read_data(os.path.join(data_dir, f'data_{i}'))

env = Dynamic_Thor(
    house_info=house_info,
    fully_scene=True,
    schedule_path=schedules,
    object_poses_dict=object_poses_dict
)

task = tasks[0]
env.set_task_setting(task)
```

## Running Experiments

### CLIP Experiment

```bash
python inference_dynamic.py \
    experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_codebook_ddppo_dynamic \
    experiment_model=clip_codebook \
    agent=default \
    target_object_types=robothor_habitat2022 \
    machine.num_train_processes=1 \
    machine.num_test_processes=1 \
    machine.num_test_gpus=1 \
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    seed=100 \
    eval=true \
    evaluation.tasks=["procthor-10k"] \
    evaluation.minival=false \
    data_dir='DynamicSceneDataset/dynamic_scene/sampled_scenes_normal' \
    checkpoint='ckpt/exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO-CodeBook__stage_02__steps_000420684456.pt' \
    visualize=true \
    output_dir='.'  # where to save qualitative and quantitative results
```

### DINOv2 Experiment

```bash
python inference_dynamic.py \
    experiment=procthor_objectnav/experiments/rgb_dinov2gru_codebook_ddppo_dynamic \
    experiment_model=dino_codebook \
    agent=default \
    target_object_types=robothor_habitat2022 \
    machine.num_train_processes=1 \
    machine.num_test_processes=1 \
    machine.num_test_gpus=1 \
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    seed=100 \
    eval=true \
    evaluation.tasks=["procthor-10k"] \
    evaluation.minival=false \
    data_dir='DynamicSceneDataset/dynamic_scene/sampled_scenes_normal' \
    checkpoint='ckpt/exp_ObjectNav-RGB-DINOv2GRU-DDPPO-CodeBook__stage_02__steps_000405359832.pt' \
    visualize=true \
    output_dir='.'
```

Make sure to set `checkpoint` and `data_dir` appropriately before running.

---

