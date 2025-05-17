import torch
import os
import sys
import attr
import json
import pickle
from tqdm import tqdm
from typing import Optional, cast, Tuple, Dict
import copy

from allenact.utils.experiment_utils import set_seed
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.misc import (
    Memory,
    ObservationType,
    ActorCriticOutput,
    DistributionType,
)
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.utils.tensor_utils import batch_observations
from allenact.utils import spaces_utils as su

from procthor_objectnav.experiments.rgb_clipresnet50gru_ddppo_dynamic import DynamicProcTHORObjectNavRGBClipResNet50PPOExperimentConfig
from procthor_objectnav.experiments.rgb_clipresnet50gru_codebook_ddppo_dynamic import DynamicProcTHORObjectNavRGBClipResNet50PPOCodeBookExperimentConfig
from procthor_objectnav.experiments.rgb_dinov2gru_codebook_ddppo_dynamic import DynamicProcTHORObjectNavRGBDINOv2PPOCodebookExperimentConfig

import hydra
import prior
import importlib
import inspect
import ast
from omegaconf import DictConfig
from allenact.main import load_config, _config_source, find_sub_modules
from allenact.utils.system import get_logger
import time

if "PROCTHOR_HYDRA_CONFIG_DIR" not in os.environ:
    os.environ["PROCTHOR_HYDRA_CONFIG_DIR"] = os.path.join(os.getcwd(), "config")
else:
    os.environ["PROCTHOR_HYDRA_CONFIG_DIR"] = os.path.abspath(
        os.environ["PROCTHOR_HYDRA_CONFIG_DIR"]
    )
    

from simulator.dynamic_thor import Dynamic_Thor

set_seed(1)

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

def init_config(cfg: DictConfig) -> DictConfig:
    print(cfg)

    # NOTE: Support loading in model from prior
    allenact_checkpoint = None
    if cfg.checkpoint is not None and cfg.pretrained_model.name is not None:
        raise ValueError(
            f"Cannot specify both checkpoint {cfg.checkpoint}"
            f" and prior_checkpoint {cfg.pretrained_model.name}"
        )
    elif cfg.checkpoint is None and cfg.pretrained_model.name is not None:
        cfg.checkpoint = prior.load_model(
            project=cfg.pretrained_model.project, model=cfg.pretrained_model.name
        )

    return cfg


def load_config(args) -> Tuple[ExperimentConfig, Dict[str, str]]:
    assert os.path.exists(
        args.experiment_base
    ), "The path '{}' does not seem to exist (your current working directory is '{}').".format(
        args.experiment_base, os.getcwd()
    )
    rel_base_dir = os.path.relpath(  # Normalizing string representation of path
        os.path.abspath(args.experiment_base), os.getcwd()
    )
    rel_base_dot_path = rel_base_dir.replace("/", ".")
    if rel_base_dot_path == ".":
        rel_base_dot_path = ""

    exp_dot_path = args.experiment
    if exp_dot_path[-3:] == ".py":
        exp_dot_path = exp_dot_path[:-3]
    exp_dot_path = exp_dot_path.replace("/", ".")

    module_path = (
        f"{rel_base_dot_path}.{exp_dot_path}"
        if len(rel_base_dot_path) != 0
        else exp_dot_path
    )

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
            print('error: ',e.args)
            print(module_path)
            raise e
        all_sub_modules = set(find_sub_modules(os.getcwd()))
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            f"Could not import experiment '{module_path}', are you sure this is the right path?"
            f" Possibly relevant files include {relevant_submodules}."
            f" Note that the experiment must be reachable along your `PYTHONPATH`, it might"
            f" be helpful for you to run `export PYTHONPATH=$PYTHONPATH:$PWD` in your"
            f" project's top level directory."
        ) from e

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config_kwargs = {}
    if args.config_kwargs is not None:
        if os.path.exists(args.config_kwargs):
            with open(args.config_kwargs, "r") as f:
                config_kwargs = json.load(f)
        else:
            try:
                config_kwargs = json.loads(args.config_kwargs)
            except json.JSONDecodeError:
                get_logger().warning(
                    f"The input for --config_kwargs ('{args.config_kwargs}')"
                    f" does not appear to be valid json. Often this is due to"
                    f" json requiring very specific syntax (e.g. double quoted strings)"
                    f" we'll try to get around this by evaluating with `ast.literal_eval`"
                    f" (a safer version of the standard `eval` function)."
                )
                config_kwargs = ast.literal_eval(args.config_kwargs)

        assert isinstance(
            config_kwargs, Dict
        ), "`--config_kwargs` must be a json string (or a path to a .json file) that evaluates to a dictionary."

    config = experiments[0](cfg=args, **config_kwargs)
    sources = _config_source(config_type=experiments[0])
    CONFIG_KWARGS_STR = "__CONFIG_KWARGS__"
    sources[CONFIG_KWARGS_STR] = json.dumps(config_kwargs)
    return config, sources

@attr.s(kw_only=True)
class InferenceAgent:
    actor_critic: ActorCriticModel = attr.ib()
    rollout_storage: RolloutStorage = attr.ib()
    device: torch.device = attr.ib()
    sensor_preprocessor_graph: Optional[SensorPreprocessorGraph] = attr.ib()
    steps_before_rollout_refresh: int = attr.ib(default=128)
    memory: Optional[Memory] = attr.ib(default=None)
    steps_taken_in_task: int = attr.ib(default=0)
    last_action_flat= attr.ib(default=None)
    

    def __attrs_post_init__(self):
        self.actor_critic.eval()
        self.actor_critic.to(device=self.device)
        if self.memory is not None:
            self.memory.to(device=self.device)
        if self.sensor_preprocessor_graph is not None:
            self.sensor_preprocessor_graph.to(self.device)

        self.rollout_storage.to(self.device)
        self.rollout_storage.set_partition(index=0, num_parts=1)

    @classmethod
    def from_experiment_config(
        cls,
        exp_config: ExperimentConfig,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
    ):
        rollout_storage = exp_config.training_pipeline().rollout_storage

        machine_params = exp_config.machine_params("test")
        if not isinstance(machine_params, MachineParams):
            machine_params = MachineParams(**machine_params)

        sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph

        actor_critic = cast(
            ActorCriticModel,
            exp_config.create_model(
                sensor_preprocessor_graph=sensor_preprocessor_graph
            ),
        )

        if checkpoint_path is not None:
            actor_critic.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
            )

        return cls(
            actor_critic=actor_critic,
            rollout_storage=rollout_storage,
            device=device,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    def reset(self):
        self.steps_taken_in_task = 0
        self.memory = None

    def act(self, observations: ObservationType):
        # Batch of size 1
        obs_batch = batch_observations([observations], device=self.device)
        if self.sensor_preprocessor_graph is not None:
            obs_batch = self.sensor_preprocessor_graph.get_observations(obs_batch)

        if self.steps_taken_in_task == 0:
            self.rollout_storage.initialize(
                observations=obs_batch,
                num_samplers=1,
                recurrent_memory_specification=self.actor_critic.recurrent_memory_specification,
                action_space=self.actor_critic.action_space,
            )
            self.rollout_storage.after_updates()
        else:
            dummy_val = torch.zeros((1, 1), device=self.device)  # Unused dummy value
            self.rollout_storage.add(
                observations=obs_batch,
                memory=self.memory,
                actions=self.last_action_flat[0],
                action_log_probs=dummy_val,
                value_preds=dummy_val,
                rewards=dummy_val,
                masks=torch.ones(
                    (1, 1), device=self.device
                ),  # Always == 1 as we're in a single task until `reset`
            )

        agent_input = self.rollout_storage.agent_input_for_next_step()

        actor_critic_output, self.memory = cast(
            Tuple[ActorCriticOutput[DistributionType], Optional[Memory]],
            self.actor_critic(**agent_input),
        )

        action = actor_critic_output.distributions.sample()
        self.last_action_flat = su.flatten(self.actor_critic.action_space, action)

        self.steps_taken_in_task += 1

        if self.steps_taken_in_task % self.steps_before_rollout_refresh:
            self.rollout_storage.after_updates()

        return su.action_list(self.actor_critic.action_space, self.last_action_flat)[0]



def test_rgb_clipresnet50gru_codebook_ddppo(cfg):
    
    
    exp_config = DynamicProcTHORObjectNavRGBClipResNet50PPOCodeBookExperimentConfig(cfg)
    
    ckpt = cfg.checkpoint
    agent = InferenceAgent.from_experiment_config(
            exp_config=exp_config, device=torch.device("cuda"),checkpoint_path=ckpt
        )

    task_sampler = exp_config.make_sampler_fn(
        exp_config.test_task_sampler_args(process_ind=0, total_processes=1)["task_sampler_args"]
    )
    
    data_dirs = [cfg.data_dir]
    for level_id, data_dir in enumerate(data_dirs):
        data_num = len(os.listdir(data_dir))
        timestamp = time.time()
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp))
        # result_output_dir = '/data/wdz/procthor-rl/procthor_objectnav/results/test_{}_{}_{}'.format('CLIP', level, current_time)
        result_output_dir = os.path.join(cfg.output_dir, 'test_{}_{}'.format('CLIP', current_time))
        if not os.path.exists(result_output_dir):
            os.mkdir(result_output_dir)
        total_num = 0
        for i in tqdm(range(data_num)):
            
            schedules, tasks, house_info, object_poses_dict = read_data(os.path.join(data_dir, 'data_{}'.format(i)))
            oracle_success = False
            oracle_success_steps = -1
            
            data_results = []
            for j, task in enumerate(tasks):
                oracle_success = False
                oracle_success_steps = -1
                task_obj_type = task['target_object_type']
                if not task_obj_type in cfg.target_object_types:
                    continue
                agent.reset()
                env = Dynamic_Thor(house_info=house_info, fully_scene=True, schedule_path=schedules, object_poses_dict=object_poses_dict)
                navtask = task_sampler.next_task2(env, task)
                observations = navtask.get_observations()
                actions = []
                while not navtask.is_done():
                    action = agent.act(observations=observations)
                    observations = navtask.step(action).observation
                    actions.append(action)
                    
                    if navtask._is_goal_in_range() and not oracle_success:
                        oracle_success_steps = navtask.num_steps_taken()
                        oracle_success = True
                
                print(actions)
                metrics = navtask.metrics()
                metrics['oracle_success'] = oracle_success
                metrics['oracle_success_steps'] = oracle_success_steps
                metrics['task_info']['id'] = 'data_{}_{}'.format(i,j)
                total_num += 1
                
                data_results.append(metrics)
            
                env.stop()
                # break
            
            
            with open(os.path.join(result_output_dir,'data_{}.json'.format(i)), 'w',encoding='utf-8') as f:
                    json.dump(data_results, f, indent=4)

        
    print('total num: ', total_num)

def test_rgb_dinoresnet50gru_codebook_ddppo(cfg):
    
    
    exp_config = DynamicProcTHORObjectNavRGBDINOv2PPOCodebookExperimentConfig(cfg)
    ckpt = cfg.checkpoint
    agent = InferenceAgent.from_experiment_config(
            exp_config=exp_config, device=torch.device("cuda"),checkpoint_path=ckpt
        )

    task_sampler = exp_config.make_sampler_fn(
        exp_config.test_task_sampler_args(process_ind=0, total_processes=1)["task_sampler_args"]
    )
    
    data_dirs = [cfg.data_dir]
    for level_id, data_dir in enumerate(data_dirs):
        data_num = len(os.listdir(data_dir))
        timestamp = time.time()
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(timestamp))
     
        result_output_dir = os.path.join(cfg.output_dir, 'test_{}_{}'.format('DINO', current_time))
        if not os.path.exists(result_output_dir):
            os.mkdir(result_output_dir)
        total_num = 0

        for i in tqdm(range(data_num)):
            schedules, tasks, house_info, object_poses_dict = read_data(os.path.join(data_dir, 'data_{}'.format(i)))
            oracle_success = False
            oracle_success_steps = -1
            
            data_results = []
            for j, task in enumerate(tasks):
                oracle_success = False
                oracle_success_steps = -1
                task_obj_type = task['target_object_type']
                if not task_obj_type in cfg.target_object_types:
                    continue
                agent.reset()
                env = Dynamic_Thor(house_info=house_info, fully_scene=True, schedule_path=schedules, object_poses_dict=object_poses_dict)
                navtask = task_sampler.next_task2(env, task)
                observations = navtask.get_observations()
                actions = []
                while not navtask.is_done():
                    action = agent.act(observations=observations)
                    observations = navtask.step(action).observation
                    actions.append(action)
                    
                    if navtask._is_goal_in_range() and not oracle_success:
                        oracle_success_steps = navtask.num_steps_taken()
                        oracle_success = True
                
                print(actions)
                metrics = navtask.metrics()
                metrics['oracle_success'] = oracle_success
                metrics['oracle_success_steps'] = oracle_success_steps
                metrics['task_info']['id'] = 'data_{}_{}'.format(i,j)
                total_num += 1
                
                data_results.append(metrics)
            
                env.stop()
                # break
            
            
            with open(os.path.join(result_output_dir,'data_{}.json'.format(i)), 'w',encoding='utf-8') as f:
                    json.dump(data_results, f, indent=4)

    
        # print('total num: ', total_num)
   
   
    
@hydra.main(config_path=os.environ["PROCTHOR_HYDRA_CONFIG_DIR"], config_name="main")
def main(cfg: DictConfig) -> None:
    
    cfg = init_config(cfg=cfg)

    exp_cfg, srcs = load_config(cfg)
    #test_rgb_clipresnet50gru_ddppo(cfg)
    #test_rgb_clipresnet50gru_ddppo_unseen_obj_class(cfg)
    experiment_model = cfg.experiment_model
    if experiment_model == 'clip_codebook':
        test_rgb_clipresnet50gru_codebook_ddppo(cfg)
    elif experiment_model == 'dino_codebook':
        test_rgb_dinoresnet50gru_codebook_ddppo(cfg)
    

if __name__=='__main__':
    sys.argv.append("hydra.run.dir=./")
    main()

