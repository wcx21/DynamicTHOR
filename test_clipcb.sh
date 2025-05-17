python inference_dynamic.py \
    experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_codebook_ddppo_dynamic \
    experiment_model=clip_codebook\
    agent=default \
    target_object_types=robothor_habitat2022 \
    machine.num_train_processes=1 \
    machine.num_test_processes=1 \
    machine.num_test_gpus=1\
    ai2thor.platform=CloudRendering \
    model.add_prev_actions_embedding=true \
    seed=100 \
    eval=true \
    evaluation.tasks=["procthor-10k"] \
    evaluation.minival=false \
    data_dir='/data/wdz/Dynamic_Scene/data/grounding_to_scene_2024_data/sampled_scenes_2024_easy_no_entropy'\
    checkpoint='ckpt/exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO-CodeBook__stage_02__steps_000420684456.pt' \
    visualize=true \
    output_dir='.' # dir to store both qualitative and quantitative results
