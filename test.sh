python inference.py \
    experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_ddppo \
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
    checkpoint='ckpt/exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt' \
    visualize=true \
    output_dir='procthor_objectnav/results/test1' # dir to store both qualitative and quantitative results
