## Instructions for collecting data and training a policy on the RB-Y1

### Instructions for teleoperating the robot
Start the server in teleop mode using the command 

```
sudo .venv/bin/python -m rby1_standalone.arms_teleop --address 192.168.30.1:50051
```

Testing the teleop
```
conda run -n policy_new --no-capture-output lerobot-teleoperate \
--robot.type=rby1 \
--robot.robot_address=192.168.30.1:50051 \
--robot.with_torso=false \
--robot.with_head=false \
--teleop.type=rby1_leader \
--teleop.robot_address=192.168.30.1:50051 \
--teleop.with_torso=false \
--teleop.with_head=false 
```

Record a trajectory at 10Hz duration using teleop
```
export STAMP=$(date +%Y%m%d_%H%M%S) && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_test_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_test_${STAMP} --dataset.single_task="teleop test" --dataset.num_episodes=1 --dataset.episode_time_s=15 --dataset.fps=10 --dataset.video=false --dataset.push_to_hub=false --dataset.prompt_before_episode=true
```

Playback a trajectory recorded using teleop 
```
conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_test_20260418_022626 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_test_20260418_022626 --dataset.episode=0 --dataset.fps=10  
```

Record the episode with grippers enabled 
```
export STAMP=$(date +%Y%m%d_%H%M%S) && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_${STAMP} --dataset.single_task="describe task here" --dataset.num_episodes=1 --dataset.episode_time_s=15 --dataset.fps=10 --dataset.video=false --dataset.push_to_hub=false --dataset.prompt_before_episode=true
```

Playback the trajectory with grippers enabled
```
conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_${STAMP} --dataset.episode=0 --dataset.fps=10
```


### Collecting data with Zed data streaming mode enabled 

> **Color convention note (2026-04-21):** the ZMQ camera consumer
> (`src/lerobot/cameras/zmq/camera_zmq.py`) now does a `BGR → RGB` conversion
> after `cv2.imdecode`, so images recorded via ZMQ are stored as RGB.
> Datasets recorded **before** this change (e.g. `rby1_pick_v2_20260419_180507`)
> contain channel-swapped (BGR-as-RGB) images. Checkpoints trained on pre-fix
> data — including `act_rby1_pick_v2_20260419_180507` — must be re-trained on
> newly recorded data before they can be used for live inference, otherwise
> there will be a color distribution shift between train and deploy.

Start the zed streaming / publishing 
```
# start the zed publisher / streaming mode 
/data/objsearch/rby1_policy_learning/scripts/rby1/start_zed_publisher.sh           # 10 fps

# start the master arm teleop server
sudo .venv/bin/python -m rby1_standalone.arms_teleop --address 192.168.30.1:50051

# record the data 
export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.cameras="$CAMS" --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_cam_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_cam_${STAMP} --dataset.single_task="cameras test" --dataset.num_episodes=1 --dataset.episode_time_s=10 --dataset.fps=10 --dataset.video=false --dataset.push_to_hub=false --dataset.prompt_before_episode=true

# running the record step with the video encoding and saving enabled 

export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.cameras="$CAMS" --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_cam_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_cam_${STAMP} --dataset.single_task="cameras test" --dataset.num_episodes=1 --dataset.episode_time_s=10 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=true

# Do a trajectory replay of the latest dataset that we recorded 

conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_cam_20260418_204936 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_cam_20260418_204936 --dataset.episode=0 --dataset.fps=10
```

### Saving data for training ACT policies
```
# Quick sanity check by recording 5 episodes 

export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.cameras="$CAMS" --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_pick_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=5 --dataset.episode_time_s=25 --dataset.reset_time_s=10 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=true
```

Playing back a single episode from the recorded trajectories 
```
# Ensure that the master teleop setup is stopped before running the replay

conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=0 --dataset.fps=10
```

```
# Specify a different ampere current value for the gripper 

conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.gripper_current_a=12 --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10
```

conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10 


conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10


conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10 


conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.gripper_current_a=12 --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10 


conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10 


conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10

conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.gripper_close_current=10 --robot.gripper_open_current=10 --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10 

conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_pick_20260418_225702 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_20260418_225702 --dataset.episode=2 --dataset.fps=10 

export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.cameras="$CAMS" --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_pick_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=25 --dataset.episode_time_s=20 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=true


export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_inference --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.cameras="$CAMS" --policy.path=/data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_20260419_020813/checkpoints/last/pretrained_model --policy.device=cuda --dataset.repo_id=local/eval_rby1_pick_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/eval_rby1_pick_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=1 --dataset.episode_time_s=30 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=true


export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_inference --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.cameras="$CAMS" --policy.path=/data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_20260419_020813/checkpoints/last/pretrained_model --policy.device=cuda --dataset.repo_id=local/eval_rby1_pick_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/eval_rby1_pick_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=false


## Collecting 50 episodes on the robot 
export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.cameras="$CAMS" --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_pick_v2_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_v2_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=50 --dataset.episode_time_s=30 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=true


## Command for starting the policy training
WANDB_API_KEY=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 CUDA_VISIBLE_DEVICES=1 conda run -n policy_new --no-capture-output lerobot-train --dataset.repo_id=local/rby1_pick_v2_20260419_180507 --dataset.root=/home/scenecomplete/research/rby1_policy_training/datasets/local/rby1_pick_v2_20260419_180507 --policy.type=act --policy.device=cuda --policy.push_to_hub=false --output_dir=/home/scenecomplete/research/rby1_policy_training/outputs/train/act_rby1_pick_v2_20260419_180507 --job_name=act_rby1_pick_v2 --steps=150000 --batch_size=16 --save_checkpoint=true --save_freq=10000 --wandb.enable=true

export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_inference --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.cameras="$CAMS" --policy.path=/data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_v2_20260419_180507/checkpoints/last/pretrained_model --policy.device=cuda --dataset.repo_id=local/eval_rby1_pick_v2_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/eval_rby1_pick_v2_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=1 --dataset.episode_time_s=120 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=false

export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_inference --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.cameras="$CAMS" --policy.path=/data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_v2_20260419_180507/checkpoints/last/pretrained_model --policy.device=cuda --dataset.repo_id=local/eval_rby1_pick_v2_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/eval_rby1_pick_v2_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=1 --dataset.episode_time_s=150 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=false

export STAMP=$(date +%Y%m%d_%H%M%S) && export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}' && conda run -n policy_inference --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --robot.cameras="$CAMS" --policy.path=/data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_v2_20260419_180507/checkpoints/70000/pretrained_model --policy.device=cuda --dataset.repo_id=local/eval_rby1_pick_v2_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/eval_rby1_pick_v2_${STAMP} --dataset.single_task="pick block place in bowl" --dataset.num_episodes=1 --dataset.episode_time_s=150 --dataset.fps=10 --dataset.push_to_hub=false --dataset.prompt_before_episode=false
