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

Recording one short episode 
```
conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_test --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_test --dataset.single_task="teleop test" --dataset.num_episodes=1 --dataset.episode_time_s=10 --dataset.fps=15 --dataset.video=false --dataset.push_to_hub=false
```

Recording time-stamped episodes 
```
STAMP=$(date +%Y%m%d_%H%M%S) && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_test_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_test_${STAMP} --dataset.single_task="teleop test" --dataset.num_episodes=1 --dataset.episode_time_s=10 --dataset.fps=15 --dataset.video=false --dataset.push_to_hub=false
```

Replaying a trajectory that we recorded 
```
conda run -n policy_new --no-capture-output lerobot-replay --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --robot.use_external_commands=false --dataset.repo_id=local/rby1_test_20260418_005028 --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_test_20260418_002526 --dataset.episode=0 --dataset.fps=15
```

export STAMP=$(date +%Y%m%d_%H%M%S) && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_test_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_test_${STAMP} --dataset.single_task="teleop test" --dataset.num_episodes=1 --dataset.episode_time_s=15 --dataset.fps=15 --dataset.video=false --dataset.push_to_hub=false --dataset.prompt_before_episode=true


Record a trajectory at 10Hz duration 
```
export STAMP=$(date +%Y%m%d_%H%M%S) && conda run -n policy_new --no-capture-output lerobot-record --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 --robot.with_torso=false --robot.with_head=false --teleop.type=rby1_leader --teleop.robot_address=192.168.30.1:50051 --teleop.with_torso=false --teleop.with_head=false --dataset.repo_id=local/rby1_test_${STAMP} --dataset.root=/data/objsearch/rby1_policy_learning/datasets/local/rby1_test_${STAMP} --dataset.single_task="teleop test" --dataset.num_episodes=1 --dataset.episode_time_s=15 --dataset.fps=10 --dataset.video=false --dataset.push_to_hub=false --dataset.prompt_before_episode=true
```