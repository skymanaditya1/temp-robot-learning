## Data recording 

```
# Start the zed streaming
/data/objsearch/rby1_policy_learning/scripts/rby1/start_zed_publisher.sh
```

```
# Start the master arm teleop 
cd ~/LIS_ws
sudo .venv/bin/python -m rby1_standalone.arms_teleop --address 192.168.30.1:50051
```

```
# Start the data collection (with 1 episode - smoke test)
cd /data/objsearch/rby1_policy_learning
conda activate policy_inference 
scripts/rby1/record_dataset.sh rby1_pick_v3 1 30
```

```
# Test the replay trajectory (runs playback on the robot and visualizes images in rerun)
# we specify the episode number and whether we want to run the robot or not
./scripts/rby1/replay_dataset_rerun.sh rby1_pick_v3_20260422_154546 0 true
```

```
# Command for recording an entire dataset of recording on the robot
scripts/rby1/record_dataset.sh rby1_pick_v3 50 30
```

```
# Command for starting the training (bash script)
scripts/rby1/train_policy.sh <dataset_name> <tag_name> <num_steps> <save_steps> <kl_divergence>

# With W&B enabled on GPU 0
scripts/rby1/train_policy.sh rby1_pick_v3_20260422_174437 act_rby1_pick_v3 300000 10000 1
```

```
# Testing the overfit inference on the train dataset with a trained policy
scripts/rby1/overfit_inference_plot.sh <episode_number> <dataset_name> <checkpoint_path> <checkpoint_step>

# An example command for running the overfit inference
scripts/rby1/overfit_inference_plot.sh 0 rby1_pick_v3_20260422_174437 /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v3_20260422_174437_act_vega last
```

```
# Command for starting inference (bash script)

scripts/rby1/rollout_policy.sh /data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_v3_rby1_pick_v3_20260422_174437 last
```