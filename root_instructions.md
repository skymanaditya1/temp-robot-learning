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
./scripts/rby1/replay_dataset_rerun.sh rby1_pick_v3_20260422_154546 5 true

```