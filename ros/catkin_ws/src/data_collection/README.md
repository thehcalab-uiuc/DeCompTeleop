# Instructions for Collecting Dataset from Offline Terrasentia SVO Files

[T#] represents a unique terminal open on the computer.
We assume the working directories of your terminals are at `decompteleop/ros/catkin_ws` and each terminal has the [`terra-torch3d`](../../../../README.md) Conda environment activated.

- Download SVO files collected from the Terrasentia driven under desired conditions
    - Follow the link to explore the Terrasentia Dataset [here](https://github.com/jrcuaranv/terrasentia-dataset)
- Update paths in [`svo2sync.py`](svo2sync.py) for desired `SAVE_DIR` and `FILENAME` representing where extracted data should be saved (the `FILENAME` should change for every new SVO file)
- [T1] Run `roscore`
- [T2] Run [`python src/data_collection/svo2sync.py`](svo2sync.py)
- [T3] Run [`python src/combined_node/publish_transform.py`](../combined_node/publish_transform.py)
- [T4] Run `source devel/setup.bash`
- `roslaunch` the desired downloaded SVO file to collect frames from
    - Launch with visualization from RTAB-MAP
        - [T4] Run `roslaunch rtab_mapping rtab_launch.launch rtabmap_args:="--delete_db_on_start" rgb_topic:=/zed2/zed_node/rgb/image_rect_color depth_topic:=/zed2/zed_node/depth/depth_registered camera_info_topic:=/zed2/zed_node/depth/camera_info odom_topic:=/zed2/zed_node/odom imu_topic:=/zed2/zed_node/imu/data visual_odometry:=false frame_id:=base_link approx_sync:=true rgbd_sync:=true approx_rgbd_sync:=false svo_file:="</path/to/.svo>"`
    - Launch with minimal visualization (without RTAB-MAP)
        - [T4] Run `roslaunch rtab_mapping rtab_launch_minimal.launch rtabmap_args:="--delete_db_on_start" rgb_topic:=/zed2/zed_node/rgb/image_rect_color depth_topic:=/zed2/zed_node/depth/depth_registered camera_info_topic:=/zed2/zed_node/depth/camera_info odom_topic:=/zed2/zed_node/odom imu_topic:=/zed2/zed_node/imu/data visual_odometry:=false frame_id:=base_link approx_sync:=true rgbd_sync:=true approx_rgbd_sync:=false svo_file:="</path/to/.svo>"`
- Data will be collected at the designated folder for the current playback
- Once the desired number of frames have been collected for the current SVO file, close the terminals in reverse order (T4, T3, T2, T1)
