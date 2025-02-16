# Instructions for Running Real-Time Delay Compensation ROS Node

[T#] represents a unique terminal open on the computer.
We assume the working directories of your terminals are at `decompteleop/ros/catkin_ws` and each terminal has the [`terra-torch3d`](../../../../README.md) Conda environment activated.

- Download SVO files collected from the Terrasentia driven under desired conditions
    - Follow the link to explore the Terrasentia Dataset [here](https://github.com/jrcuaranv/terrasentia-dataset)
- Update paths in [`decomp_render.py`](decomp_render.py) for pretrained depth and inpainting checkpoints, as well as arguments for delay and low framerate emulation
- [T1] Run `roscore`
- [T2] Run [`python src/combined_node/decomp_render.py`](decomp_render.py)
- [T3] Run [`python src/combined_node/publish_transform.py`](publish_transform.py)
- [T4] Run `source devel/setup.bash`
- `roslaunch` the desired downloaded SVO file to compensate for
    - Launch with visualization from RTAB-MAP
        - [T4] Run `roslaunch rtab_mapping rtab_launch.launch rtabmap_args:="--delete_db_on_start" rgb_topic:=/zed2/zed_node/rgb/image_rect_color depth_topic:=/zed2/zed_node/depth/depth_registered camera_info_topic:=/zed2/zed_node/depth/camera_info odom_topic:=/zed2/zed_node/odom imu_topic:=/zed2/zed_node/imu/data visual_odometry:=false frame_id:=base_link approx_sync:=true rgbd_sync:=true approx_rgbd_sync:=false svo_file:="</path/to/.svo>"`
    - Launch with minimal visualization (without RTAB-MAP)
        - [T4] Run `roslaunch rtab_mapping rtab_launch_minimal.launch rtabmap_args:="--delete_db_on_start" rgb_topic:=/zed2/zed_node/rgb/image_rect_color depth_topic:=/zed2/zed_node/depth/depth_registered camera_info_topic:=/zed2/zed_node/depth/camera_info odom_topic:=/zed2/zed_node/odom imu_topic:=/zed2/zed_node/imu/data visual_odometry:=false frame_id:=base_link approx_sync:=true rgbd_sync:=true approx_rgbd_sync:=false svo_file:="</path/to/.svo>"`
- Once done visualizing delay compensation, close the terminals in reverse order (T4, T3, T2, T1)
