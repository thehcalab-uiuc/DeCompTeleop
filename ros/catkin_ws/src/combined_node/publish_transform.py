#!/usr/bin/env python

import sys
import math
import rospy
from geometry_msgs.msg import TransformStamped

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import tf.transformations as tf_transformations


class StaticFramePublisher(object):

    def __init__(self):
        rospy.init_node('static_odom_pytorch3d_pulsar_tf2_broadcaster')

        self._tf_publisher = StaticTransformBroadcaster()

        # Publish static transforms once at startup
        R = rospy.Rate(150)
        while not rospy.is_shutdown():
            self.make_transforms('odom', 'pytorch3d_world', math.pi/2., 0.0, math.pi/2.)
            self.make_transforms('odom', 'pulsar_world', math.pi/2., 0.0, -math.pi/2.)
            self.make_transforms('zed2_left_camera_optical_frame', 'pytorch3d_cam', 0.0, 0.0, math.pi)
            self.make_transforms('pytorch3d_cam', 'pulsar_cam', 0.0, math.pi, 0.0)
            R.sleep()

    def make_transforms(self, base_frame, child_frame, roll, pitch, yaw):
        static_transformStamped = TransformStamped()
        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = base_frame
        static_transformStamped.child_frame_id = child_frame
        static_transformStamped.transform.translation.x = 0.0
        static_transformStamped.transform.translation.y = 0.0
        static_transformStamped.transform.translation.z = 0.0
        quat = tf_transformations.quaternion_from_euler(
            float(roll), float(pitch), float(yaw))
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        self._tf_publisher.sendTransform(static_transformStamped)

def main():
    # pass parameters and initialize node
    node = StaticFramePublisher()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
