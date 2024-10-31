#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from copy import deepcopy

class MocapSwapperYZ:
    def __init__(self, mocap_topic, impact_point_topic, z_threshold=0.1):
        # get object name from topic
        object_name = mocap_topic.split("/")[-1]
        swapper_yz_topic = "NAE/mocap_data_swapper/" + object_name
        swapper_filter_yz_topic = "NAE/mocap_data_swapper/filter/" + object_name

        self.mocap_subscriber = rospy.Subscriber(mocap_topic, PoseStamped, self.callback_swap)
        self.impact_point_subscriber = rospy.Subscriber(impact_point_topic, PoseStamped, self.callback_impact_point)
        self.swapper_pub = rospy.Publisher(swapper_yz_topic, PoseStamped, queue_size=1)
        self.swapper_filter_pub = rospy.Publisher(swapper_filter_yz_topic, PoseStamped, queue_size=1)

        self.filtered_pose = None
        self.impact_point = None

        self.z_threshold = z_threshold

    def callback_swap(self, msg: PoseStamped):
        # Tạo bản sao của msg để không thay đổi msg gốc
        new_msg = deepcopy(msg)
        
        # Swap y và z trong new_msg
        y = msg.pose.position.y
        z = msg.pose.position.z
        new_msg.pose.position.y = z
        new_msg.pose.position.z = y
        cur_pose = new_msg

        # khi cur z < self.z_threshold nhưng filtered_pose đã được lưu trước đó có z >= self.z_threshold
        # thì publish filtered_pose
        if cur_pose.pose.position.z >= self.z_threshold:
            # Lưu cur_pose làm filtered_pose nếu z >= self.z_threshold
            self.filtered_pose = cur_pose
        
        # Publish filtered_pose nếu đã có giá trị
        if self.filtered_pose is not None:
            self.swapper_filter_pub.publish(self.filtered_pose)
        self.swapper_pub.publish(cur_pose)
    def callback_impact_point(self, msg: PoseStamped):
        self.impact_point = msg
        # calculate distance between impact point and filtered current position (x, y, z)
        error_value = 0
        if self.impact_point is not None and self.filtered_pose is not None:
            error_value = ((self.impact_point.pose.position.x - self.filtered_pose.pose.position.x) ** 2 + 
                           (self.impact_point.pose.position.y - self.filtered_pose.pose.position.y) ** 2 +
                           (self.impact_point.pose.position.z - self.filtered_pose.pose.position.z) ** 2) ** 0.5
        rospy.loginfo(f"Error value: {error_value}")
        
if __name__ == '__main__':
    rospy.init_node('moca_data_swapper_node')
    mocap_topic = rospy.get_param("~mocap_topic", "/mocap_pose_topic/frisbee1_pose")
    impact_point_topic = rospy.get_param("~impact_point_topic", "/NAE/impact_point")
    MocapSwapperYZ(mocap_topic, impact_point_topic, z_threshold=0.2)
    rospy.spin()