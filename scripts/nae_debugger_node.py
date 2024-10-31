#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from copy import deepcopy
from visualization_msgs.msg import Marker

class NAEDebugger:
    def __init__(self, mocap_topic, impact_point_topic, z_threshold=0.1):
        # get object name from topic
        object_name = mocap_topic.split("/")[-1]
        swapper_yz_topic = "nae_debugger/mocap_data_swapper/" + object_name
        swapper_filter_yz_topic = "nae_debugger/mocap_data_swapper/filter/" + object_name

        self.mocap_sub = rospy.Subscriber(mocap_topic, PoseStamped, self.callback_swap)
        self.pred_impact_point_sub = rospy.Subscriber(impact_point_topic, PoseStamped, self.callback_pred_impact_point)

        self.cur_pose_pub = rospy.Publisher(swapper_yz_topic, PoseStamped, queue_size=1)
        self.cur_pose_filtered_pub = rospy.Publisher(swapper_filter_yz_topic, PoseStamped, queue_size=1)
        self.error_marker_pub = rospy.Publisher("nae_debugger/prediction_error_marker", Marker, queue_size=1)

        self.cur_pose_filtered = None
        self.impact_point_pred = None

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

        # khi cur z < self.z_threshold nhưng cur_pose_filtered đã được lưu trước đó có z >= self.z_threshold
        # thì publish cur_pose_filtered
        if cur_pose.pose.position.z >= self.z_threshold:
            # Lưu cur_pose làm cur_pose_filtered nếu z >= self.z_threshold
            self.cur_pose_filtered = cur_pose
        
        # Publish cur_pose_filtered nếu đã có giá trị
        if self.cur_pose_filtered is not None:
            pub_error = cur_pose.pose.position.z < self.z_threshold
            self.publish_real_impact_point_with_error(pub_error = pub_error)
        self.cur_pose_pub.publish(cur_pose)
        
    def callback_pred_impact_point(self, msg: PoseStamped):
        self.impact_point_pred = msg
    
    def publish_real_impact_point_with_error(self, pub_error=True):
        # calculate prediction distance between predicted impact point and filtered current position (x, y, z)
        error_value = 0
        if self.impact_point_pred is not None and self.cur_pose_filtered is not None:
            error_value = ((self.impact_point_pred.pose.position.x - self.cur_pose_filtered.pose.position.x) ** 2 + 
                           (self.impact_point_pred.pose.position.y - self.cur_pose_filtered.pose.position.y) ** 2 +
                           (self.impact_point_pred.pose.position.z - self.cur_pose_filtered.pose.position.z) ** 2) ** 0.5

        # Publish predicted position
        self.cur_pose_filtered_pub.publish(self.cur_pose_filtered)
        # Create a Marker to display error as text
        error_marker = Marker()
        error_marker.header.frame_id = self.cur_pose_filtered.header.frame_id
        error_marker.header.stamp = rospy.Time.now()
        error_marker.ns = "prediction_error"
        error_marker.id = 0
        error_marker.type = Marker.TEXT_VIEW_FACING
        error_marker.action = Marker.ADD

        # Set the position of the error marker (slightly offset from the predicted position)
        error_marker.pose.position.x = self.cur_pose_filtered.pose.position.x + 0.5  # Offset on x-axis for visibility
        error_marker.pose.position.y = self.cur_pose_filtered.pose.position.y
        error_marker.pose.position.z = self.cur_pose_filtered.pose.position.z + 0.5  # Offset on z-axis for visibility

        # Display the error value as text
        error_marker.text = f"Error: {error_value:.2f}"

        # Customize appearance (scale, color, etc.)
        error_marker.scale.z = 0.1  # Text height
        # green color
        error_marker.color.r = 0.0
        error_marker.color.g = 1.0
        error_marker.color.b = 0.0
        error_marker.color.a = 1.0

        # Publish the error marker
        if pub_error:
            self.error_marker_pub.publish(error_marker)
        print(f"Error value: {error_value}")

if __name__ == '__main__':
    rospy.init_node('nae_debugger_node')
    mocap_topic = rospy.get_param("~mocap_topic", "/mocap_pose_topic/frisbee1_pose")
    impact_point_topic = rospy.get_param("~impact_point_topic", "/NAE/impact_point")
    NAEDebugger(mocap_topic, impact_point_topic, z_threshold=0.2)
    rospy.spin()