'''
This is a ROS node that:
    - subscribes to mocap topic to get position of the target object and predict its trajectory.
    - publishes the predicted trajectory to a topic.
'''

import rospy
import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from nae_online_prediction.msg import PointArray
from nae.nae import *
from nae.utils.submodules.printer import Printer
import threading

class HistoricalData:
    def __init__(self, storage_max_len=300):
        self.data_points = []
        self.time_stamps = []
        self.acc = [0, 0, 9.81]

        self.storage_max_len = storage_max_len

    def append(self, new_pos, new_time_stamp):
        # interpolate velocity, update velocity of the last data point and new data point
        if len(self.data_points) == 0:
            # append new data point
            new_vel = [0, 0, 0]
        elif len(self.data_points) == 1:
            # update velocity of the last data point based on forward difference method
            last_vel_update = (new_pos - self.data_points[-1][:3]) / (new_time_stamp - self.time_stamps[-1])
            self.data_points[-1][3:6] = last_vel_update

            # calculate velocity of the new data point based on backward difference method
            new_vel = last_vel_update
        else:
            # update velocity of the last data point based on central difference method
            last_vel_update = (new_pos - self.data_points[-2][:3]) / (new_time_stamp - self.time_stamps[-2])
            self.data_points[-1][3:6] = last_vel_update

            # calculate velocity of the new data point based on backward difference method
            try:
                new_vel = (new_pos - self.data_points[-1][:3]) / (new_time_stamp - self.time_stamps[-1])
            except ZeroDivisionError:
                rospy.logwarn('[HISTORICAL-DATA] Zero division error')
                return
            # clear oldest data points if the length of data_points exceeds storage_max_len
            self.clear_oldest_data_points()

        # append new data point
        new_data_point = np.concatenate([new_pos, new_vel, self.acc])
        self.data_points.append(new_data_point)
        self.time_stamps.append(new_time_stamp)

        # Xác nhận độ dài danh sách
        assert len(self.data_points) == len(self.time_stamps), 'Data points and time stamps have different lengths'

    # clear oldest data points if the length of data_points exceeds storage_max_len
    def clear_oldest_data_points(self,):
        if len(self.data_points) > self.storage_max_len:
            self.data_points = self.data_points[-self.storage_max_len:]
            self.time_stamps = self.time_stamps[-self.storage_max_len:]
        
    def get_data(self):
        return np.array(self.data_points)
    
    def reset(self):
        self.data_points = []
        self.time_stamps = []

class NAEOnlinePredictor:
    def __init__(self, model_path, model_params, training_params, prediction_params, mocap_topic, active_range_x=[0, 10000], active_range_y=[0, 10000], active_range_z=[0.1, 10000], swap_y_z=False):
        # Load model
        self.nae = NAE(**model_params, **training_params, device=DEVICE)
        self.nae.load_model(model_path)
        self.printer = Printer()

        self.input_len_req = training_params['input_len']
        self.future_pred_len = training_params['future_pred_len']
        self.auto_agressive_len = prediction_params['auto_agressive_len']
        self.active_range_x = active_range_x
        self.active_range_y = active_range_y
        self.active_range_z = active_range_z

        self.swap_y_z = swap_y_z
        # Subscribe to mocap topic
        self.mocap_sub = rospy.Subscriber(mocap_topic, PoseStamped, self.mocap_callback)
        # Publish predicted trajectory
        self.predicted_traj_pub = rospy.Publisher('NAE/predicted_traj', PointArray, queue_size=1)
        self.impact_point_pub = rospy.Publisher('NAE/impact_point', Point, queue_size=1)

        # Variables
        self.pred_seq = None

        # locks
        # self.enable_get_mocap_data_lock = threading.Lock()
        # self.enable_get_mocap_data = False

        self.historical_data_lock = threading.Lock()
        self.historical_data = HistoricalData(prediction_params['storage_max_len'])

        # event
        self.pressed_enter_event = threading.Event()

        # Thread to predict
        self.predict_thread = threading.Thread(target=self.online_predict)
        self.predict_thread.setDaemon(True)
        self.predict_thread.start()
    
    '''
    function mocap_callback:
        function: 
            + Collect position of the target object and append to historical data, no prediction
            + swap y and z axis if needed
    '''
    def mocap_callback(self, msg:PoseStamped):
        name = '[MC-CALLBACK] '
        # Get position of the target object
        if self.swap_y_z:
            curr_pos = [msg.pose.position.x, msg.pose.position.z, msg.pose.position.y]
        else:
            curr_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        
        if not self.pressed_enter_event.is_set():
            # rospy.logwarn(name + 'Disable getting mocap data')
            # skip this callback loop if the enter key is not pressed
            return
        
        # check current position is in active range
        if curr_pos[0] < self.active_range_x[0] or curr_pos[0] > self.active_range_x[1] or \
            curr_pos[1] < self.active_range_y[0] or curr_pos[1] > self.active_range_y[1]:
            rospy.logwarn(name + 'Current position is out of active range')
            return
    
        # TODO: Using queue to store data, prevent data loss
        if not self.historical_data_lock.acquire(blocking=False):
            # TODO: print in red color
            rospy.logwarn(name + 'Cannot get lock, data loss')
            return
        try:
            self.historical_data.append(curr_pos, msg.header.stamp.to_sec())
        finally:
            self.historical_data_lock.release()
        
    def online_predict(self):
        while not rospy.is_shutdown():
            name = '[ONLINE-PREDICT] '

            # reset historical data
            self.historical_data_lock.acquire(blocking=True)
            self.historical_data.reset()
            print('check len after reset: ', len(self.historical_data.get_data()))
            self.historical_data_lock.release()
            # print in green color
            print(name + 'Reset historical data')
            print('\n\n')
            self.printer.print_green(name + '===============================', background=True)
            self.printer.print_green(name + 'Press enter to start predicting')
            self.printer.print_green(name + '===============================\n', background=True)
            input()
            last_lenth = 0
            self.pred_seq = None
            self.pressed_enter_event.set()
            while not rospy.is_shutdown():
                # get data
                self.historical_data_lock.acquire(blocking=True)
                curr_historical_data = self.historical_data.get_data()
                self.historical_data_lock.release()               
                # print('     ', name + 'Current historical data len: ', len(curr_historical_data))

                if len(curr_historical_data) < self.input_len_req:
                    print('     ', name + 'loading more data ...')
                    continue
                if len(curr_historical_data) == last_lenth:
                    print('     ', name + 'No new data, skip this loop')
                    continue
                last_lenth = len(curr_historical_data)

                curr_pos = curr_historical_data[-1][:3]
                if curr_pos[2] < self.active_range_z[0]:
                    rospy.logwarn(name + 'Object height is under ' + str(self.active_range_z[0]) + ' m. Finish prediction !')
                    break

                # --- Predict ---
                pred_impacted = False
                while not pred_impacted:
                    # setup data
                    input_data = curr_historical_data[-self.input_len_req:]
                    # combine curr_historical_data and future prediction part of pred_seq
                    if self.pred_seq is not None:
                        nae_future_prediction = self.pred_seq[self.input_len_req-1:]
                        input_data = np.concatenate([input_data, nae_future_prediction[:self.auto_agressive_len]])[-self.input_len_req:]
                        # print(input_data.shape)
                        # input('TODO: check input_data: ')
                    # add batch dimension to input_data
                    input_data = np.expand_dims(input_data, axis=0)
                    # predict
                    self.pred_seq = self.nae.predict(input_data, evaluation=True).cpu().numpy()[0]
                    impact_point = self.pred_seq[-1]
                    # check impact point is in active range z
                    if impact_point[2] < self.active_range_z[0]:
                        pred_impacted = True
                        continue
                    # publish prediction
                    self.publish_prediction(self.pred_seq, impact_point=True, pred_traj=False)
                    print('     ', name + 'published prediction with len: ', len(self.pred_seq))

    def publish_prediction(self, pred_seq, impact_point=True, pred_traj=True):
        if impact_point:
            impact_point = Point()
            impact_point.x = pred_seq[-1][0]
            impact_point.y = pred_seq[-1][1]
            impact_point.z = pred_seq[-1][2]
            self.impact_point_pub.publish(impact_point)
        if pred_traj:
            pred_traj = PointArray()
            points = [Point(x=pred_seq[i][0], y=pred_seq[i][1], z=pred_seq[i][2]) for i in range(0, len(pred_seq))]
            pred_traj.points = points
            self.predicted_traj_pub.publish(pred_traj)

def main():
    rospy.init_node('nae_online_predictor')
    default_model = '/home/server-huynn/workspace/robot_catching_project/trajectory_prediction/nae_prediction_ws/src/nae/training_scripts/models/my_dataset_models/frisbee-input-15pred-50/run_NAE3L_26-10-2024_04-21-39/@epochs1020_data2_batchsize128_seq15_pred50_timemin34-01'
    default_swap_y_z = True
    default_mocap_topic = '/mocap_pose_topic/frisbee1_pose'

    mocap_topic = rospy.get_param('~mocap_topic', default_mocap_topic)
    model_path = rospy.get_param('~model_path', default_model)
    swap_y_z = rospy.get_param('~swap_y_z', default_swap_y_z)


    # Learning parameters
    input_lenth = 10
    future_pred_len = 50
    thrown_object = 'frisbee'
    training_params = {
        'input_len': input_lenth,
        'future_pred_len': future_pred_len,
        'num_epochs': 5000,
        'batch_size_train': 128,    
        'batch_size_val': 32,
        'save_interval': 10,
        'thrown_object' : thrown_object + '-input-' + str(input_lenth) + 'pred-' + str(future_pred_len)
    }
    model_params = {
        'input_size': 9,
        'hidden_size': 128,
        'output_size': 9,
        'num_layers_lstm': 2,
        'lr': 0.0001
    }
    prediction_params = {
        'auto_agressive_len': 5,     # number of future prediction steps that are used as input for the next prediction step
        'storage_max_len': 300
    }

    active_range_x = rospy.get_param('~active_range_x', [0, 10000])
    active_range_y = rospy.get_param('~active_range_y', [0, 10000])
    active_range_z = rospy.get_param('~active_range_z', [0.2, 10000])
    nae_online_predictor = NAEOnlinePredictor(model_path, model_params, training_params, prediction_params, mocap_topic, active_range_x, active_range_y, active_range_z, swap_y_z)
    rospy.spin()

if __name__ == '__main__':
    main()