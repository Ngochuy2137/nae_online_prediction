'''
This is a ROS node that:
    - subscribes to mocap topic to get position of the target object and predict its trajectory.
    - publishes the predicted trajectory to a topic.
'''

import rospy
import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import UInt8
from nav_msgs.msg import Path
from nae.nae import *
from nae.utils.submodules.printer import Printer
import threading

DEBUG_LOG = False

class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def is_in_range(self, val):
        if val < self.start or val > self.end:
            return False
        return True
    
class RobotOperatingArea:
    def __init__(self, x_range:Range, y_range:Range, z_range:Range, id):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.id = id

    def is_in_area(self, pos):
        if not self.x_range.is_in_range(pos[0]) or not self.y_range.is_in_range(pos[1]) or not self.z_range.is_in_range(pos[2]):
            return False
        return True
    def is_in_2d_area(self, pos):
        if not self.x_range.is_in_range(pos[0]) or not self.y_range.is_in_range(pos[1]):
            return False
        return True
    
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
            # # TODO: clear oldest data points if the length of data_points exceeds storage_max_len
            # self.clear_oldest_data_points()

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
    def __init__(self, model_path, model_params, training_params, prediction_params, mocap_topic, oprerating_areas, thow_active_range_x=[0, 10000], throw_active_range_y=[0, 10000], throw_active_range_z=[0.1, 10000], swap_y_z=False):
        # Load model
        self.nae = NAE(**model_params, **training_params, device=DEVICE)
        self.nae.load_model(model_path)
        self.printer = Printer()

        self.input_len_req = training_params['input_len']
        self.future_pred_len = training_params['future_pred_len']
        self.auto_agressive_len = prediction_params['auto_agressive_len']
        self.oprerating_areas = oprerating_areas
        self.thow_active_range_x = thow_active_range_x
        self.throw_active_range_y = throw_active_range_y
        self.throw_active_range_z = throw_active_range_z

        self.swap_y_z = swap_y_z
        # Subscribe to mocap topic
        self.mocap_sub = rospy.Subscriber(mocap_topic, PoseStamped, self.mocap_callback)
        # Publish predicted trajectory
        self.predicted_traj_pub = rospy.Publisher('NAE/predicted_traj', Path, queue_size=1)
        self.impact_point_pub = rospy.Publisher('NAE/impact_point', PoseStamped, queue_size=1)
        self.danger_zone_pub = rospy.Publisher('NAE/danger_zone', UInt8, queue_size=1)

        # Variables
        self.pred_seq = None
        self.pred_impact_point = None
        self.danger_zone_id = None

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
            # rospy.logwarn(name + 'Press ENTER to enable getting mocap data')
            # skip this callback loop if the enter key is not pressed
            return
        
        # check current position is in active range
        if curr_pos[0] < self.thow_active_range_x[0] or curr_pos[0] > self.thow_active_range_x[1] or \
            curr_pos[1] < self.throw_active_range_y[0] or curr_pos[1] > self.throw_active_range_y[1]:

            self.printer.print_purple(name + 'Current position is out of active range', enable=DEBUG_LOG)
            self.printer.print_purple('     cur_pos: ' + str(curr_pos) + ' thow_active_range_x: ' + str(self.thow_active_range_x) + ' throw_active_range_y: ' + str(self.throw_active_range_y), enable=DEBUG_LOG)
            return
    
        # TODO: Using queue to store data, prevent data loss
        if not self.historical_data_lock.acquire(blocking=True):
            # TODO: print in red color
            rospy.logwarn(name + 'Cannot get lock, data loss')
            input()
            return
        try:
            self.historical_data.append(curr_pos, msg.header.stamp.to_sec())
        finally:
            self.historical_data_lock.release()
        
    def online_predict(self):
        # LOOP: Maintain the thread
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
            self.printer.print_green(name + 'Press ENTER 2 times to start predicting')
            self.printer.print_green(name + '===============================', background=True)
            input()

            # reset variables
            last_lenth = 0
            self.pred_seq = None
            self.pred_impact_point = None
            self.danger_zone_id = None

            self.publish_prediction(default_enable=True)
            self.printer.print_green(name + 'Ready to predict. Press ENTER 1 more time', background=True)
            input()

            self.pressed_enter_event.set()
            # LOOP: Prediction for one throw
            count = 0
            while not rospy.is_shutdown():
                if count == 0:
                    # dummy predict to warm up
                    self.dummy_predict(iteration=3)
                    count += 1
                # get data
                self.historical_data_lock.acquire(blocking=True)
                curr_historical_data = self.historical_data.get_data()
                self.historical_data_lock.release()     

                if len(curr_historical_data) < self.input_len_req:
                    # print('     ', name + 'loading more data ... current length: ', len(curr_historical_data))
                    continue
             
                curr_pos = curr_historical_data[-1][:3]

                # check out of active range x, y, z
                if curr_pos[0] < self.thow_active_range_x[0] or curr_pos[0] > self.thow_active_range_x[1] or \
                    curr_pos[1] < self.throw_active_range_y[0] or curr_pos[1] > self.throw_active_range_y[1]:
                    rospy.logwarn(name + 'Current position is out of active range')
                    continue

                # if len(curr_historical_data) == last_lenth:
                #     print('     ', name + 'No new data, skip this loop, current length: ', last_lenth)
                #     continue

                # print('             get new data: ', len(curr_historical_data))
                last_lenth = len(curr_historical_data)

                if curr_pos[2] < self.throw_active_range_z[0]:
                    rospy.logwarn(name + 'Object height is under ' + str(self.throw_active_range_z[0]) + ' m. Finish prediction !')
                    # cleat enter event
                    self.pressed_enter_event.clear()
                    break

                # --- Predict ---
                count = 0
                predict_time = time.time()

                # LOOP: Predict until impact ground
                while not rospy.is_shutdown():
                    count += 1
                    if count > 1:
                        self.printer.print_purple('\n' + name + 'predict ... ' + str(count))
                    if count == 5:
                        print('\n----- 222')
                        self.pred_impact_point = self.publish_prediction(impact_point_enable=True, pred_traj_enable=True, danger_zone_enable=True, print_title=name)
                        # print in green color
                        self.printer.print_green(name + 'published prediction with len: ' + str(len(self.pred_seq)), enable=DEBUG_LOG)
                        pred_rate = 1 / (time.time() - predict_time)
                        if pred_rate <200:
                            self.printer.print_purple('             predict rate: ' + str(pred_rate))
                        else:
                            self.printer.print_green('             predict rate: ' + str(pred_rate), enable=DEBUG_LOG)
                        break

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

                    is_impact_prediction, self.pred_seq = self.filter_impact_point(self.pred_seq)
                    self.pred_impact_point = self.pred_seq[-1]
                    self.danger_zone_id = self.get_danger_zone(self.pred_impact_point)

                    if is_impact_prediction:
                        # publish prediction
                        print('\n----- 111')
                        self.publish_prediction(impact_point_enable=True, pred_traj_enable=True, danger_zone_enable=True, print_title=name)

                        # log
                        self.printer.print_green(name + 'published prediction with len: ' + str(len(self.pred_seq)), enable=DEBUG_LOG)
                        pred_rate = 1 / (time.time() - predict_time)
                        if pred_rate <200:
                            self.printer.print_purple('             predict rate: ' + str(pred_rate))
                        else:
                            self.printer.print_green('             predict rate: ' + str(pred_rate), enable=DEBUG_LOG)
                        
                        break

    
    '''
    function: Filter impact point whose z close to self.throw_active_range_z[0], shorten the predicted trajectory
    return:
        + is_impact_prediction: flag to check if the predicted seq pass the height limit with z = self.throw_active_range_z[0]
        + pred_seq: the pred_seq might be shorten
    '''
    def filter_impact_point(self, pred_seq):
        # filter out impact point whose z close to self.throw_active_range_z[0]
        is_impact_prediction = False
        for i in range(len(pred_seq)):
            if pred_seq[i][2] <= self.throw_active_range_z[0]:
                pred_seq = pred_seq[:i+1]
                is_impact_prediction = True
                break
        return is_impact_prediction, pred_seq
    
    '''
    dummy predict for nae model to warm up
    '''
    def dummy_predict(self, iteration=1):
        print('     dummy predict ... ')
        input_data = np.random.rand(iteration, self.input_len_req, 9)
        self.nae.predict(input_data, evaluation=True)

    '''
    Function publish_prediction:
        function: 
            + Publish predicted trajectory to a topic
            + Filter and Publish predicted impact point to a topic
        input:
            + pred_seq: predicted trajectory
            + impact_point: flag to publish impact point
            + pred_traj: flag to publish predicted trajectory
    '''
    def publish_prediction(self, impact_point_enable=False, pred_traj_enable=False, danger_zone_enable=False, default_enable = False, print_title=''):
        impact_point = PoseStamped()
        if default_enable:
            impact_point.header.stamp = rospy.Time.now()
            impact_point.header.frame_id = 'world'
            impact_point.pose.position.x = self.thow_active_range_x[0]
            impact_point.pose.position.y = self.throw_active_range_y[0]
            impact_point.pose.position.z = 0
            impact_point.pose.orientation.w = 1
            self.impact_point_pub.publish(impact_point)
            self.printer.print_green('[PUBLISH-PREDICTION] Reset prediction to ' + str(impact_point.pose.position.x) + ' ' + str(impact_point.pose.position.y) + ' ' + str(impact_point.pose.position.z))
            return 
        
        if impact_point_enable:
            impact_point.header.stamp = rospy.Time.now()
            impact_point.header.frame_id = 'world'
            impact_point.pose.position.x = self.pred_impact_point[0]
            impact_point.pose.position.y = self.pred_impact_point[1]
            impact_point.pose.position.z = self.pred_impact_point[2]
            impact_point.pose.orientation.w = 1
            self.impact_point_pub.publish(impact_point)
            self.printer.print_green(print_title + 'pred_impact_point: ' + str(self.pred_impact_point[0]) + ' ' + str(self.pred_impact_point[1]) + ' ' + str(self.pred_impact_point[2]))


        if pred_traj_enable:
            pred_traj = Path()
            pred_traj.header.stamp = rospy.Time.now()
            pred_traj.header.frame_id = 'world'
            # setup points
            for i in range(0, len(self.pred_seq)):
                point = Point()
                point.x = self.pred_seq[i][0]
                point.y = self.pred_seq[i][1]
                point.z = self.pred_seq[i][2]

                one_data_pose_stamp = PoseStamped()
                one_data_pose_stamp.pose.position = point
                one_data_pose_stamp.pose.orientation.w = 1
                one_data_pose_stamp.header = pred_traj.header
                one_data_pose_stamp.header.seq = i
                pred_traj.poses.append(one_data_pose_stamp)

            self.predicted_traj_pub.publish(pred_traj)

        if danger_zone_enable:
            self.danger_zone_pub.publish(self.danger_zone_id)
            self.printer.print_green(print_title + 'danger_zone:       ' + str(self.danger_zone_id))

        return
    
    def get_danger_zone(self, pred_impact_point):
        # check which area the impact point is in
        impact_point = [pred_impact_point[0], pred_impact_point[1], pred_impact_point[2]]
        for area in self.oprerating_areas:
            if area.is_in_2d_area(impact_point):
                return area.id

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

    # thow_active_range_x = rospy.get_param('~thow_active_range_x', [2.5, 10000])
    # throw_active_range_y = rospy.get_param('~throw_active_range_y', [0, 100000])
    # throw_active_range_z = rospy.get_param('~throw_active_range_z', [0.2, 10000])
    thow_active_range_x = rospy.get_param('~thow_active_range_x', [-0.5, 10000])
    throw_active_range_y = rospy.get_param('~throw_active_range_y', [-1.5, 100000])
    throw_active_range_z = rospy.get_param('~throw_active_range_z', [0.3, 10000])

    # setup robot operating area
    area_1_x = Range(1.5, 3.0) 
    area_1_y = Range(-3.2, 0)
    area_1_id = 1

    area_2_x = Range(1.5, 3.0)
    area_2_y = Range(0, 3.2)
    area_2_id = 2

    area_3_x = Range(3, 4.5)
    area_3_y = Range(0, 3.2)
    area_3_id = 3

    area_4_x = Range(3, 4.5)
    area_4_y = Range(-3.2, 0)
    area_4_id = 4



    area_1 = RobotOperatingArea(area_1_x, area_1_y, Range(0.2, 0.5), area_1_id)
    area_2 = RobotOperatingArea(area_2_x, area_2_y, Range(0.2, 0.5), area_2_id)
    area_3 = RobotOperatingArea(area_3_x, area_3_y, Range(0.2, 0.5), area_3_id)
    area_4 = RobotOperatingArea(area_4_x, area_4_y, Range(0.2, 0.5), area_4_id)
    oprerating_areas = [area_1, area_2, area_3, area_4]

    nae_online_predictor = NAEOnlinePredictor(model_path, model_params, training_params, prediction_params, mocap_topic, oprerating_areas, thow_active_range_x, throw_active_range_y, throw_active_range_z, swap_y_z)
    rospy.spin()

if __name__ == '__main__':
    main()