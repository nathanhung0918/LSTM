#!/usr/bin/python
# This is a MIRO ROS client for Python.

import rospy
# http://docs.ros.org/api/geometry_msgs/html/msg/TwistStamped.html
from geometry_msgs.msg import TwistStamped

import math
import numpy as np
import time
import sys
import os
import datetime
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, JointState
import matplotlib.pyplot as plt
import numpy.matlib
from random import randrange
from pynput import keyboard
# The miro2 Python module provides constants and tools for working with MiRo from Python.
# with PYTHONPATH=~/mdk/share/python:$PYTHONPATH
import miro2 as miro
import nsm

################################################################

def error(msg):
    print(msg)
    sys.exit(0)


################################################################

class controller:

    def cam_right_callback(self, ros_image):
        try:
            self.cam_right_image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            im_h, im_w = self.cam_right_image.shape[:2]
            if self.frame_w != im_w and self.frame_h != im_h:
                self.frame_w, self.frame_h = im_w, im_h
                self.cam_model.set_frame_size(self.frame_w, self.frame_h)
        except CvBridgeError as e:
            print("Conversion of right image failed \n")
            print(e)

    def callback_caml(self, ros_image):
        # ignore until active
        if not self.active:
            return


        # detect ball
        try:
            self.cam_left_image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            im_h, im_w = self.cam_left_image.shape[:2]
            if self.frame_w != im_w and self.frame_h != im_h:
                self.frame_w, self.frame_h = im_w, im_h
                self.cam_model.set_frame_size(self.frame_w, self.frame_h)
        except CvBridgeError as e:
             print("Conversion of left image failed \n")
             print(e)



        # convert compressed ROS image to raw CV image


        image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
        # if self.i < self.n:
        #     print('taking photo', str(self.i))
        #     cv2.imwrite(str(self.i) + '.jpg', image)
        #     time.sleep(1.5)
        #     self.i += 1


        # get current observation
        image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "bgr8")
        res = cv2.resize(image, dsize=(self.s, self.s), interpolation=cv2.INTER_CUBIC)
        res = np.mean(res, 2)
        self.has_move = True
        self.observation = self.match(res)  # get the observation. Take a frame and encode it

    def callback_package(self, msg):

        x = msg.sonar.range

        # if len(self.distance) >= 10:
        #     self.distance.pop(0)
        #
        # self.distance.append(x)
        #
        # self.dis = sum(self.distance) / len(self.distance)
        self.dis = x
        # print ("sonar ", x)

    def loop(self):
        # loop
        while self.active and not rospy.core.is_shutdown():
            # print('lengthhhhhhh',len(self.performance))
            if len(self.performance) > 20 :
                plt.plot(self.performance)
                plt.title('Performance (steps till reward)')
                self.saveEXP()
                plt.show()
                self.performance = []
                self.iter = 0
            #
            else:
                self.task()
            # self.task()
            # self.movement()
            # self.offline()
            # self.read()
            pass

    def getImage(self,name):
        img = cv2.imread(name)
        res = cv2.resize(img, dsize = (self.s,self.s), interpolation=cv2.INTER_CUBIC)
        res = np.mean(res,2)

        return res

    def countV(self):

        # repeat this n times
        # Apply PCA
        h, w = self.X.shape
        mu = np.reshape(np.mean(self.X, 1), (h, 1))
        # Substract mean
        self.X = self.X - np.matlib.repmat(mu, 1, w)
        # Covariance matrix
        C = np.dot(self.X.T, self.X)
        # Eigendecpmposition
        d, v = np.linalg.eig(C)
        # This the projection matrix
        self.v = v[:, 0:3]
        # save file
        np.save('test.npy', self.v)

    def process(self):
        temp = []
        # Compute the projection
        for i in range(self.n):
            z = np.dot(self.v.T, self.X[i].flatten())
            print(self.encode(z))
            temp.append(self.encode(z))
            plt.plot(z[0], z[1], 'r*')
            plt.annotate( 'image'+str(i) + ':class' + str(temp[i]) ,xy=(z[0],z[1]))
            # print('plot ' + str(i))
    # pre process data. take photos and count projection matrix
    def offline(self):
        # count projection + plot
        for i in range(self.n):
            print('processing: ' + str(i))
            image = self.getImage(str(i) + '.jpg')
            self.X[i, :] = image.flatten()
        self.countV()
        print('finish count')
        self.process()
        print('finish process')
        plt.show()

    def match(self,image):
        self.X[0, :] = image.flatten()
        z = np.dot(self.v.T, self.X[0].flatten())
        return self.encode(z)
    # test read file
    def read(self):
        t1 = time.time()

        for i in range(self.n):
            print('processing: ' + str(i))
            image = self.getImage(str(i) + '.jpg')
            self.X[i,:] = image.flatten()

        # This the projection matrix
        self.v = np.load('test.npy')
        self.process()

        plt.show()
        t2 = time.time()
        print('time: ', t2 - t1)

    def encode(self,z):
        temp_code = ''
        for i in z:
            if i > 0:
                temp_code += str(1)
            else:
                temp_code += str(0)

        return int(temp_code, 2)

    def movement(self):

        # Create object to hold published data
        velocity = TwistStamped()
        l_val = 0
        r_val = 0
        if self.cmd == 'a':
            self.rotate('a')
            l_val = 3
            r_val = 3
        elif self.cmd == 'd':
            self.rotate('d')
            l_val = 3
            r_val = 3

        elif self.cmd =='w':
            l_val = 3
            r_val = 3
        elif self.cmd == 's':
            # print('do back')
            l_val = -3
            r_val = -3
            start = datetime.datetime.now()
            wheel_speed = [l_val, r_val]
            (dr, dtheta) = miro.utils.wheel_speed2cmd_vel(wheel_speed)
            velocity.twist.linear.x = dr
            velocity.twist.angular.z = dtheta

            # self.pub_cmd_vel.publish(velocity)
            while True:
                end = datetime.datetime.now()
                if (end - start).seconds > 0:
                    velocity.twist.linear.x = 0
                    velocity.twist.angular.z = 0
                    self.pub_cmd_vel.publish(velocity)
                    break
                else:
                    self.pub_cmd_vel.publish(velocity)

        wheel_speed = [l_val, r_val]
        # print(wheel_speed)
        (dr, dtheta) = miro.utils.wheel_speed2cmd_vel(wheel_speed)
        # print(dr,dtheta)

        # Set velocity values
        velocity.twist.linear.x = dr
        velocity.twist.angular.z = dtheta

        start = datetime.datetime.now()

        # self.pub_cmd_vel.publish(velocity)
        while self.dis > 0.1:
            end = datetime.datetime.now()
            if (end - start).seconds > 0:
                velocity.twist.linear.x = 0
                velocity.twist.angular.z = 0

                self.pub_cmd_vel.publish(velocity)
                break
            else:
                self.pub_cmd_vel.publish(velocity)
        return True

    def on_press(self, key):
        try:
            self.cmd = str(key.char)
            self.movement()
            self.neck()
            # print('alphanumeric key {0} pressed'.format(
            #     key.char))
        except AttributeError:
            pass
            # print('special key {0} pressed'.format(
            #     key))

    def on_release(self, key):
        # print('{0} released'.format(
        #     key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    def neck(self):
        tilt, lift, yaw, pitch = range(4)
        diff_yaw = miro.constants.YAW_RAD_MAX - miro.constants.YAW_RAD_MIN

        if self.cmd == 'j':
            self.kin_joints.position[yaw] += diff_yaw / 5

        elif self.cmd =='l':
            self.kin_joints.position[yaw] -= diff_yaw / 5

        elif self.cmd == 'i':
            self.kin_joints.position[lift] -= math.radians(10.0)

        elif self.cmd == 'k':
            self.kin_joints.position[lift] += math.radians(10.0)

        self.pub_kin.publish(self.kin_joints)

        # time.sleep(0.01)

    def reset_neck(self):
        self.kin_joints.position = [0.0, math.radians(34.0), 0.0, 0.0]
        self.pub_kin.publish(self.kin_joints)
        # time.sleep(1)

    def check_wall(self):
        print(self.dis)

        if self.dis <= 0.15 :
            print('is wall')
            return True
        else:
            print('not wall')
            return False

    def __init__(self, args):
        self.cam_left_image = None
        self.cam_right_image = None
        self.image = None
        rospy.init_node("client", anonymous=True)
        #forself.task_init() image data
        self.n = 100
        self.s = 64
        self.X = np.zeros((self.n,self.s*self.s))
        self.i = 0
        self.v = np.load('test.npy')  # loads the projection matrix
        #sonar
        self.distance = []
        self.dis = 0
        #movement
        self.speed = 1.5
        self.cmd = None
        self.has_move = False
        self.action_list = ['w', 'a', 'd']
        self. action_rev = ['s', 'd', 'a']
        # Global state variable
        self.observation = None
        self.reward = 0
        self.similar_states = None
        self.action = None
        #ros state
        self.active = False
        #camera
        self.cam_model = miro.utils.CameraModel()
        self.frame_w = 0
        self.frame_h = 0
        self.image_converter = CvBridge()


        # handle args
        for arg in args:
            f = arg.find('=')
            if f == -1:
                key = arg
                val = ""
            else:
                key = arg[:f]
                val = arg[f + 1:]
            if key == "pass":
                pass
            else:
                error("argument not recognised \"" + arg + "\"")

        # robot name
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # publish
        #wheel speed
        topic = topic_base_name + "/control/cmd_vel"
        print("publish", topic)
        self.pub_cmd_vel = rospy.Publisher(topic, TwistStamped, queue_size=1)
        #kinematic
        topic = topic_base_name + "/control/kinematic_joints"
        print("publish", topic)
        self.pub_kin = rospy.Publisher(topic, JointState, queue_size=0)
        self.kin_joints = JointState()
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        # self.reset_neck()

        # subscribe
        # sensors/caml, Frames from the left eye camera (sample rate is variable, see control/command).
        topic = topic_base_name + "/sensors/caml/compressed"
        print("subscribe", topic)
        self.sub_caml = rospy.Subscriber(topic, CompressedImage, self.callback_caml, queue_size=1,
                                         tcp_nodelay=True)
        topic = topic_base_name + "/sensors/camr/compressed"
        self.cam_right_sub = rospy.Subscriber(topic, CompressedImage,
                                              self.cam_right_callback, queue_size=1,
                                         tcp_nodelay=True)
        #sonar
        topic = topic_base_name + "/sensors/package"
        print("subscribe", topic)
        self.sub_package = rospy.Subscriber(topic, miro.msg.sensors_package, self.callback_package)

        # wait for connect
        print
        "wait for connect..."
        # time.sleep(1)

        # set to active
        self.active = True
        # keyboard listener
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()
        self.task_init()

    def saveEXP(self):
        a = []
        r = []
        o = []
        for exp in self.chain.container:
            a.append(exp.action)
            r.append(exp.reward)
            o.append(exp.observation)
        with open("action_list.txt", "w") as f:
            for temp in a:
                f.write(str(temp) + "\n")
        with open("reward_list.txt", "w") as f:
            for temp in r:
                f.write(str(temp) + "\n")
        with open("observation_list.txt", "w") as f:
            for temp in o:
                f.write(str(temp) + "\n")

    def readEXP(self):
        a = []
        r = []
        o = []
        with open("action_list.txt", "r") as f:
            for line in f:
                a.append(int(line.strip()))
        with open("reward_list.txt", "r") as f:
            for line in f:
                r.append(int(line.strip()))
        with open("observation_list.txt", "r") as f:
            for line in f:
                o.append(int(line.strip()))

        mChain = Chain(self.N)
        for i in range(len(a)):
            exp = Experience(a[i],r[i],o[i])
            mChain.add(exp)

        return mChain

    def checkReward(self):
        temp_l = self.find_ball(cam_id=0)
        temp_r = self.find_ball(cam_id=1)

        if temp_l is not None:
            if temp_l > 5:
                print('get a reward l',temp_l)
                self.current_reward = 1
                if temp_l > 10:
                    self.current_reward = 2
                return True
        if temp_r is not None:
            if temp_r > 5:
                print('get a reward r',temp_r)
                self.current_reward = 1
                if temp_r > 10:
                    self.current_reward = 2
                return True

        return False

    def rotate(self,direction):
        velocity = TwistStamped()
        if direction == 'd':
            self.cmd = 'd'
            start = datetime.datetime.now()
            while self.dis > 0.1:
                end = datetime.datetime.now()
                if (end - start).seconds > 0:
                    velocity.twist.linear.x = 0
                    velocity.twist.angular.z = 0
                    self.pub_cmd_vel.publish(velocity)
                    break
                else:
                    # print('do rotate')
                    velocity.twist.linear.x = 0
                    velocity.twist.angular.z = -math.pi/2
                    self.pub_cmd_vel.publish(velocity)
        elif direction == 'a':
            self.cmd = 'a'
            start = datetime.datetime.now()
            while self.dis > 0.1:
                end = datetime.datetime.now()
                if (end - start).seconds > 0:
                    velocity.twist.linear.x = 0
                    velocity.twist.angular.z = 0
                    self.pub_cmd_vel.publish(velocity)
                    break
                else:
                    # print('do rotate')
                    velocity.twist.linear.x = 0
                    velocity.twist.angular.z = math.pi / 2
                    self.pub_cmd_vel.publish(velocity)

    def step(self, action):
        observation = None
        # self.checkReward()
        if self.dis < 0.2:
            self.reward = -1
        elif self.checkReward():  # set the reward
            self.reward = self.current_reward
        else:
            self.reward = 0

    def is_valid(self, action):
        if self.dis < 0.2 and self.action_list[action] is 'w':
            self.speed = 0.3
            return False
        # else:
        #     self.cmd = self.action_list[action]
        #     print('not valid so do back')
        #     self.movement()
        #     return False
        self.speed = 0.5
        return True

    def sample(self):
        # if self.dis < 0.2:
        #     return randrange(2)+1
        # else:
        #     return 0
        return randrange(3)

    def task_init(self, k=3, N=1000, beta=0.4, gamma=0.99, e=.1):
        self.k = k
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.e = e
        print("Initializing the chain")
        self.chain = Chain(self.N)
        # self.chain = self.readEXP()
        self.performance = []
        self.done = False
        self.decay = 25.0
        self.min_e = 0.01

        self.f = lambda x, min_x: max(min_x, min(1.0, 1.0 - np.log10((x + 1) / self.decay)))
        self.iter = 0

    def getQTable(self, similar_states):
        Q = np.zeros((3, 1))
        counts = np.zeros((3, 1))

        for s in similar_states:
            Q[s.action] = Q[s.action] + s.q
            counts[s.action] += 1

        for i in range(3):
            if counts[i] > 0:
                Q[i] /= float(counts[i])
            else:
                Q[i] = 0

        return Q

    def getVotingStates(self, action, similar_states):
        voting_states = []

        for s in similar_states:
            if s.action == action:
                voting_states.append(s)

        return voting_states

    def nextAction(self, Q):

        # Make the decision using the e-greedy policy
        if np.random.random() < self.e:
            action = self.sample()
        else:
            action = np.argmax(Q)

        Qmax = np.max(Q)

        return action, Qmax

    def task(self):
        # self.chain = self.readEXP()
        steps = 0
        self.done = False
        self.iter += 1
        while not self.done:
            if not self.has_move:

                # We get the similar experiences
                # They are instances of Experience
                self.similar_states = self.chain.getKNeighbours(self.k)
                print('Similar states:')

                for i in self.similar_states:
                    print(i.action, i.reward, i.observation)

                Q = self.getQTable(self.similar_states)

                self.action, self.Qmax = self.nextAction(Q)

                while not self.is_valid(self.action):
                    self.action, self.Qmax = self.nextAction(Q)

                if self.dis < 0.1:
                    self.cmd = 's'
                    self.movement()

                self.cmd = self.action_list[self.action]
                self.movement()

                # print("sonar ", self.dis)
                print('do ',self.cmd)

                # get observation from the world
                self.step(self.action)

                time.sleep(0.5)
                self.has_move = True
            elif self.observation is not None:

                if self.reward == 2:
                    self.performance.append(steps)
                    print('yeahhhhhhhhh')
                    self.done = True


                # print(self.env.location_s,self.env.location_h,reward)
                e = Experience(self.action, self.reward, self.observation)

                print('Current experience: {}, {}, {}'.format(e.action, e.reward, e.observation))
                self.chain.add(e)

                voting_states = self.getVotingStates(self.action, self.similar_states)

                # Update the Q values
                for s in voting_states:
                    s.q = (1 - self.beta) * s.q + self.beta * (self.reward + self.gamma * self.Qmax)

                self.observation = None
                self.has_move = False

            steps += 1

        # self.e = self.f(self.iter, self.min_e)
        # print('eeee ', self.e)


    def find_ball(self, cam_id, colour_str='#0000FF', prop=2):
        # self.pause()
        if colour_str[0] != "#" and len(colour_str) != 7:
            print("colour choice should be a string in the form \"#RRGGBB\"")
            return
        if cam_id < 0 or cam_id > 1:
            return
        if prop < 0 or prop > 2:
            return

        # create colour code from user selected colour
        red = int(colour_str[1:3], 16)
        green = int(colour_str[3:5], 16)
        blue = int(colour_str[5:7], 16)
        bgr_colour = np.uint8([[[blue, green, red]]])
        hsv_colour = cv2.cvtColor(bgr_colour, cv2.COLOR_BGR2HSV)

        # extract boundaries for masking image
        target_hue = hsv_colour[0, 0][0]
        lower_bound = np.array([target_hue - 20, 70, 70])
        upper_bound = np.array([target_hue + 20, 255, 255])

        if np.shape(self.cam_left_image) != () and np.shape(self.cam_right_image) != ():
            # convert camera image to HSV colour space
            if cam_id == miro.constants.CAM_L:
                hsv_image = cv2.cvtColor(self.cam_left_image, cv2.COLOR_BGR2HSV)
                output = self.cam_left_image.copy()
            else:
                hsv_image = cv2.cvtColor(self.cam_right_image, cv2.COLOR_BGR2HSV)
                output = self.cam_right_image.copy()
        else:
            return None

        # cv2.imshow("eye", output)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        im_h = np.size(hsv_image, 0)
        im_w = np.size(hsv_image, 1)
        im_centre_h = im_h / 2.0
        im_centre_w = im_w / 2.0
        cv2.line(output, (0, int(round(im_centre_h))), (im_w, int(round(im_centre_h))), (100, 100, 100), 1)
        cv2.line(output, (int(round(im_centre_w)), 0), (int(round(im_centre_w)), im_h), (100, 100, 100), 1)

        # mask image
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        seg = mask

        # Do some processing
        seg = cv2.GaussianBlur(seg, (11, 11), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)

        # get circles
        circles = cv2.HoughCircles(seg, cv2.HOUGH_GRADIENT, 1, 40, param1=10, param2=20, minRadius=0, maxRadius=0)

        # Get largest circle
        max_circle = None
        max_circle_norm = [None, None, None]
        if circles is not None:
            self.max_rad = 0
            circles = np.uint16(np.around(circles))

            for c in circles[0, :]:
                cv2.circle(seg, (c[0], c[1]), c[2], (0, 255, 0), 2)

                if c[2] > self.max_rad:
                    self.max_rad = c[2]
                    max_circle = c
                    max_circle_norm[0] = int(round(((max_circle[0] - im_centre_w) / im_centre_w) * 100.0))
                    max_circle_norm[1] = int(round(-((max_circle[1] - im_centre_h) / im_centre_h) * 100.0))
                    max_circle_norm[2] = int(round((max_circle[2] / im_centre_w) * 100.0))

                # Debug Only
                cv2.circle(output, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 0), 2)
                cv2.circle(output, (max_circle[0], max_circle[1]), 1, (0, 255, 0), 2)
                location_str = "x: " + str(max_circle_norm[0]) + "," + "y: " + str(
                    max_circle_norm[1]) + "," + "r: " + str(max_circle[2])
                text_y_offset = 18
                for i, line in enumerate(location_str.split(",")):
                    text_y = max_circle[1] - text_y_offset + i * text_y_offset
                    cv2.putText(output, line, (max_circle[0] + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(output, line, (max_circle[0] + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                1)

        else:
            return None

        # debug_image = self.image_converter.cv2_to_imgmsg(output, "bgr8")
        # self.debug_image_pub.publish(debug_image)
        return max_circle_norm[prop]

class Experience:
# Stores one experience element of the  agent
	def __init__( self, action, reward, observation ):
		self.action = action
		self.reward = reward
		self.observation = observation
		self.q = 0

	def __eq__(self, s):
		return (self.action == s.action) and (self.reward == s.reward) and (self.observation == s.observation)

class Chain:
    # Full sequence of experiences by the agent
    def __init__(self, N):
        self.N = N
        self.container = []

    # self.container = [Experience(1,0,14),Experience(1,0,10)]

    def add(self, experience):

        if len(self.container) < self.N:
            self.container.append(experience)
        else:
            self.container.pop(0)
            self.container.append(experience)

    def getKNeighbours(self, k):
        m = len(self.container)

        if m < 2:
            return []

        n = np.zeros((m, m))
        for i in range(len(n[0])):
            for j in range(len(n[1])):
                n[i][j] = randrange(3)

        for i in range(m):
            for j in range(m):
                if i > 0 and j > 0:
                    si = self.container[i - 1]
                    sj = self.container[j - 1]

                    if si == sj:
                        n[i, j] = 1 + n[i - 1, j - 1]
                    else:
                        n[i, j] = 0
                else:
                    n[i, j] = 1

        return [self.container[i] for i in range(len(n[-1])-1) if n[-1][i] >= k-1]



if __name__ == "__main__":
    main = controller(sys.argv[1:])
    main.loop()












