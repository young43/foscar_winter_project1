#!/usr/bin/python
#-*- encoding: utf-8 -*-

import cv2, rospy, time
import numpy as np
import math

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from turtlesim.msg import Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler

bridge = CvBridge()
img = np.empty(shape=[0])
x, y, theta = 0, 0, 0

control_msg = Twist()

class Turtle:
    def __init__(self):
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.pose_callback)

        self.pose = Pose()
        self.goal_pose = Pose()

    def control_msg_publish(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = angular_z

        self.velocity_publisher.publish(msg)

    def pose_callback(self, data):
        self.pose = data
        self.pose.x = round(self.pose.x, 6)
        self.pose.y = round(self.pose.y, 6)
        self.pose.theta = round(self.pose.theta, 6)

    def euclidean_distance(self):
        return np.sqrt(pow((self.goal_pose.x - self.pose.x), 2) +
                    pow((self.goal_pose.y - self.pose.y), 2))

    def steering_angle(self):
        return math.atan2(self.goal_pose.y - self.pose.y, self.goal_pose.x - self.pose.x)

    def set_goal_pose(self, goal_pose):
        self.goal_pose.x = goal_pose[0]
        self.goal_pose.y = goal_pose[1]

    def move2goal(self):
        distance_tolerance = 0.3

        # rotation 각도 미리설정
        if self.steering_angle() > self.pose.theta:
            while self.steering_angle() > self.pose.theta:
                self.control_msg_publish(0, 1)
        else:
            while self.steering_angle() < self.pose.theta:
                self.control_msg_publish(0, -1)

        self.control_msg_publish(0, 0)

        # straight
        while self.euclidean_distance() > distance_tolerance:
            self.control_msg_publish(3, 0)

        self.control_msg_publish(0, 0)


def image_callback(img_data):
    global bridge
    global img
    img = bridge.imgmsg_to_cv2(img_data, "bgr8")



def get_green(img):
    # static 변수로 선언
    get_green.ROI = [
        [[0, 0], [160, 210]],
        [[0, 210], [160, 420]],
        [[0, 420], [160, 640]],

        [[160, 0], [320, 210]],
        [[160, 210], [320, 420]],
        [[160, 420], [320, 640]],

        [[320, 0], [480, 210]],
        [[320, 210], [480, 420]],
        [[320, 420], [480, 640]],
    ]

    max_element = None
    max_value = 5000

    for region in get_green.ROI:
        roi_img = img[region[0][0]:region[1][0], region[0][1]:region[1][1]]
        if cv2.countNonZero(roi_img) > max_value:
            max_value = cv2.countNonZero(roi_img)

            points = cv2.findNonZero(roi_img).reshape(-1, 2)
            max_element = np.mean(points, axis=0)
            max_element[0] += region[0][0]
            max_element[1] += region[0][1]

    return max_element



if __name__ == "__main__":
    rospy.init_node("foscar_project")
    rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)

    time.sleep(3)

    turtle = Turtle()
    tutlesim_window = np.array([11, 11], dtype=float)
    img_shape = np.array([480, 640], dtype=float)

    # 화면비율
    ratio = tutlesim_window/img_shape

    while not rospy.is_shutdown():
        if cv2.waitKey(1) & 0xff == 27:
            break

        # 이미지 hsv 변환해서 저장하기
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([40, 40, 40])
        upper = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)

        coord = get_green(mask)

        if coord is not None:
            # 거북이 pose 정보 출력
            print("x : ", turtle.pose.x)
            print("y : ", turtle.pose.x)
            print("theta : ", math.degrees(turtle.pose.theta))

            print("goal_pose(cv):", coord)
            coord[0] = abs(coord[0] - img_shape[0])
            target_pos = coord * ratio
            target_pos = target_pos[::-1]
            print("goal_pose(turtle):", target_pos)

            turtle.set_goal_pose(target_pos)
            turtle.move2goal()

        # 격자무늬 시각화
        cv2.line(img, (0, 160), (639, 160), (0, 0, 0), 1)
        cv2.line(img, (0, 320), (639, 320), (0, 0, 0), 1)
        cv2.line(img, (210, 0), (210, 480), (0, 0, 0), 1)
        cv2.line(img, (420, 0), (420, 480), (0, 0, 0), 1)

        # 비디오 화면 띄우기
        # 동시에 여러 창 띄우는것도 가능
        cv2.imshow("image", img)
        cv2.imshow("mask", res)

    cv2.destroyAllWindows()
