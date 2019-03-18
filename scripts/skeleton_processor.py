#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from PIL import ImageFont, ImageDraw
from PIL import Image as PIL_Image
import cv2 
from cv_bridge import CvBridge, CvBridgeError
import sys


import rospy
from body_tracker_msgs.msg import Skeleton
from sensor_msgs.msg import Image
import message_filters



class skelly:

    def __init__(self):
        # self.image_pub = rospy.Publisher("/skeleton_projection",Image, queue_size=10)
        
        self.bridge = CvBridge()
        self.skeleton_image = None
        self.noconf_list = {}
        self.prev_frame = None
        self.noframe_counter = 0
        self.notrack_timeout = 5
        self.fontpath = "/etc/alternatives/fonts-japanese-gothic.ttf"
        self.skelly_sub = rospy.Subscriber("/body_tracker/skeleton", Skeleton, self.callback)
        self.image_sub = rospy.Subscriber("/camera/color/image", Image, self.image_callback)

        self.prev_neck_angle = None
        self.prev_left_shoulder_angle = None
        self.prev_left_elbow_angle = None
        self.prev_left_wrist_angle = None
        self.prev_right_shoulder_angle = None
        self.prev_right_elbow_angle = None
        self.prev_right_wrist_angle = None
        self.dev_angle = 10 #deviation angle to limit instant jumps
        self.angle_percent = 0.01 #speed of drift towards new extreme angle
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.skelly_sub, self.image_sub], 10, 0.1, allow_headerless=True)

        # self.ts.registerCallback(self.callback)

    def image_callback(self, image):
        # font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(image,"bgr8")
        except CvBridgeError as e:
            print(e)
        if self.skeleton_image is None:
            self.skeleton_image = self.cv_image
            # font = ImageFont.truetype(self.fontpath, 32)
            # img_pil = PIL_Image.fromarray(self.skeleton_image)
            # draw = ImageDraw.Draw(img_pil)
            # draw.text((15,50),u""+"トラッキング待ち", font=font, fill=(0,255,0,0))
            self.skeleton_image = self.unicode_draw(self.skeleton_image, u"トラッキング待ち", 50, self.fontpath, (10,30), (0,255,0,0))
            # np.array(img_pil)
            # cv2.putText(self.skeleton_image,"*WAITING FOR TRACKING*",(15,50), font, 2,(0,255,0),2,cv2.LINE_AA)
        if self.prev_frame is None:
            self.prev_frame = self.skeleton_image
        
        #check if tracking initiated (new frames in self.skeleton_image)
        if self.skeleton_image.shape == self.prev_frame.shape:
            difference = cv2.subtract(self.skeleton_image, self.prev_frame)
            b,g,r = cv2.split(difference)
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                self.noframe_counter += 1
            else:
                self.noframe_counter = 0
                self.prev_frame = self.skeleton_image
        else:
            self.noframe_counter = 0
            self.prev_frame = self.skeleton_image
        if self.noframe_counter > self.notrack_timeout:
            self.cv_image = self.unicode_draw(self.cv_image, u"骨格検知失敗", 60, self.fontpath, (10,30), (0,0,255,0))
            # cv2.putText(self.cv_image,"NO TRACKING LOCK",(15,50), font, 2,(0,0,255),2,cv2.LINE_AA)
            cv2.imshow("camera_output", self.cv_image)
        else:    
            cv2.imshow("camera_output", self.skeleton_image)
        cv2.waitKey(3)

    def callback(self, skeleton):
        
        # v1, v2 = self.get_joint_vectors(skeleton.joint_position_right_elbow_real, \
        #                         skeleton.joint_position_right_wrist_real, \
        #                         skeleton.joint_position_right_shoulder_real)
        # angle = self.angle_between(v1, v2)/np.pi*180

        m = self.make_matrix(skeleton.joint_orientation1_neck,\
                            skeleton.joint_orientation2_neck,\
                            skeleton.joint_orientation3_neck)
        # print(m)
        rotations = self.rotationmatrix_to_angles(m)/np.pi*180
        # print(rotations)


        image = self.cv_image
        
        #draw head and spine
        # angle = self.get_angle(skeleton.joint_position_neck_real, skeleton.joint_position_head_real, )
        neck_angle = None
        left_shoulder_angle = None
        left_elbow_angle = None
        left_wrist_angle = None
        right_shoulder_angle = None
        right_elbow_angle = None
        right_wrist_angle = None


        neck_angle = self.get_angle(skeleton.joint_position_neck_real, skeleton.joint_position_head_real, skeleton.joint_position_left_collar_real)
        # if self.prev_neck_angle is None:    
        #     self.prev_neck_angle = neck_angle
        # elif self.prev_neck_angle == -9999:
        #     self.prev_neck_angle = neck_angle
        # elif np.absolute(self.prev_neck_angle - neck_angle) > self.dev_angle:
        #     neck_angle = (self.prev_neck_angle - neck_angle)/(np.absolute(self.prev_neck_angle - neck_angle))*(np.absolute(self.prev_neck_angle - neck_angle)/20)*-1
        #     self.prev_neck_angle = neck_angle
        # else:
        #     self.prev_neck_angle = neck_angle
        self.prev_neck_angle, neck_angle = self.angle_logic(self.prev_neck_angle, neck_angle)
        image = self.drawbone(skeleton.joint_position_head_proj, skeleton.joint_position_neck_proj, image, "head-neck", neck_angle)

        image = self.drawbone(skeleton.joint_position_neck_proj, skeleton.joint_position_left_collar_proj, image, "neck-collar")
        image = self.drawbone(skeleton.joint_position_left_collar_proj, skeleton.joint_position_torso_proj, image, "collar-torso")
        image = self.drawbone(skeleton.joint_position_torso_proj, skeleton.joint_position_waist_proj, image, "torso-waist")
        
        #draw left side
        left_shoulder_angle = self.get_angle(skeleton.joint_position_left_shoulder_real, skeleton.joint_position_left_collar_real, skeleton.joint_position_left_elbow_real)
        # if self.prev_left_shoulder_angle is None:    
        #     self.prev_left_shoulder_angle = left_shoulder_angle
        # elif self.prev_left_shoulder_angle == -9999:
        #     self.prev_left_shoulder_angle = left_shoulder_angle
        # elif np.absolute(self.prev_left_shoulder_angle - left_shoulder_angle) > self.dev_angle:
        #     left_shoulder_angle = None
        # else:
        #     self.prev_left_shoulder_angle = left_shoulder_angle
        self.prev_left_shoulder_angle, left_shoulder_angle = self.angle_logic(self.prev_left_shoulder_angle, left_shoulder_angle)
        image = self.drawbone(skeleton.joint_position_left_collar_proj, skeleton.joint_position_left_shoulder_proj, image, "collar-leftshoulder", left_shoulder_angle)
                
        left_elbow_angle = self.get_angle(skeleton.joint_position_left_elbow_real, skeleton.joint_position_left_shoulder_real, skeleton.joint_position_left_wrist_real)
        # if self.prev_left_elbow_angle is None:    
        #     self.prev_left_elbow_angle = left_elbow_angle
        # elif self.prev_left_elbow_angle == -9999:
        #     self.prev_left_elbow_angle = left_elbow_angle
        # elif np.absolute(self.prev_left_elbow_angle - left_elbow_angle) > self.dev_angle:
        #     left_elbow_angle = None
        # else:
        #     self.prev_left_elbow_angle = left_elbow_angle
        self.prev_left_elbow_angle, left_elbow_angle = self.angle_logic(self.prev_left_elbow_angle, left_elbow_angle)
        image = self.drawbone(skeleton.joint_position_left_shoulder_proj, skeleton.joint_position_left_elbow_proj, image, "leftshoulder-leftelbow", left_elbow_angle)
        
        left_wrist_angle = self.get_angle(skeleton.joint_position_left_wrist_real, skeleton.joint_position_left_elbow_real, skeleton.joint_position_left_hand_real)
        # if self.prev_left_wrist_angle is None:    
        #     self.prev_left_wrist_angle = left_wrist_angle
        # elif self.prev_left_wrist_angle == -9999:
        #     self.prev_left_wrist_angle = left_wrist_angle
        # elif np.absolute(self.prev_left_wrist_angle - left_wrist_angle) > self.dev_angle:
        #     left_wrist_angle = None
        # else:
        #     self.prev_left_wrist_angle = left_wrist_angle
        self.prev_left_wrist_angle, left_wrist_angle = self.angle_logic(self.prev_left_wrist_angle, left_wrist_angle)
        image = self.drawbone(skeleton.joint_position_left_elbow_proj, skeleton.joint_position_left_wrist_proj, image, "leftelbow-leftwrist", left_wrist_angle)
        image = self.drawbone(skeleton.joint_position_left_wrist_proj, skeleton.joint_position_left_hand_proj, image, "leftwrist-lefthand")

        #draw right side
        right_shoulder_angle = self.get_angle(skeleton.joint_position_right_shoulder_real, skeleton.joint_position_left_collar_real, skeleton.joint_position_right_elbow_real)
        # if self.prev_right_shoulder_angle is None:    
        #     self.prev_right_shoulder_angle = right_shoulder_angle
        # elif self.prev_right_shoulder_angle == -9999:
        #     self.prev_right_shoulder_angle = right_shoulder_angle
        # elif np.absolute(self.prev_right_shoulder_angle - right_shoulder_angle) > self.dev_angle:
        #     right_shoulder_angle = None
        # else:
        #     self.prev_right_shoulder_angle = right_shoulder_angle
        self.prev_right_shoulder_angle, right_shoulder_angle = self.angle_logic(self.prev_right_shoulder_angle, right_shoulder_angle)    
        image = self.drawbone(skeleton.joint_position_left_collar_proj, skeleton.joint_position_right_shoulder_proj, image, "collar-rightshoulder", right_shoulder_angle)

        right_elbow_angle = self.get_angle(skeleton.joint_position_right_elbow_real, skeleton.joint_position_right_shoulder_real, skeleton.joint_position_right_wrist_real)
        # if self.prev_right_elbow_angle is None:    
        #     self.prev_right_elbow_angle = right_elbow_angle
        # elif self.prev_right_elbow_angle == -9999:
        #     self.prev_right_elbow_angle = right_elbow_angle
        # elif np.absolute(self.prev_right_elbow_angle - right_elbow_angle) > self.dev_angle:
        #     right_elbow_angle = None
        # else:
        #     self.prev_right_elbow_angle = right_elbow_angle
        self.prev_right_elbow_angle, right_elbow_angle = self.angle_logic(self.prev_right_elbow_angle, right_elbow_angle)
        image = self.drawbone(skeleton.joint_position_right_shoulder_proj, skeleton.joint_position_right_elbow_proj, image, "rightshoulder-rightelbow", right_elbow_angle)

        right_wrist_angle = self.get_angle(skeleton.joint_position_right_wrist_real, skeleton.joint_position_right_elbow_real, skeleton.joint_position_right_hand_real)
        # if self.prev_right_wrist_angle is None:    
        #     self.prev_right_wrist_angle = right_wrist_angle
        # elif self.prev_right_wrist_angle == -9999:
        #     self.prev_right_wrist_angle = right_wrist_angle
        # elif np.absolute(self.prev_right_wrist_angle - right_wrist_angle) > self.dev_angle:
        #     right_wrist_angle = None
        # else:
        #     self.prev_right_elbow_angle = right_elbow_angle
        self.prev_right_wrist_angle, right_wrist_angle = self.angle_logic(self.prev_right_wrist_angle, right_wrist_angle)
        image = self.drawbone(skeleton.joint_position_right_elbow_proj, skeleton.joint_position_right_wrist_proj, image, "rightelbow-rightwrist", right_wrist_angle)
        image = self.drawbone(skeleton.joint_position_right_wrist_proj, skeleton.joint_position_right_hand_proj, image, "rightwrist-righthand")

        if len(self.noconf_list) > 0:
            y = 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            textlength = len(u"不安定ボーン：")
            for item in self.noconf_list:
                if len(item) > textlength: textlength = len(item)
            cv2.rectangle(image,(5,5),(textlength*18, 25*(len(self.noconf_list)+2)),(0,255,0),-1)
            image = self.unicode_draw(image, u"不安定ボーン：", 32, self.fontpath, (10,y), (0,0,255,0))
            # cv2.putText(image,"Low confidence:",(10,y), font, 1,(0,0,255),1,cv2.LINE_AA)
            for key in sorted(self.noconf_list.iterkeys()):
                y += 25
                image = self.unicode_draw(image, key, 32, self.fontpath, (10,y), (0,0,255,0))
                # cv2.putText(image,key,(10,y), font, 1,(0,0,255),1,cv2.LINE_AA)
                if self.noconf_list[key] == 0:
                    del self.noconf_list[key]
                else:
                    self.noconf_list[key] -= 1
        self.skeleton_image = image
        # cv2.line(self.skeleton_image,(cols,rows),(0,0),(255,0,0),5)
        # cv2.circle(self.skeleton_image, (int(proj_elbow[0]*rows), int(proj_elbow[1]*cols)), 10, 255)
    
    def unicode_draw(self, image, text, fontsize, fontpath, coords, color):
        font = ImageFont.truetype(fontpath, fontsize)
        img_pil = PIL_Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text(coords, text, font=font, fill=color)
        image = np.array(img_pil)
        return image

    # def get_joint_distance()
# np.linalg.norm(np.array([1,2,4])-np.array([1,2,3]))


    def angle_logic(self, prev_angle, angle):
        if prev_angle is None:    
            prev_angle = angle
        if prev_angle == -9999:
            prev_angle = angle
        elif np.absolute(prev_angle - angle) > self.dev_angle:
            if angle != -9999:
                angle = prev_angle + (prev_angle - angle)/(np.absolute(prev_angle - angle))*(np.absolute(prev_angle - angle)*self.angle_percent)*-1
                prev_angle = angle
            else:
                angle = None
        else:
            prev_angle = angle
        return prev_angle, angle


    def get_angle(self, rootjoint, joint2, joint3):
        v1, v2 = self.get_joint_vectors(rootjoint, joint2, joint3)
        if str(type(v1)) == "<type 'int'>":
            angle = -9999
        else: 
            angle = self.angle_between(v1, v2)/np.pi*180
        return angle

    def drawbone(self,joint1,joint2,image,name, angle=None):
        joint1 = self.make_array(joint1)
        joint2 = self.make_array(joint2)
        (rows,cols,channels) = image.shape
        if (joint1[0] == -9999) or (joint2[0] == -9999):
            if name in self.noconf_list:
                self.noconf_list[name] += 1
            else:
                self.noconf_list[name] = 15
            return image
        else:
            cv2.circle(image, (int(joint1[0]*cols),int(joint1[1]*rows)), 8, (255,0,0), -1)
            cv2.circle(image, (int(joint2[0]*cols),int(joint2[1]*rows)), 8, (255,0,0), -1)
            if angle != None and str(type(angle)) != "<type 'int'>":
                sa = str(angle).split(".")[0] + "." + str(angle).split(".")[1][:1]
                sa = unicode(sa) + u"°"
                image = self.unicode_draw(image, sa, 28, self.fontpath, (int(joint2[0]*cols)+15,int(joint2[1]*rows)+15), (0,255,255,0))

            cv2.line(image,(int(joint1[0]*cols),int(joint1[1]*rows)),(int(joint2[0]*cols),int(joint2[1]*rows)),(255,0,0),5)
            return image

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def make_array(self, message):
        arr = np.array([message.x,message.y,message.z])
        return arr

    def get_joint_vectors(self, rootjoint, joint2, joint3):
        rootjoint = self.make_array(rootjoint)
        joint2 = self.make_array(joint2)
        joint3 = self.make_array(joint3)
        if rootjoint[0] == -9999 or joint2[0] == -9999 or joint3[0] == -9999:
            v1, v2 = -9999, -9999
        else:
            v1 = joint2 - rootjoint
            v2 = joint3 - rootjoint
        return v1, v2

    def make_matrix(self, part1, part2, part3):
        part1 = self.make_array(part1)
        part2 = self.make_array(part2)
        part3 = self.make_array(part3)
        # m = np.array([[part1[0],part2[0],part3[0]],[part1[1],part2[1],part3[1]],[part1[2],part2[2],part3[2]]])
        m = np.array([part1,part2,part3])
        # print(m)
        return m

    def is_rotation_matrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt,R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
    
    def rotationmatrix_to_angles(self, R):
        assert(self.is_rotation_matrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0] , R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x,y,z])

def main(args):
    sk = skelly()
    rospy.init_node("skeleton_processor")
    try: 
        rospy.spin()
    except KeyboardInterrupt:
        print("Exiting...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)