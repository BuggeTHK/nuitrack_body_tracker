#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from PIL import ImageFont, ImageDraw
from PIL import Image as PIL_Image
import cv2 
from cv_bridge import CvBridge, CvBridgeError
import sys

import pygame
import time
import rospy
from body_tracker_msgs.msg import Skeleton
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import message_filters
import commands

JOINT_STATE_SIM = True


class skelly:

    def __init__(self):
        # self.image_pub = rospy.Publisher("/skeleton_projection",Image, queue_size=10)
        if not JOINT_STATE_SIM:
            self.master_pub = rospy.Publisher("master_states", JointState, queue_size=1)
            self.master_state_ = JointState()
            self.master_state_.header = Header()
            self.master_state_.header.frame_id = "master_base"
            self.master_state_.name = [""] * 17
            self.master_state_.position = [0] * 17 
            self.master_state_.name[0] = "waist_y"
            self.master_state_.name[1] = "r_shoulder_p"
            self.master_state_.name[2] = "r_shoulder_r"
            self.master_state_.name[3] = "r_shoulder_y"
            self.master_state_.name[4] = "r_elbow_p"
            self.master_state_.name[5] = "r_wrist_y"
            self.master_state_.name[6] = "r_wrist_p"
            self.master_state_.name[7] = "r_wrist_r"
            self.master_state_.name[8] = "r_hand"
            self.master_state_.name[9] = "l_shoulder_p"
            self.master_state_.name[10] = "l_shoulder_r"
            self.master_state_.name[11] = "l_shoulder_y"
            self.master_state_.name[12] = "l_elbow_p"
            self.master_state_.name[13] = "l_wrist_y"
            self.master_state_.name[14] = "l_wrist_p"
            self.master_state_.name[15] = "l_wrist_r"
            self.master_state_.name[16] = "l_hand"
        else:
            self.joint_state_publisher_ = rospy.Publisher("upper_joint_states", JointState, queue_size=1)
            self.joint_state_ = JointState()
            self.joint_state_.header = Header()
            self.joint_state_.name = [""] * 28
            self.joint_state_.position = [0] * 28
            self.joint_state_.header.frame_id = "waist_link"
            self.joint_state_.name[0] = "l_elbow_joint"
            self.joint_state_.name[1] = "l_indexbase_joint"
            self.joint_state_.name[2] = "l_indexend_joint"
            self.joint_state_.name[3] = "l_indexmid_joint"
            self.joint_state_.name[4] = "l_shoulder_p_joint"
            self.joint_state_.name[5] = "l_shoulder_r_joint"
            self.joint_state_.name[6] = "l_shoulder_y_joint"
            self.joint_state_.name[7] = "l_thumb_joint"
            self.joint_state_.name[8] = "l_wrist_p_joint"
            self.joint_state_.name[9] = "l_wrist_r_joint"
            self.joint_state_.name[10] = "l_wrist_y_joint"
            self.joint_state_.name[11] = "neck_p_joint"
            self.joint_state_.name[12] = "neck_r_joint"
            self.joint_state_.name[13] = "neck_y_joint"
            self.joint_state_.name[14] = "r_elbow_joint"
            self.joint_state_.name[15] = "r_indexbase_joint"
            self.joint_state_.name[16] = "r_indexend_joint"
            self.joint_state_.name[17] = "r_indexmid_joint"
            self.joint_state_.name[18] = "r_shoulder_p_joint"
            self.joint_state_.name[19] = "r_shoulder_r_joint"
            self.joint_state_.name[20] = "r_shoulder_y_joint"
            self.joint_state_.name[21] = "r_thumb_joint"
            self.joint_state_.name[22] = "r_wrist_p_joint"
            self.joint_state_.name[23] = "r_wrist_r_joint"
            self.joint_state_.name[24] = "r_wrist_y_joint"
            self.joint_state_.name[25] = "waist_p_joint"
            self.joint_state_.name[26] = "waist_r_joint"
            self.joint_state_.name[27] = "waist_y_joint"

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
        self.prev_left_shoulder_p = None
        self.prev_left_shoulder_y = None
        self.prev_right_shoulder_p = None
        self.prev_right_shoulder_y = None
        self.prev_waist_rotation = None
        self.dev_angle = 10 #deviation angle to limit instant jumps
        self.angle_percent = 0.25 #speed of drift towards new extreme angle
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.skelly_sub, self.image_sub], 10, 0.1, allow_headerless=True)

        # self.ts.registerCallback(self.callback)
        self.running = True

        pygame.init()
        pygame.event.clear()
        self.eventclearedtime = time.time()
        pygame.display.set_caption('Controller Input')
        self.surface = None 

        pygame.mouse.set_visible(False)
        # pygame.event.set_grab(True)
        self.Rdown = False

        #checks to see if we've timed out
        self.isalive = time.time()
        
        #toggle for turning sending data off
        self.play = False
        
        #head value set by hand-controller
        self.headrot = 0
        self.headrot2 = 0
        #bow motion controller by user
        self.bowpos = [0, 0]

        #is mouse button 8 and 1 depressed
        self.mouse8down = False
        self.mouse1down = False
        
        #keep track of frames per second
        self.fps = 0
        self.fpscounter = 0
        self.second = 0

        self.averaging_length = 17 #3 is recommended
        self.averagelist = [None] * self.averaging_length

    def screenprint(self, message, surface, pos):
        font = pygame.font.Font(None, 20)
        linesize = font.get_linesize()
        position = [10, 10]
        position[1] += linesize*pos 
        image = font.render(message, 1, (0,200,0))
        surface.blit(image, position)

    def image_callback(self, image):
        # font = cv2.FONT_HERSHEY_SIMPLEX
        self.isalive = time.time()

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(image,"bgr8")
        except CvBridgeError as e:
            print(e)
        if self.skeleton_image is None:
            self.skeleton_image = self.cv_image
            self.skeleton_image = self.unicode_draw(self.skeleton_image, u"トラッキング待ち", 50, self.fontpath, (10,30), (0,255,0,0))
        if self.prev_frame is None:
            self.prev_frame = self.skeleton_image
            cv2.imshow("camera_output", self.skeleton_image)
            cv2.waitKey(3)
            time.sleep(1.5)
            self.surface = pygame.display.set_mode((300, 300))
            pygame.event.set_grab(True)

        
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
            self.cv_image = self.unicode_draw(self.cv_image, u"骨格検出失敗", 60, self.fontpath, (10,30), (0,0,255,0))
            cv2.imshow("camera_output", self.cv_image)

            for event in pygame.event.get():
                self.surface.fill((0,0,0))
                if event.type == pygame.KEYDOWN:
                    if event.scancode == 180: 
                        self.screenprint("start: 180 (repeats)", self.surface,3)
                        if not pygame.event.get_grab():
                            pygame.event.set_grab(True)
                            self.screenprint("GRAB: on", self.surface,11)
                        self.Rdown = False
                    elif event.scancode == 147: 
                        self.screenprint("select: 147 (repeats)", self.surface,4)
                        if not pygame.event.get_grab() and self.Rdown:
                            self.screenprint("QUITTING", self.surface,11)
                            self.running = False
                        else:
                            self.Rdown = False
                    elif event.scancode == 174: 
                        self.screenprint("x: 174 (repeats)", self.surface,5)
                        self.Rdown = False
                    elif event.scancode == 171: 
                        self.screenprint("y: 171 (repeats)", self.surface,6)
                        self.Rdown = False
                    elif event.scancode == 173: 
                        self.screenprint("a: 173 (repeats)", self.surface,7)
                        self.Rdown = False
                    elif event.scancode == 172: 
                        self.screenprint("b: 172 (repeats)", self.surface,8)
                        if self.Rdown:
                            pygame.event.set_grab(False)
                            self.screenprint("GRAB: off", self.surface,11)
                            self.Rdown = False
                    elif event.scancode == 123: 
                        self.screenprint("L(volume +): 123 (repeats)", self.surface,9)
                        self.Rdown = False
                    elif event.scancode == 122: 
                        self.screenprint("R(volume -): 122 (repeats)", self.surface,10)
                        self.Rdown = True
                    elif event.scancode == 9:
                        pygame.event.set_grab(False)
                        self.screenprint("GRAB: off", self.surface,11)
                        self.Rdown = False
                    elif event.scancode == 24:
                        if not pygame.event.get_grab():
                            self.screenprint("QUITTING", self.surface,11)
                            self.running = False
                            
                    else: self.screenprint(str(event.scancode), self.surface,12)
            pygame.display.flip()


        else:    
            cv2.imshow("camera_output", self.skeleton_image)
        cv2.waitKey(3)




    def callback(self, skeleton):

        self.isalive = time.time()

        image = self.cv_image
        
        if self.second == 0:
            self.second = time.time()
        elif time.time() - self.second > 1:
            self.fps = self.fpscounter
            self.fpscounter = 0
            self.second = time.time()
        else:
            self.fpscounter += 1
        image = self.unicode_draw(image, "FPS: " + str(self.fps), 32, self.fontpath, (750,0), (255,0,0,0))
        
        if self.play:
            image = self.unicode_draw(image, u"実行", 32, self.fontpath, (765,425), (0,255,0,0))
        else:
            image = self.unicode_draw(image, u"停止", 32, self.fontpath, (765,425), (0,0,255,0))

        neck_angle = None
        left_shoulder_angle = None
        left_elbow_angle = None
        left_wrist_angle = None
        right_shoulder_angle = None
        right_elbow_angle = None
        right_wrist_angle = None
        left_shoulder_p = None
        left_shoulder_y = None
        right_shoulder_p = None
        right_shoulder_y = None
        waist_rotation = None
        
        WAIST_Y = None
        RARM_SHOULDER_P = None
        RARM_SHOULDER_R = None
        RARM_SHOULDER_Y = None
        RARM_ELBOW_P = None
        RARM_WRIST_Y = None
        RARM_WRIST_R = None
        RARM_WRIST_P = None
        R_HAND = None
        LARM_SHOULDER_P = None
        LARM_SHOULDER_R = None
        LARM_SHOULDER_Y = None
        LARM_ELBOW_P = None
        LARM_WRIST_P = None
        LARM_WRIST_R = None
        LARM_WRIST_Y = None
        L_HAND = None
        
        HEAD_P = None
        HEAD_R = None
        HEAD_Y = None
        WAIST_P = None
        WAIST_R = None


        #draw head and spine
        neck_angle = self.get_angle(skeleton.joint_position_neck_real, skeleton.joint_position_head_real, skeleton.joint_position_left_collar_real)
        self.prev_neck_angle, neck_angle = self.angle_logic(self.prev_neck_angle, neck_angle)
        image = self.drawbone(skeleton.joint_position_head_proj, skeleton.joint_position_neck_proj, image, "head-neck", neck_angle)

        image = self.drawbone(skeleton.joint_position_neck_proj, skeleton.joint_position_left_collar_proj, image, "neck-collar")
        image = self.drawbone(skeleton.joint_position_left_collar_proj, skeleton.joint_position_torso_proj, image, "collar-torso")
        image = self.drawbone(skeleton.joint_position_torso_proj, skeleton.joint_position_waist_proj, image, "torso-waist")
        
        #draw left side
        left_shoulder_angle = self.get_angle(skeleton.joint_position_left_shoulder_real, skeleton.joint_position_left_collar_real, skeleton.joint_position_left_elbow_real)
        self.prev_left_shoulder_angle, left_shoulder_angle = self.angle_logic(self.prev_left_shoulder_angle, left_shoulder_angle)
        LARM_SHOULDER_R = left_shoulder_angle
        image = self.drawbone(skeleton.joint_position_left_collar_proj, skeleton.joint_position_left_shoulder_proj, image, "collar-leftshoulder", left_shoulder_angle)
                
        left_elbow_angle = self.get_angle(skeleton.joint_position_left_elbow_real, skeleton.joint_position_left_shoulder_real, skeleton.joint_position_left_wrist_real)
        self.prev_left_elbow_angle, left_elbow_angle = self.angle_logic(self.prev_left_elbow_angle, left_elbow_angle)
        LARM_ELBOW_P = left_elbow_angle
        image = self.drawbone(skeleton.joint_position_left_shoulder_proj, skeleton.joint_position_left_elbow_proj, image, "leftshoulder-leftelbow", left_elbow_angle)
        
        left_wrist_angle = self.get_angle(skeleton.joint_position_left_wrist_real, skeleton.joint_position_left_elbow_real, skeleton.joint_position_left_hand_real)
        self.prev_left_wrist_angle, left_wrist_angle = self.angle_logic(self.prev_left_wrist_angle, left_wrist_angle)
        image = self.drawbone(skeleton.joint_position_left_elbow_proj, skeleton.joint_position_left_wrist_proj, image, "leftelbow-leftwrist", left_wrist_angle)
        image = self.drawbone(skeleton.joint_position_left_wrist_proj, skeleton.joint_position_left_hand_proj, image, "leftwrist-lefthand")

        #draw right side
        right_shoulder_angle = self.get_angle(skeleton.joint_position_right_shoulder_real, skeleton.joint_position_left_collar_real, skeleton.joint_position_right_elbow_real)
        self.prev_right_shoulder_angle, right_shoulder_angle = self.angle_logic(self.prev_right_shoulder_angle, right_shoulder_angle)    
        RARM_SHOULDER_R = right_shoulder_angle
        image = self.drawbone(skeleton.joint_position_left_collar_proj, skeleton.joint_position_right_shoulder_proj, image, "collar-rightshoulder", right_shoulder_angle)

        right_elbow_angle = self.get_angle(skeleton.joint_position_right_elbow_real, skeleton.joint_position_right_shoulder_real, skeleton.joint_position_right_wrist_real)
        self.prev_right_elbow_angle, right_elbow_angle = self.angle_logic(self.prev_right_elbow_angle, right_elbow_angle)
        RARM_ELBOW_P = right_elbow_angle
        image = self.drawbone(skeleton.joint_position_right_shoulder_proj, skeleton.joint_position_right_elbow_proj, image, "rightshoulder-rightelbow", right_elbow_angle)

        right_wrist_angle = self.get_angle(skeleton.joint_position_right_wrist_real, skeleton.joint_position_right_elbow_real, skeleton.joint_position_right_hand_real)
        self.prev_right_wrist_angle, right_wrist_angle = self.angle_logic(self.prev_right_wrist_angle, right_wrist_angle)
        image = self.drawbone(skeleton.joint_position_right_elbow_proj, skeleton.joint_position_right_wrist_proj, image, "rightelbow-rightwrist", right_wrist_angle)
        image = self.drawbone(skeleton.joint_position_right_wrist_proj, skeleton.joint_position_right_hand_proj, image, "rightwrist-righthand")

        rotationdata = [[skeleton.joint_orientation1_left_shoulder, skeleton.joint_orientation2_left_shoulder,\
                        skeleton.joint_orientation3_left_shoulder, "rot_left_shoulder"],\
                        [skeleton.joint_orientation1_left_elbow, skeleton.joint_orientation2_left_elbow,\
                        skeleton.joint_orientation3_left_elbow, "rot_left_elbow"],\
                        [skeleton.joint_orientation1_right_shoulder, skeleton.joint_orientation2_right_shoulder,\
                        skeleton.joint_orientation3_right_shoulder, "rot_right_shoulder"],\
                        [skeleton.joint_orientation1_right_elbow, skeleton.joint_orientation2_right_elbow,\
                        skeleton.joint_orientation3_right_elbow, "rot_right_elbow"],\
                        [skeleton.joint_orientation1_head, skeleton.joint_orientation2_head,\
                        skeleton.joint_orientation3_head, "rot_head"],\
                        [skeleton.joint_orientation1_neck, skeleton.joint_orientation2_neck,\
                        skeleton.joint_orientation3_neck, "rot_neck"],\
                        [skeleton.joint_orientation1_waist, skeleton.joint_orientation2_waist,\
                        skeleton.joint_orientation3_waist, "rot_waist"]]
        
        rotationlist = []
        for data in rotationdata:
            rotationlist.append([self.make_matrix(data[0], data[1], data[2]),data[3]])
        

        for data in rotationlist:
            if data[0][0][0] != -9999:
                rotations = self.rotationmatrix_to_angles(data[0])/np.pi*180
            else:
                rotations = [-9999, -9999, -9999]
            if data[1] == "rot_left_shoulder":
                left_shoulder_p = rotations[0]
                self.prev_left_shoulder_p, left_shoulder_p = self.angle_logic(self.prev_left_shoulder_p, left_shoulder_p)    
                LARM_SHOULDER_P = left_shoulder_p
                # LARM_SHOULDER_R = rotations[2]
            elif data[1] == "rot_left_elbow":
                #LARM_SHOULDER_Y tracking unstable, values limited in post.#####
                left_shoulder_y = rotations[2]
                self.prev_left_shoulder_y, left_shoulder_y = self.angle_logic(self.prev_left_shoulder_y, left_shoulder_y)    
                LARM_SHOULDER_Y = left_shoulder_y
                ################################################################
                # LARM_SHOULDER_Y = 0
            elif data[1] == "rot_right_shoulder":
                right_shoulder_p = rotations[0]
                self.prev_right_shoulder_p, right_shoulder_p = self.angle_logic(self.prev_right_shoulder_p, right_shoulder_p)    
                RARM_SHOULDER_P = right_shoulder_p
            elif data[1] == "rot_right_elbow":
                #LARM_SHOULDER_Y tracking unstable, values limited in post.#####
                right_shoulder_y = rotations[2]
                self.prev_right_shoulder_y, right_shoulder_y = self.angle_logic(self.prev_right_shoulder_y, right_shoulder_y)
                RARM_SHOULDER_Y = right_shoulder_y
                ################################################################
                # RARM_SHOULDER_Y = 0
            elif data[1] == "rot_head":
                waist_rotation = rotations[1]
                self.prev_waist_rotation, waist_rotation = self.angle_logic(self.prev_waist_rotation, waist_rotation)
                WAIST_Y = waist_rotation
            elif data[1] == "rot_waist":
                pass
                    # print("rot_waist")
                    # print(str(rotations))



        #Display bones made of low-confidence joints. 
        if len(self.noconf_list) > 0:
            y = 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            textlength = len(u"不安定ボーン：")
            for item in self.noconf_list:
                if len(item) > textlength: textlength = len(item)
            cv2.rectangle(image,(5,5),(textlength*18, 25*(len(self.noconf_list)+2)),(0,255,0),-1)
            image = self.unicode_draw(image, u"不安定ボーン：", 32, self.fontpath, (10,y), (0,0,255,0))
            for key in sorted(self.noconf_list.iterkeys()):
                y += 25
                image = self.unicode_draw(image, key, 32, self.fontpath, (10,y), (0,0,255,0))
                if self.noconf_list[key] == 0:
                    del self.noconf_list[key]
                else:
                    self.noconf_list[key] -= 1
        self.skeleton_image = image

        #setting joints with value None to 0
        jointlist = [   WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
                        R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
                        L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R]
        jointlist = self.none_to_zero(jointlist)
        WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
        R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
        L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R = jointlist

        #Processing angles to match physical robot:
        if np.absolute(WAIST_Y) < 7: #deadzone for neutral position
            WAIST_Y = 0
        else: WAIST_Y = WAIST_Y*2 #exagerrating waist angle for easy control while facing screen
        RARM_SHOULDER_P += 20 #shift 20 degrees forward to match neutral position.
        LARM_SHOULDER_P += 20 #                      ""
        if np.absolute(RARM_SHOULDER_P) < 6: #deadzone for neutral position
            RARM_SHOULDER_P = 0
        if np.absolute(LARM_SHOULDER_P) < 6: #deadzone for neutral position
            LARM_SHOULDER_P = 0
        RARM_SHOULDER_P = RARM_SHOULDER_P * 1.1 #slightly exaggerating angle for easier control
        LARM_SHOULDER_P = LARM_SHOULDER_P * 1.1 #                  ""
        RARM_SHOULDER_R -= 100 #shifting angle to match neutral position
        LARM_SHOULDER_R -= 100 #                ""
        RARM_SHOULDER_R *= -1  #flipping sign to match robot
        if np.absolute(RARM_SHOULDER_R) < 7:
            RARM_SHOULDER_R = 0
        if np.absolute(LARM_SHOULDER_R) < 7:
            LARM_SHOULDER_R = 0
        RARM_SHOULDER_Y += 90 #shifting angle to match neutral position
        LARM_SHOULDER_Y -= 90 #                 ""
        RARM_SHOULDER_Y *= -1 #Flipping sign to match robot
        LARM_SHOULDER_Y *= -1 #            ""
        if np.absolute(RARM_SHOULDER_Y) < 7: #deadzone for neutral position
            RARM_SHOULDER_Y = 0
        if np.absolute(LARM_SHOULDER_Y) < 7: #deadzone for neutral position
            LARM_SHOULDER_Y = 0
        RARM_ELBOW_P -= 210 #shifting angle to match robot
        LARM_ELBOW_P -= 210 #            ""
        
        
        #applying typeF physical movement limits        
        if RARM_SHOULDER_P < -87: RARM_SHOULDER_P = -87
        if RARM_SHOULDER_P > 19: RARM_SHOULDER_P = 19
        if RARM_SHOULDER_R < -89: RARM_SHOULDER_R = -89
        if RARM_SHOULDER_R > 0: RARM_SHOULDER_R = 0
        if RARM_SHOULDER_Y < -35: RARM_SHOULDER_Y = -35
        if RARM_SHOULDER_Y > 10: RARM_SHOULDER_Y = 10
        if RARM_ELBOW_P < -180: RARM_ELBOW_P = -180
        if RARM_ELBOW_P > 0: RARM_ELBOW_P = 0
        if RARM_WRIST_Y < -360: RARM_WRIST_Y = -360
        if RARM_WRIST_Y > 360: RARM_WRIST_Y = 360
        if RARM_WRIST_P < -25: RARM_WRIST_P = -25
        if RARM_WRIST_P > 25: RARM_WRIST_P = 25
        if RARM_WRIST_R < -40: RARM_WRIST_R = -40
        if RARM_WRIST_R > 80: RARM_WRIST_R = 80
        if LARM_SHOULDER_P < -87: LARM_SHOULDER_P = -87
        if LARM_SHOULDER_P > 19: LARM_SHOULDER_P = 19
        if LARM_SHOULDER_R < 0: LARM_SHOULDER_R = 0
        if LARM_SHOULDER_R > 89: LARM_SHOULDER_R = 89
        if LARM_SHOULDER_Y < -10: LARM_SHOULDER_Y = -10
        if LARM_SHOULDER_Y > 35: LARM_SHOULDER_Y = 35
        if LARM_ELBOW_P < -180: LARM_ELBOW_P = -180
        if LARM_ELBOW_P > 0: LARM_ELBOW_P = 0
        if LARM_WRIST_Y < -360: LARM_WRIST_Y = -360
        if LARM_WRIST_Y > 360: LARM_WRIST_Y = 360
        if LARM_WRIST_P < -25: LARM_WRIST_P = -25
        if LARM_WRIST_P > 25: LARM_WRIST_P = 25
        if LARM_WRIST_R < -80: LARM_WRIST_R = -80
        if LARM_WRIST_R > 40: LARM_WRIST_R = 40
        if WAIST_Y < -45: WAIST_Y = -45
        if WAIST_Y > 45: WAIST_Y = 45
        # if WAIST_P < -9: WAIST_P = -9
        # if WAIST_P > 39: WAIST_P = 39
        if WAIST_R < -7: WAIST_R = -7
        if WAIST_R > 7: WAIST_R = 7
        # if HEAD_Y < -360: HEAD_Y = -360
        # if HEAD_Y > 360: HEAD_Y = 360
        # if HEAD_P < -20: HEAD_P = -20
        # if HEAD_P > 60: HEAD_P = 60
        if HEAD_R < -20: HEAD_R = -20
        if HEAD_R > 20: HEAD_R = 20 


        #average out values controlled by skeleton tracking:
        jointlist = [   WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
                        R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
                        L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R]
        jointlist = self.average_out(jointlist)
        WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
        R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
        L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R = jointlist



        #control head rotation with joystick
        mouse_rel = pygame.mouse.get_rel()
        print(mouse_rel)
        print(self.headrot2)
        print(self.headrot)
        if self.mouse8down and self.play:
            if mouse_rel[0] < -50:
                if self.headrot < 45:
                    self.headrot += 5
            elif mouse_rel[0] > 50:
                if self.headrot > -45:
                    self.headrot -= 5
            if mouse_rel[1] < -50:
                if self.headrot2 < 15:
                    self.headrot2 += 5
            elif mouse_rel[1] > 50:
                if self.headrot2 > -10:
                    self.headrot2 -= 5

        elif self.play:
            if self.headrot > 0:
                self.headrot -= 5
            elif self.headrot < 0:
                self.headrot += 5
            if self.headrot2 > 0:
                self.headrot2 -= 5
            elif self.headrot2 < 0:
                self.headrot2 += 5
        HEAD_Y, HEAD_P = self.headrot, self.headrot2
        
        # print(HEAD_Y)

        #function for ojigi button.
        if self.mouse1down and self.play:
            if self.bowpos[0] < 26:
                self.bowpos[0] += 2
            if self.bowpos[1] < 20:
                self.bowpos[1] += 2
        elif self.play:
            if self.bowpos[0] > 0:
                self.bowpos[0] -= 2
            if self.bowpos[1] > 0:
                self.bowpos[1] -= 2
        
        if self.headrot2 == 0:        
            HEAD_P = self.bowpos[0]
        WAIST_P = self.bowpos[1]
        # print(HEAD_P, WAIST_P)
            


        if time.time() - self.eventclearedtime > 0.5:
            pygame.event.clear()

        for event in pygame.event.get():
            self.surface.fill((0,0,0))
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.screenprint("click(trigger): 1 (no repeat)", self.surface,1)
                    self.Rdown = False
                    self.mouse1down = True
                elif event.button == 8:
                    self.screenprint("back(trigger): 8 (no repeat)", self.surface,2)
                    self.Rdown = False
                    self.mouse8down = True
                    # print(mouse_rel)
                    # print(pygame.mouse.get_rel())
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.screenprint("click(trigger) UP: 1 (no repeat)", self.surface,1)
                    self.mouse1down = False
                elif event.button == 8:
                    self.screenprint("back(trigger) UP: 8 (no repeat)", self.surface,2)
                    self.mouse8down = False
            elif event.type == pygame.MOUSEMOTION:
                self.screenprint('mouse movement: ' + str(event.rel), self.surface,0)
                # print(str(pygame.key.get_pressed()))

            elif event.type == pygame.KEYDOWN:
                if event.scancode == 180: 
                    self.screenprint("start: 180 (repeats)", self.surface,3)
                    if not pygame.event.get_grab():
                        pygame.event.set_grab(True)
                        self.screenprint("GRAB: on", self.surface,11)
                    else:
                        if not self.play:
                            self.play = True
                        else:
                            self.play = False
                    self.Rdown = False
                elif event.scancode == 147: 
                    self.screenprint("select: 147 (repeats)", self.surface,4)
                    # self.Rdown = False
                    if not pygame.event.get_grab() and self.Rdown:
                        self.screenprint("QUITTING", self.surface,11)
                        self.running = False
                    else:
                        self.Rdown = False
                elif event.scancode == 174: 
                    self.screenprint("x: 174 (repeats)", self.surface,5)
                    self.Rdown = False
                elif event.scancode == 171: 
                    self.screenprint("y: 171 (repeats)", self.surface,6)
                    self.Rdown = False
                elif event.scancode == 173: 
                    self.screenprint("a: 173 (repeats)", self.surface,7)
                    self.Rdown = False
                elif event.scancode == 172: 
                    self.screenprint("b: 172 (repeats)", self.surface,8)
                    if self.Rdown:
                        pygame.event.set_grab(False)
                        self.screenprint("GRAB: off", self.surface,11)
                        # self.Rdown = False
                elif event.scancode == 123: 
                    self.screenprint("L(volume +): 123 (repeats)", self.surface,9)
                    self.Rdown = False
                elif event.scancode == 122: 
                    self.screenprint("R(volume -): 122 (repeats)", self.surface,10)
                    self.Rdown = True
                elif event.scancode == 9:
                    pygame.event.set_grab(False)
                    self.screenprint("GRAB: off", self.surface,11)
                elif event.scancode == 24:
                    if not pygame.event.get_grab():
                        self.screenprint("QUITTING", self.surface,11)
                        self.running = False
                else: self.screenprint(str(event.scancode), self.surface,12)
        self.eventclearedtime = time.time()
        pygame.display.flip()

        if self.play:

            if not JOINT_STATE_SIM:
                jointlist = [   WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
                                R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
                                L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R]
                for j, line in enumerate(jointlist): 
                    jointlist[j] *= 10
                    jointlist[j] = int(jointlist[j])
                WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
                R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
                L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R = jointlist
                
                
                #     #head y p r:    
                # HEAD_Y,HEAD_P,HEAD_R = int(-HEAD_Y),int(-HEAD_P),int(HEAD_R)
                # #r shoulder p r y:
                # RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y = int(RARM_SHOULDER_P),int(RARM_SHOULDER_R),int(RARM_SHOULDER_Y)
                # #r elbow:
                # RARM_ELBOW_P = int(-(RARM_ELBOW_P+1800))
                # #r wrist y p r:
                # RARM_WRIST_Y,RARM_WRIST_P,RARM_WRIST_R = int(RARM_WRIST_Y),int(RARM_WRIST_P),int(-RARM_WRIST_R)
                # #waist r, r hand(no value,int(angles[12], waist y:
                # WAIST_R,R_HAND,WAIST_Y = int(WAIST_R),int(R_HAND),int(-WAIST_Y)
                # #l shoulder p r y:
                # LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y = int(LARM_SHOULDER_P),int(-LARM_SHOULDER_R),int(-LARM_SHOULDER_Y)
                # #l elbow:
                # LARM_ELBOW_P = int(-(LARM_ELBOW_P+1800))
                # #l wrist y p r:
                # LARM_WRIST_Y,LARM_WRIST_P,LARM_WRIST_R = int(-LARM_WRIST_Y),int(LARM_WRIST_P),int(LARM_WRIST_R)
                # #waist p, l hand(no value,angles[27]):
                # WAIST_P,L_HAND = int(WAIST_P),int(L_HAND)



                #additional processing to fit master format
                LARM_SHOULDER_P *= -1
                LARM_ELBOW_P += 1800
                LARM_WRIST_P *= -1
                LARM_WRIST_R *= -1

                RARM_SHOULDER_P *= -1
                RARM_SHOULDER_R *= -1
                RARM_SHOULDER_Y *= -1
                RARM_ELBOW_P += 1800
                RARM_WRIST_Y *= -1
                RARM_WRIST_P *= -1

                # print(  "master_waist: "+str(WAIST_Y)+"\n"+\
                #         "master_rarm shoulder p: "+str(RARM_SHOULDER_P)+"\n"+\
                #         "master_rarm shoulder r: "+str(RARM_SHOULDER_R)+"\n"+\
                #         "master_rarm shoulder y: "+str(RARM_SHOULDER_Y)+"\n"+\
                #         "master_rarm elbow p: "+str(RARM_ELBOW_P)+"\n"+\
                #         "master_rarm wrist p: "+str(RARM_WRIST_P)+"\n"+\
                #         "master_rarm wrist r: "+str(RARM_WRIST_R)+"\n"+\
                #         "master_rarm wrist y: "+str(RARM_WRIST_Y)+"\n"+\
                #         "master_r hand: "+str(R_HAND)+"\n"+\
                #         "master_larm shoulder p: "+str(LARM_SHOULDER_P)+"\n"+\
                #         "master_larm shoulder r: "+str(LARM_SHOULDER_R)+"\n"+\
                #         "master_larm shoulder y: "+str(LARM_SHOULDER_Y)+"\n"+\
                #         "master_larm elbow p: "+str(LARM_ELBOW_P)+"\n"+\
                #         "master_larm wrist p: "+str(LARM_WRIST_P)+"\n"+\
                #         "master_larm wrist r: "+str(LARM_WRIST_R)+"\n"+\
                #         "master_larm wrist y: "+str(LARM_WRIST_Y)+"\n"+\
                #         "master_l hand: "+str(L_HAND)+"\n"+\
                #         "master_head p: "+str(HEAD_P)+"\n"+\
                #         "master_head r: "+str(HEAD_R)+"\n"+\
                #         "master_head y: "+str(HEAD_Y)+"\n"+\
                #         "master_waist p: "+str(WAIST_P)+"\n"+\
                #         "master_waist r: "+str(WAIST_R)+"\n")

                #publishing to ROS:
                # if commands.getoutput('xset q | grep LED')[65] == '3':
                self.master_state_.position[0] = WAIST_Y
                self.master_state_.position[1] = RARM_SHOULDER_P
                self.master_state_.position[2] = RARM_SHOULDER_R
                self.master_state_.position[3] = RARM_SHOULDER_Y
                self.master_state_.position[4] = RARM_ELBOW_P
                self.master_state_.position[5] = RARM_WRIST_Y
                self.master_state_.position[6] = RARM_WRIST_P
                self.master_state_.position[7] = RARM_WRIST_R
                self.master_state_.position[8] = 0 #RARM_HAND
                self.master_state_.position[9] = LARM_SHOULDER_P
                self.master_state_.position[10] = LARM_SHOULDER_R
                self.master_state_.position[11] = LARM_SHOULDER_Y
                self.master_state_.position[12] = LARM_ELBOW_P
                self.master_state_.position[13] = LARM_WRIST_Y
                self.master_state_.position[14] = LARM_WRIST_P
                self.master_state_.position[15] = LARM_WRIST_R
                self.master_state_.position[16] = 0 #LARM_HAND
                self.master_state_.header.stamp = rospy.Time.now()
                self.master_pub.publish(self.master_state_)

            else:
                deg2rad = np.pi / 180
                jointlist = [   WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
                                R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
                                L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R]
                for j, line in enumerate(jointlist): 
                    jointlist[j] *= deg2rad
                WAIST_Y,RARM_SHOULDER_P,RARM_SHOULDER_R,RARM_SHOULDER_Y,RARM_ELBOW_P,RARM_WRIST_P,RARM_WRIST_R,RARM_WRIST_Y,\
                R_HAND,LARM_SHOULDER_P,LARM_SHOULDER_R,LARM_SHOULDER_Y,LARM_ELBOW_P,LARM_WRIST_P,LARM_WRIST_R,LARM_WRIST_Y,\
                L_HAND,HEAD_P,HEAD_R,HEAD_Y,WAIST_P,WAIST_R = jointlist
                # if commands.getoutput('xset q | grep LED')[65] == '3':
                #publishing to ROS
                self.joint_state_.position[0] = LARM_ELBOW_P	   	#"l_elbow_joint"
                self.joint_state_.position[1] = 0	               	#"l_indexbase_joint"
                self.joint_state_.position[2] = 0           	   	#"l_indexend_joint"
                self.joint_state_.position[3] = 0           	   	#"l_indexmid_joint"
                self.joint_state_.position[4] = LARM_SHOULDER_P	   	#"l_shoulder_p_joint"
                self.joint_state_.position[5] = LARM_SHOULDER_R	   	#"l_shoulder_r_joint"
                self.joint_state_.position[6] = LARM_SHOULDER_Y	   	#"l_shoulder_y_joint"
                self.joint_state_.position[7] = 0           	   	#"l_thumb_joint"
                self.joint_state_.position[8] = LARM_WRIST_P	   	#"l_wrist_p_joint"
                self.joint_state_.position[9] = LARM_WRIST_R	   	#"l_wrist_r_joint"
                self.joint_state_.position[10] = LARM_WRIST_Y	   	#"l_wrist_y_joint"
                self.joint_state_.position[11] = HEAD_P             #"neck_p_joint"
                self.joint_state_.position[12] = 0              	#"neck_r_joint"
                self.joint_state_.position[13] = HEAD_Y            	#"neck_y_joint"
                self.joint_state_.position[14] = RARM_ELBOW_P   	#"r_elbow_joint"
                self.joint_state_.position[15] = 0              	#"r_indexbase_joint"
                self.joint_state_.position[16] = 0                 	#"r_indexend_joint"
                self.joint_state_.position[17] = 0              	#"r_indexmid_joint"
                self.joint_state_.position[18] = RARM_SHOULDER_P	#"r_shoulder_p_joint"
                self.joint_state_.position[19] = RARM_SHOULDER_R	#"r_shoulder_r_joint"
                self.joint_state_.position[20] = RARM_SHOULDER_Y	#"r_shoulder_y_joint"
                self.joint_state_.position[21] = 0              	#"r_thumb_joint"
                self.joint_state_.position[22] = RARM_WRIST_P   	#"r_wrist_p_joint"
                self.joint_state_.position[23] = RARM_WRIST_R   	#"r_wrist_r_joint"
                self.joint_state_.position[24] = RARM_WRIST_Y   	#"r_wrist_y_joint"
                self.joint_state_.position[25] = WAIST_P        	#"waist_p_joint"
                self.joint_state_.position[26] = WAIST_R        	#"waist_r_joint"
                self.joint_state_.position[27] = WAIST_Y        	#"waist_y_joint"
                self.joint_state_.header.stamp = rospy.Time.now()
                self.joint_state_publisher_.publish(self.joint_state_)

                # print(  "waist: "+str(WAIST_Y)+"\n"+\
                #         "rarm shoulder p: "+str(RARM_SHOULDER_P)+"\n"+\
                #         "rarm shoulder r: "+str(RARM_SHOULDER_R)+"\n"+\
                #         "rarm shoulder y: "+str(RARM_SHOULDER_Y)+"\n"+\
                #         "rarm elbow p: "+str(RARM_ELBOW_P)+"\n"+\
                #         "rarm wrist p: "+str(RARM_WRIST_P)+"\n"+\
                #         "rarm wrist r: "+str(RARM_WRIST_R)+"\n"+\
                #         "rarm wrist y: "+str(RARM_WRIST_Y)+"\n"+\
                #         "r hand: "+str(R_HAND)+"\n"+\
                #         "larm shoulder p: "+str(LARM_SHOULDER_P)+"\n"+\
                #         "larm shoulder r: "+str(LARM_SHOULDER_R)+"\n"+\
                #         "larm shoulder y: "+str(LARM_SHOULDER_Y)+"\n"+\
                #         "larm elbow p: "+str(LARM_ELBOW_P)+"\n"+\
                #         "larm wrist p: "+str(LARM_WRIST_P)+"\n"+\
                #         "larm wrist r: "+str(LARM_WRIST_R)+"\n"+\
                #         "larm wrist y: "+str(LARM_WRIST_Y)+"\n"+\
                #         "l hand: "+str(L_HAND)+"\n"+\
                #         "head p: "+str(HEAD_P)+"\n"+\
                #         "head r: "+str(HEAD_R)+"\n"+\
                #         "head y: "+str(HEAD_Y)+"\n"+\
                #         "waist p: "+str(WAIST_P)+"\n"+\
                #         "waist r: "+str(WAIST_R)+"\n")

    def average_out(self,jointlist):
        self.averagelist.pop()
        jl = list(jointlist)
        self.averagelist.insert(0,jl)
        for j in self.averagelist:
            if j == None:
                return jointlist
        for i in range(len(jointlist)):
            jointlist[i] = 0
            for joints in self.averagelist:
                jointlist[i] += joints[i]
            jointlist[i] /= len(self.averagelist)
        return jointlist

    def none_to_zero(self, jointlist):
        for i, joint in enumerate(jointlist):
            if jointlist[i] is None:
                jointlist[i] = 0
        return jointlist
    
    def unicode_draw(self, image, text, fontsize, fontpath, coords, color):
        font = ImageFont.truetype(fontpath, fontsize)
        img_pil = PIL_Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text(coords, text, font=font, fill=color)
        image = np.array(img_pil)
        return image

    def angle_logic(self, prev_angle, angle):
        if prev_angle is None:    
            prev_angle = angle
        if prev_angle == -9999:
            prev_angle = angle
        elif np.absolute(prev_angle - angle) > self.dev_angle: #if angle has instantaneously jumped due to tracking error
            if angle != -9999:
                angle = prev_angle + (prev_angle - angle)/(np.absolute(prev_angle - angle))*(np.absolute(prev_angle - angle)*self.angle_percent)*-1
                prev_angle = angle
            else:
                angle = prev_angle
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
        m = np.array([[part1[0],part2[0],part3[0]],[part1[1],part2[1],part3[1]],[part1[2],part2[2],part3[2]]])
        #m = np.array([part1,part2,part3])
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
    while sk.running:
        time.sleep(0.1)
        if time.time() - sk.isalive > 6:
            if sk.surface != None:
                sk.surface.fill((0,0,0))
                sk.screenprint("No activity on subscribed topics detected.", sk.surface,0)
                sk.screenprint("Press 'Esc', 'Q' or 'Ctrl' button to exit.", sk.surface,1)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.scancode == 24 or event.scancode == 9 or event.scancode == 37 or event.scancode == 147:
                            sk.screenprint("QUITTING", sk.surface,11)
                            sk.running = False
                        else: sk.screenprint(str(event.scancode), sk.surface,12)
                pygame.display.flip()
        if time.time() - sk.isalive > 20:
            print("Timed out waiting for activity on subscribed topics. Exiting.")
            sk.running = False
    # pygame.quit()
    # rospy.signal_shutdown("Quit")

if __name__ == "__main__":
    main(sys.argv)