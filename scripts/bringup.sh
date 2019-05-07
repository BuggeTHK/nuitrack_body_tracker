#!/bin/bash
source /home/seed/ros/kinetic/devel/setup.bash
cd ~/ros/kinetic/src/nuitrack_body_tracker/scripts/

gnome-terminal --tab -e 'bash -c "source /home/seed/ros/kinetic/devel/setup.bash; roslaunch teleonoid robot_bringup.launch"' --tab -e 'bash -c "sleep 3; source /home/seed/ros/kinetic/devel/setup.bash; roslaunch nuitrack_body_tracker nuitrack_body_tracker.launch"' --tab -e 'bash -c "sleep 6; source /home/seed/ros/kinetic/devel/setup.bash; rosrun nuitrack_body_tracker skeleton_processor.py"' 
sleep 12
./placewindows
