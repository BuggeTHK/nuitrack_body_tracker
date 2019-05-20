# メカレスマスター
### インストール：
以下のNuitrack Body TrackerのInstallation Instructionsを参照。    
下記コマンドでツールをインストールする必要あり。
```
sudo apt-get install python-pygame
sudo apt-get install bluez-tools
```

### 起動方法：
```
~/ros/kinetic/src/nuitrack_body_tracker/scripts/bringup.sh
```

### コントローラ：
<img src="https://user-images.githubusercontent.com/31090954/57743704-17d4b780-7701-11e9-804a-5f8e8fe16443.png" width="600" height="600">

### Bluetooth：
```
bluetoothctl
```
で使用するアダプターとコントローラをペアリング。

内蔵BTアダプターをパソコン起動時にオフにするには
```
gedit ~/ros/kinetic/src/nuitrack_body_tracker/scripts/bluetoothpowerdown.sh
```
を実行、アダプタID（08:D4:0C:3D:6D:18）をオフにしたいアダプターのIDに変えます。
```
cp ~/ros/kinetic/src/nuitrack_body_tracker/scripts/bluetoothpowerdown.sh.desktop ~/.config/autostart/bluetoothpowerdown.sh.desktop
```
上記を実行して、スクリプトを起動時に実行させます。
# FPSの動きのスムージング   
  skeleton_processor.py #153
  ```
  self.averaging_length = 10 #0.5s of frames is recommended example for 20fps
  ```
  

# Nuitrack Body Tracker

# Info
   This is a ROS Node to provide functionality of the NuiTrack SDK (https://nuitrack.com)
   
   - NuiTrack is NOT FREE, but reasonably inexpensive, and works with a number of cameras (see their webpage)
   - Also see Orbbec Astra SDK as a possible alternative (but only for Orbbec cameras)

   Publishes 3 messages:
   
   1. body_tracking_position_pub_ custom message:  <body_tracker_msgs::BodyTracker>
   Includes:
   2D position of person relative to head camera; allows for fast, smooth tracking
     Joint.proj:  position in normalized projective coordinates
     (x, y from 0.0 to 1.0, z is real)
     Astra Mini FOV: 60 horz, 49.5 vert (degrees)

   3D position of the shoulder joint in relation to robot (using TF)
     Joint.real: position in real world coordinates
     Useful for tracking person in 3D
   
   2. body_tracking_skeleton_pub_ custom message: <body_tracker_msgs::Skeleton>
   Includes:
   Everyting in BodyTracker message above, plus 3D position of upper body 
   joints in relation to robot (using TF)

   3. marker_pub_  message: <visualization_msgs::Marker>
   Publishes 3d markers for selected joints.  Visible as markers in RVIZ.


# Installation Instructions / Prerequisites

## Follow instructions on Nuitrack website
  - http://download.3divi.com/Nuitrack/doc/Installation_page.html

### Summary instructions below, but bits might not be up to date!  Go to the website to get the latest SDK.


  - Clone body_tracker_msgs into your Catkin workspace and build as below:     
    ```
    roscd 
    cd ../src
    git clone https://github.com/hi-kondo/body_tracker_msgs
    catkin build body_tracker_msgs
    ```
    - catkin_make to confirm complies OK

  - Remove OpenNI - it conflicts with the version supplied by Nuitrack!
    -   sudo apt-get purge --auto-remove openni-utils

  - Download BOTH the nuitrack linux drivers and the Nuitrack SDK
    - Download nuitrack-ubuntu-amd64.deb from [here](http://download.3divi.com/Nuitrack/doc/Installation_page.html) 
    - ダウンロードしたフォルダ名を覚えておくこと!ドライバのインストール時に同フォルダでの作業が必須
    
  - Install Nuitrack Linux drivers:
    -   cd ~/Downloads/ #Default
    -   sudo dpkg -i nuitrack-ubuntu-amd64.deb
    -   sudo reboot
    -   confirm environment variables set correctly:
        - echo $NUITRACK_HOME    (should be /usr/etc/nuitrack)
        - echo $LD_LIBRARY_PATH  (should include /usr/local/lib/nuitrack)

  - Install Nuitrack SDK (NuitrackSDK.zip)
    - Download from: http://download.3divi.com/Nuitrack/
    - mkdir ~/sdk
    - cp NuitrackSDK.zip ~/sdk
    - extract ZIP archive with ubuntu Archive Manager (double click the zip file)
    - delete the zip file

  - Clone this project into your Catkin workspace
    ```
    roscd 
    cd ../src
    git clone https://github.com/hi-kondo/mechaless_master
    catkin build nuitrack_body_tracker
    ```
    - Edit CMakeLists.txt if you installed the SDK to a different location:    
    set(NUITRACK_SDK_PATH /home/system/sdk/NuitrackSDK)

# NOTE: Nuitrack install may break other RealSense applications!
  - See this discussion: 
    - https://community.nuitrack.com/t/nuitrack-prevents-other-realsense-apps-from-working/893
    - if needed: export LD_LIBRARY_PATH="{$LD_LIBRARY_PATH}:/usr/local/lib/nuitrack"


# Option: Edit Nuitrack Config parameters (now done in the node code)
  - sudo vi $NUITRACK_HOME/data/nuitrack.config
  - set Faces.ToUse and DepthProvider.Depth2ColorRegistration to true

# Test NuiTrack SDK
  - If you run into errors, it is probably becuse you did not reboot after driver install

  - Try the license tool, it will test the system:
    - sudo -E nuitrack_license_tool
    - click on "compatibility test"    
      →何かしら表示されたらOK.
    - if you have a license, enter it after the test completes.    
      →"Get available Licenses"の横にライセンス番号を入力して、"Get available Licenses"を押す

  - Follow instructions at: ~/sdk/NuitrackSDK/Examples/nuitrack_gl_sample/README.txt

# ボタン配置を変えたい場合
/scripts/skeleton_processor.pyの``event.scancode``の番号を変える    
デバイスの各番号割り付けを知りたい場合はデバイスを繋いだ状態で``xinput test [device ID]``を実行する

