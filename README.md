# Gesture recognition for Robot gripper control

The goal of our project is to identify hand gestures in order to control the end effector motion of a robotic arm. The idea is to regulate the closure of the gripper and its final position by reproducing the motion of the hand, and also to command a variety of movements.

We used a pre-trained Convolutional Neural Network from the Pytorch model zoo to assure a better performance with a well tested CNN instead of training a new one from scratch. In particular we used the the MobileNetv2 for a better performance with a live interaction through the pc webcam. 

At the end of the process, we used classification from the output of the model to generate commands for the robot arm and gripper.
