#!/bin/bash
sudo docker run -it --runtime=nvidia --privileged --shm-size=1g -v "$PWD":"$PWD" \
-v "/home/ddangelo/Documents/Tensorflow/doom-ckpts":"/home/ddangelo/Documents/Tensorflow/doom-ckpts" \
-w "$PWD" \
doomrl ./run.sh
