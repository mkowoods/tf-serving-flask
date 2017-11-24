#!/usr/bin/env bash

#INSTALL BAZEL https://docs.bazel.build/versions/master/install.html

sudo apt-get install -y openjdk-8-jdk

#should be one time set-up need to test if already exists
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install -y bazel

sudo apt-get upgrade -y bazel


#INSTALL GRPC https://github.com/grpc/grpc/tree/master/src/python/grpcio

sudo pip install grpcio

#INSTALL Other prereqs

sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev


#INSTALL Model Server

#below line only needs to occur one time
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install -y tensorflow-model-server

sudo apt-get upgrade -y tensorflow-model-server