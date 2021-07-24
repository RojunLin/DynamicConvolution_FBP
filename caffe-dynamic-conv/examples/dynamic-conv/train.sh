#!/usr/bin/env sh
set -e

TOOLS=../../build/tools


GLOG_logtostderr=0 GLOG_log_dir=./log/ $TOOLS/caffe train --solver=./alexnet-solver-dy.prototxt
