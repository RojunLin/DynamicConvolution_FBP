#!/usr/bin/env sh
TOOLS=~/caffe-a/build/tools

SHUFFLE=True
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi
echo "Creating train leveldb..."
rm -rf ./256_train_leveldb

GLOG_logtostderr=1 $TOOLS/convert_image.bin \
    --backend=leveldb  \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=$SHUFFLE \
    ../faces/    \
    ./train_1.txt \
    ./256_train_leveldb

echo "creating mean file..."
$TOOLS/compute_image_mean --backend=leveldb ./256_train_leveldb ./256_train_mean.binaryproto

