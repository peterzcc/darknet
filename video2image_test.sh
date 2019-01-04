#!/bin/zsh
cd ./data_southgate
TARGET_DIR=frames_southgate
mkdir ../${TARGET_DIR}
for video_file_name in *.mp4; do
    echo ${video_file_name}
    mkdir ../${TARGET_DIR}/${video_file_name}
    ffmpeg -i ${video_file_name} -f image2 ../${TARGET_DIR}/${video_file_name}/f%06d.jpg
done
