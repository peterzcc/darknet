#!/bin/zsh
SOURCE_DIR=frames_output
TARGET_DIR=videos_output
FRAME_RATE=15
rm -r ${TARGET_DIR}
mkdir ${TARGET_DIR}
cd ${SOURCE_DIR}
for video_file_name in *.mp4; do
    echo ${video_file_name}
    #mkdir ../${TARGET_DIR}/${video_file_name}
    #ffmpeg -i ${video_file_name} -f image2 ../${TARGET_DIR}/${video_file_name}/f%06d.jpg
    ffmpeg -framerate ${FRAME_RATE} -i ${video_file_name}/f%06d.jpg -vf "fps=${FRAME_RATE},format=yuv420p" ../${TARGET_DIR}/${video_file_name} &
done
wait
