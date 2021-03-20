#https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting

import ffmpeg
import cv2


def check_rotation(path_video_file):

    # this returns meta-data of the video file in form of a dictionary
    meta_description = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_description['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_description['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_description['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    real_fps = meta_description['steams'][0]['index']['r_frame_rate']
    return real_fps, rotateCode


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)