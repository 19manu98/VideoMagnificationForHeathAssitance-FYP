# from https://www.pythonforengineers.com/image-and-video-processing-in-python/
# build a program that take a video and recognizes faces

import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy import signal

def gaussinan_pyramid(image, level):
    copy = image.copy()
    gPyramid = [copy]
    for i in range(level):
        copy = cv2.pyrDown(copy)
        gPyramid.append(copy)
    return gPyramid

def gaussinan_video_amplification(image):
    for i in range(image.shape[0]):
        frame = image[i]

def laplacian_pyramid(image, level):
    gPyramid = gaussinan_pyramid(image,level)
    lPyramid = []
    for i in range(level,0,-1):
        gElement = cv2.pyrUp(gPyramid[i])
        lElement = cv2.subtract(gPyramid[i-1],gElement)
        lPyramid.append(lElement)
    return lPyramid

def recognize_face(image):
    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + casc_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    return faces


def main():
    red_channel_values = []
    green_channel_values = []
    blue_channel_values = []
    times = []
    buffer_size = 150
    buffer_green_mean = []

    # get a predefined video or webcam
    if len(sys.argv) < 2:
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(sys.argv[1])
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(fps)
    while True:
        # get the first frame and get the face
        return_code, last_image = video_capture.read()
        faces_last = recognize_face(last_image)
        if len(faces_last) > 0:
            break
        cv2.imshow("Faces found", last_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # assign the values to variables
    xl, yl, hl = faces_last[0][0], faces_last[0][1], faces_last[0][3]
    top_left = (xl, yl)
    bottom_right = (xl + hl, yl + hl)

    # calculate the different magnitude between the faces
    difference_top_left = []
    difference_bottom_right = []
    while True:

        while True:
            # get the first frame and get the face
            return_code, current_image = video_capture.read()
            if return_code:
                faces_current = recognize_face(current_image)
                if len(faces_current) > 0:
                    break
                else:
                    cv2.imshow("Faces found", current_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break

        if return_code:
            xc, yc, hc = faces_current[0][0], faces_current[0][1], faces_current[0][3]
            face = current_image[yc:yc + hc,xc:xc + hc]
            face = cv2.resize(face,None,fx=0.5,fy=0.5)
            rwidth = face.shape[0]
            height, width = current_image.shape[:2]
            swidth = width - rwidth -10
            current_image[10:10+rwidth,swidth:width-10] = face

            gPyramid = gaussinan_pyramid(face,3)

            try:
                change_top_left = math.sqrt(((xl - xc) ** 2) + (yl - yc) ** 2)
                change_bottom_right = math.sqrt(((xl + hl - xc - hc) ** 2) + (yl + hl - yc - hc) ** 2)
                if (change_top_left < 6 or change_bottom_right < 8):
                    difference_top_left.append(0)
                    difference_bottom_right.append(0)
                else:
                    difference_top_left.append(change_top_left)
                    difference_bottom_right.append(change_bottom_right)
            except:
                pass

            if (change_top_left > 6 or change_bottom_right > 8):
                try:
                    top_left = (xc, yc)
                    bottom_right = (xc + hc, yc + hc)
                except:
                    pass

            cv2.rectangle(current_image, top_left, bottom_right, (0, 255, 0), 2)
            forehead_x1 = xc + int(hc * 0.45)
            forehead_x2 = xc + int(hc * 0.55)
            forehead_y1 = yc + int(hc * 0.1)
            forehead_y2 = yc + int(hc * 0.25)
            forehead_region = current_image[forehead_y1: forehead_y2, forehead_x1:forehead_x2]
            b,g,r = cv2.split(forehead_region)
            b_mean = np.mean(b)
            g_mean = np.mean(g)
            r_mean = np.mean(r)
            red_channel_values.append(r_mean*5)
            green_channel_values.append(g_mean*5)
            blue_channel_values.append(b_mean*5)
            current_size = len(buffer_green_mean)
            times.append(time.time())
            buffer_green_mean.append(g_mean)
            if current_size > buffer_size:
                times = times[1:]
                buffer_green_mean = buffer_green_mean[1:]

                # calculate real fps regarding processor
                real_fps = current_size / (times[-1]-times[0])

                # signal detrending
                signal_detrend = signal.detrend(buffer_green_mean)

                # signal interpolation
                even_times = np.linspace(times[0],times[-1],current_size)
                interp = np.interp(even_times,times,signal_detrend)
                signal_interpolated = np.hamming(current_size) * interp

                # normalize the signal
                signal_normalization = signal_interpolated/np.linalg.norm(signal_interpolated)

            cv2.rectangle(current_image, (forehead_x1,forehead_y1),(forehead_x2,forehead_y2),(0,0,255),2)
            # capture frame-by-frame
            image_last = current_image
            faces_last = faces_current
            xl, yl, hl = xc, yc, hc

            cv2.imshow("Faces found", current_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print(len(blue_channel_values))
    # dataframe
    df = pd.DataFrame({'x':range(0,len(signal_normalization)),'signal_normalized':signal_normalization})

    # style
    plt.style.use('seaborn-darkgrid')

    # palette
    palette = plt.get_cmap('Set1')

    # multiple line plot
    num = 0
    for column in df.drop('x',axis=1):
        num += 1
        plt.plot(df['x'],df[column],marker='',color=palette(num),linewidth=1,alpha=0.9,label=column)

    # Add legend
    plt.legend(loc=2,ncol=2)

    # Add titles
    plt.title("Change of the different channels")
    plt.xlabel("Frames")
    plt.ylabel("Channel value")
    plt.show()
    average_change_top_left = sum(difference_top_left)/len(difference_top_left)
    average_change_bottom_right = sum(difference_bottom_right) / len(difference_bottom_right)
    print(average_change_top_left)
    print(average_change_bottom_right)


if __name__ == "__main__":
    main()



