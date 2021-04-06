# from https://www.pythonforengineers.com/image-and-video-processing-in-python/
# build a program that take a video and recognizes faces

import cv2
import os
import math
import numpy as np
import time
import pandas as pd
import matplotlib as plt
from face import recognize_face, forehead
from plotting import create_bpm_plot, plot, create_green_plot,plot_green
from rotation import check_rotation, correct_rotation
from processingSignal import processing


def Photoplethysmography(video):
    green_channel_values = []
    times = []
    buffer_size = 350
    buffer_green_mean = []
    bpms = []
    rotateCode = None
    real_fps = None
    # get a predefined video or webcam
    if video == 'webcam':
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(video)
        absPath = os.path.abspath(video)
        real_fps, rotateCode = check_rotation(absPath)

    index = 0

    while True:
        # get the first frame and get the face
        return_code, last_image = video_capture.read()
        if rotateCode is not None:
            last_image = correct_rotation(last_image, rotateCode)
        faces_last = recognize_face(last_image)
        if len(faces_last) > 0:
            break
        cv2.imshow("Analysing video", last_image)
        k = cv2.waitKey(1)
        # esc = 27
        if k == 27:
            cv2.destroyAllWindows()
            return
        if k == 103:
            try:
                fig1.close()
            except:
                try:
                    plt.close(fig1)
                    axes1, fig1 = create_green_plot()
                    plot_green(dfg, fig1, axes1)
                except:
                    try:
                        axes1, fig1 = create_green_plot()
                        plot_green(dfg, fig1, axes1)
                    except:
                        pass



    # assign the values to variables
    xl, yl, hl = faces_last[0][0], faces_last[0][1], faces_last[0][3]
    top_left = (xl, yl)
    bottom_right = (xl + hl, yl + hl)
    forehead_x1, forehead_x2, forehead_y1, forehead_y2 = forehead(xl,yl,hl)

    while True:

        while True:
            # get the first frame and get the face
            return_code, current_image = video_capture.read()
            if rotateCode is not None:
                current_image = correct_rotation(current_image, rotateCode)
            if return_code:
                faces_current = recognize_face(current_image)
                if len(faces_current) > 0:
                    break
                else:
                    cv2.imshow("Analysing video", current_image)

                    k = cv2.waitKey(1)

                    if k == 27:
                        cv2.destroyAllWindows()
                        exit()
                    if k == 103:
                        try:
                            plt.close(fig1)
                            axes1, fig1 = create_green_plot()
                            plot_green(dfg, fig1, axes1)
                        except:
                            try:
                                axes1, fig1 = create_green_plot()
                                plot_green(dfg, fig1, axes1)
                            except:
                                pass
            else:
                break

        if return_code:
            xc, yc, hc = faces_current[0][0], faces_current[0][1], faces_current[0][3]
            face = current_image[yc:yc + hc,xc:xc + hc]
            face = cv2.resize(face,None,fx=0.5,fy=0.5)
            rwidth = face.shape[0]
            height, width = current_image.shape[:2]
            swidth = width - rwidth - 10
            current_image[10:10+rwidth,swidth:width-10] = face

            try:
                change_top_left = math.sqrt(((xl - xc) ** 2) + (yl - yc) ** 2)
                change_bottom_right = math.sqrt(((xl + hl - xc - hc) ** 2) + (yl + hl - yc - hc) ** 2)
            except:
                pass

            if (change_top_left > 8 or change_bottom_right > 10):
                try:
                    top_left = (xc, yc)
                    bottom_right = (xc + hc, yc + hc)
                    forehead_x1, forehead_x2, forehead_y1, forehead_y2 = forehead(xc,yc,hc)
                except:
                    pass

            cv2.rectangle(current_image, top_left, bottom_right, (0, 255, 0), 2)
            forehead_region = current_image[forehead_y1: forehead_y2, forehead_x1:forehead_x2]
            b,g,r = cv2.split(forehead_region)
            g_mean = np.mean(g,dtype=np.float64)
            green_channel_values.append(g_mean)
            times.append(time.time())
            buffer_green_mean.append(g_mean)
            current_size = len(buffer_green_mean)
            dfg = pd.DataFrame({'x': range(0, len(green_channel_values)), 'green': green_channel_values})
            if current_size > buffer_size:
                index+=1
                times = times[1:]
                buffer_green_mean = buffer_green_mean[1:]

            if (current_size == (buffer_size+1)):
                bpm = processing(buffer_green_mean,times,buffer_size,real_fps)
                bpms.append(int(bpm))
                string = 'BPM: ' + str(int(bpm))
                cv2.putText(current_image, string, (top_left[0]+10,top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.rectangle(current_image, (forehead_x1,forehead_y1),(forehead_x2,forehead_y2),(0,0,255),2)

            # capture frame-by-frame
            image_last = current_image
            faces_last = faces_current
            xl, yl, hl = xc, yc, hc

            cv2.imshow("Analysing video", current_image)

            k = cv2.waitKey(1)

            if k==27:
                break
            if k == 103:
                try:
                    plt.close(1)
                    axes1, fig1 = create_green_plot()
                    plot_green(dfg, fig1, axes1)
                except:
                    try:
                        axes1, fig1 = create_green_plot()
                        plot_green(dfg, fig1, axes1)
                    except:
                        pass

        else:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    for i in range(1, 5):
        cv2.waitKey(1)
    return

    # When everything is done, release the capture
    video_capture.release()




