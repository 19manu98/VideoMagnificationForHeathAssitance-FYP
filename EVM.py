# from https://www.pythonforengineers.com/image-and-video-processing-in-python/
# build a program that take a video and recognizes faces

import cv2
import math
import numpy as np
import pandas as pd
import time
from scipy import fftpack
from face import recognize_face, forehead
from pyramids import gaussinan_pyramid, gaussinan_video_amplification,laplacian_pyramid
from plotting import create_bpm_plot, plot, create_green_plot,plot_green
import os
from rotation import check_rotation, correct_rotation
from processingSignal import processing

def EVM(video):
    red_channel_values = []
    green_channel_values = []
    blue_channel_values = []
    times = []
    faces = []
    pyramids = []
    buffer_size = 250
    buffer_green_mean = []
    bpms = []
    rotateCode = None
    real_fps = None
    # get a predefined video or webcam
    if video == "webcam":
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(video)
        absPath = os.path.abspath(video)
        real_fps, rotateCode = check_rotation(absPath)

    index = 0

    axes, fig = create_bpm_plot()
    fig.show()

    while True:
        # get the first frame and get the face
        return_code, last_image = video_capture.read()
        if rotateCode is not None:
            last_image = correct_rotation(last_image, rotateCode)
        faces_last = recognize_face(last_image)
        if len(faces_last) > 0:
            break
        cv2.imshow("Analysing video", last_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # assign the values to variables
    xl, yl, hl = faces_last[0][0], faces_last[0][1], faces_last[0][3]
    if hl %2 == 1:
        hl += 1
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

                    k = cv2.waitKey(33)
                    if k == 27:
                        break
            else:
                break

        if return_code:
            xc, yc, hc = faces_current[0][0], faces_current[0][1], faces_current[0][3]
            if hc % 2 == 1:
                hc += 1
            # current_image[10:10+rwidth,swidth:width-10] = face
            axes1, fig1 = create_green_plot()
            try:
                change_top_left = math.sqrt(((xl - xc) ** 2) + (yl - yc) ** 2)
                change_bottom_right = math.sqrt(((xl + hl - xc - hc) ** 2) + (yl + hl - yc - hc) ** 2)
            except:
                pass

            if (change_top_left > 6 or change_bottom_right > 8):
                try:
                    top_left = (xc, yc)
                    bottom_right = (xc + hc, yc + hc)
                    forehead_x1, forehead_x2, forehead_y1, forehead_y2 = forehead(xc,yc,hc)
                except:
                    pass

            xc, yc = top_left
            xc1, yx1 = bottom_right
            face = current_image[yc:yx1, xc:xc1]
            face = cv2.resize(face, None, fx=0.5, fy=0.5)
            rwidth = face.shape[0]
            height, width = current_image.shape[:2]
            swidth = width - rwidth - 10
            final_video = current_image

            faces.append(face)
            gaussianPyramid = gaussinan_pyramid(face, 7)
            pyramids.append(gaussianPyramid[-1])

            if len(pyramids) > 30:
                frames_arr = np.asarray(pyramids[-30:], dtype=np.float64)
                fft = fftpack.fft(frames_arr, axis=0)
                frequencies = fftpack.fftfreq(frames_arr.shape[0], d=1.0 / 12)
                bound_low = (np.abs(frequencies - 0.5)).argmin()
                bound_high = (np.abs(frequencies - 3.1)).argmin()
                fft[:bound_low] = 0
                fft[bound_high:-bound_high] = 0
                fft[-bound_low:] = 0

                iff = fftpack.ifft(fft, axis=0)

                fft = np.abs(iff)
                amplifiedFrame = fft*100

                final_video = np.zeros(current_image.shape)
                img = amplifiedFrame[-1]
                for x in range(7):
                    size = (amplifiedFrame[x].shape[1], amplifiedFrame[x].shape[0])
                    img = cv2.pyrUp(img,dstsize = size)
                img = img+current_image
                final_video = img

                forehead_region = final_video[forehead_y1: forehead_y2, forehead_x1:forehead_x2]
                b, g, r = cv2.split(forehead_region)
                b_mean = np.mean(b)
                r_mean = np.mean(r)
                g_mean = np.mean(g)
                # current_image[10:10 + rwidth, swidth:width - 10] = final_video
                final_video = cv2.convertScaleAbs(final_video)


                red_channel_values.append(r_mean)
                green_channel_values.append(g_mean)
                blue_channel_values.append(b_mean)
                times.append(time.time())
                buffer_green_mean.append(g_mean)
                axes1, fig1 = create_green_plot()
                current_size = len(buffer_green_mean)
                dfg = pd.DataFrame({'x': range(0, current_size), 'green': buffer_green_mean})
                plot_green(dfg,fig1,axes1)
                if current_size > buffer_size:
                    index+=1
                    times = times[1:]
                    buffer_green_mean = buffer_green_mean[1:]



                if (current_size == (buffer_size+1)):
                    bpm = processing(buffer_green_mean, times, buffer_size, real_fps)
                    bpms.append(int(bpm))
                    print(bpm)
                    string = 'BPM: ' + str(int(bpm))
                    cv2.putText(current_image, string, (top_left[0] + 10, top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # dataframe
                    df = pd.DataFrame({'x': range(0, index), 'bpm': bpms})
                    plot(df,fig,axes)


                cv2.rectangle(final_video, (forehead_x1,forehead_y1),(forehead_x2,forehead_y2),(0,0,255),2)
                amplified_face = final_video[yc:yc + hc,xc:xc + hc]
                current_image[yc:yc + hc, xc:xc +hc] = amplified_face

            cv2.imshow("Analysing video", current_image)
            # capture frame-by-frame
            image_last = current_image
            faces_last = faces_current
            xl, yl, hl = xc, yc, hc



            k = cv2.waitKey(1)

            if k==27:
                break

        else:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    for i in range(1, 5):
        cv2.waitKey(1)
    return

    # When everything is done, release the capture
    video_capture.release()


