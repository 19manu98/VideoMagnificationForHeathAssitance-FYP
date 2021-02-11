# from https://www.pythonforengineers.com/image-and-video-processing-in-python/
# build a program that take a video and recognizes faces

import cv2
import sys
import math
import numpy as np
import pandas as pd
import time
from scipy import signal
from scipy.signal import butter,filtfilt
from face import recognize_face, forehead
import matplotlib.pyplot as plt
from plotting import create_bpm_plot, plot

def main():
    red_channel_values = []
    green_channel_values = []
    blue_channel_values = []
    times = []
    buffer_size = 250
    buffer_green_mean = []
    bpms = []

    # get a predefined video or webcam
    if len(sys.argv) < 2:
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(sys.argv[1])

    index = 0

    axes, fig = create_bpm_plot()
    fig.show()

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
    forehead_x1, forehead_x2, forehead_y1, forehead_y2 = forehead(xl,yl,hl)

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

                    k = cv2.waitKey(33)
                    if k == 27:
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

            cv2.rectangle(current_image, top_left, bottom_right, (0, 255, 0), 2)
            forehead_region = current_image[forehead_y1: forehead_y2, forehead_x1:forehead_x2]
            b,g,r = cv2.split(forehead_region)
            b_mean = np.mean(b)
            r_mean = np.mean(r)
            g_mean = np.mean(g)
            red_channel_values.append(r_mean)
            green_channel_values.append(g_mean)
            blue_channel_values.append(b_mean)
            times.append(time.time())
            buffer_green_mean.append(g_mean)
            current_size = len(buffer_green_mean)

            if current_size > buffer_size:
                index+=1
                times = times[1:]
                buffer_green_mean = buffer_green_mean[1:]


            if (current_size == (buffer_size+1)):
                # calculate real fps regarding processor
                real_fps = float(current_size) / (times[-1]-times[0])

                # signal detrending
                signal_detrend = signal.detrend(buffer_green_mean)

                #butterworth filter
                timeLF = times[-1]-times[0]
                cutoff = 0.059

                nyq = 0.5 * real_fps
                order = 2
                n = int(timeLF * real_fps)

                normal_cutoff = cutoff / nyq
                b, a = butter(order,normal_cutoff,btype='low',analog=True)
                signal_detrend = filtfilt(b,a,signal_detrend)

                # signal interpolation
                even_times = np.linspace(times[0],times[-1],current_size)
                interp = np.interp(even_times,times,signal_detrend)
                signal_interpolated = np.hamming(current_size) * interp
                signal_interpolated = signal_interpolated - np.mean(signal_interpolated)


                # normalize the signal
                #signal_normalization = signal_interpolated/np.linalg.norm(signal_interpolated)
                signal_normalization = signal_interpolated/np.std(signal_interpolated)
                #fast fourier transform
                raw_signal = np.fft.fft(signal_normalization)
                fft = np.abs(raw_signal)

                # #freqs = float(real_fps)/current_size*np.arange(current_size/2+1)
                freqs = np.fft.rfftfreq(current_size,1./real_fps)
                freqs = 60. * freqs

                idx = np.where((freqs >36) & (freqs<120))

                pruned = fft[idx]

                pfreq = freqs[idx]
                freqs = pfreq

                idx2 = np.argmax(pruned)
                bpm = freqs[idx2]

                # print(bpm)
                bpms.append(bpm)
                # dataframe
                df = pd.DataFrame({'x': range(0, index), 'bpm': bpms})
                plot(df,fig,axes)


            cv2.rectangle(current_image, (forehead_x1,forehead_y1),(forehead_x2,forehead_y2),(0,0,255),2)

            # capture frame-by-frame
            image_last = current_image
            faces_last = faces_current
            xl, yl, hl = xc, yc, hc

            cv2.imshow("Faces found", current_image)

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


if __name__ == "__main__":
    main()



