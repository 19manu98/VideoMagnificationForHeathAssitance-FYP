# from https://www.pythonforengineers.com/image-and-video-processing-in-python/
# build a program that take a video and recognizes faces
import cv2
import threading
import math
import glob
import csv

def recognize_face(image):
    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + casc_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    # The min size makes possible that the algorithm recognizes just the faces that have a size at least (100*100)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(400, 400))

    return faces

def writeCsv(name,data):
    with open(name,'a') as outcsv:
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['Parameter 1', 'Parameter 2', 'Top-Left Absolute difference', 'Bottom-Right Absolute difference'])
        for row in data:
            writer.writerow(row)

class VideoThread(threading.Thread):
    maxRetries = 20

    def __init__(self, thread_id, name, video_url, thread_lock):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.video_url = video_url
        self.thread_lock = thread_lock
        self.video_list = []

    def run(self):
        print("starting thread" + self.name)
        video_list = []
        for val1 in range(5, 11):
            for val2 in range(5, 11):
                video_capture = cv2.VideoCapture(self.video_url)

                while True:
                    # get the first frame and get the face
                    return_code, last_image = video_capture.read()
                    if return_code:
                        faces_last = recognize_face(last_image)
                        if len(faces_last) > 0:
                            break
                    else:
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
                            break
                    if return_code:
                        xc, yc, hc = faces_current[0][0], faces_current[0][1], faces_current[0][3]

                        try:
                            change_top_left = math.sqrt(((xl - xc) ** 2) + (yl - yc) ** 2)
                            change_bottom_right = math.sqrt(((xl + hl - xc - hc) ** 2) + (yl + hl - yc - hc) ** 2)
                            if (change_top_left < val1 or change_bottom_right < val2):
                                difference_top_left.append(0)
                                difference_bottom_right.append(0)
                            else:
                                difference_top_left.append(change_top_left)
                                difference_bottom_right.append(change_bottom_right)
                        except:
                            pass

                        if (change_top_left > val1 or change_bottom_right > val2):
                            try:
                                top_left = (xc, yc)
                                bottom_right = (xc + hc, yc + hc)
                            except:
                                pass

                        cv2.rectangle(current_image, top_left, bottom_right, (0, 255, 0), 2)

                        # capture frame-by-frame
                        image_last = current_image
                        faces_last = faces_current
                        xl, yl, hl = xc, yc, hc

                    else:
                        break

                # When everything is done, release the capture
                video_capture.release()
                cv2.destroyAllWindows()

                average_change_top_left = sum(difference_top_left) / len(difference_top_left)
                average_change_bottom_right = sum(difference_bottom_right) / len(difference_bottom_right)
                self.video_list.append([val1,val2,average_change_top_left,average_change_bottom_right])
        print('finished thread' + self.name)

def main():
    files = glob.glob("videos/test11.mp4")
    thread_lock = threading.Lock()
    thread1 = VideoThread(1, "thread 1", "videos/test1.mp4", thread_lock)
    thread2 = VideoThread(1, "thread 2", "videos/test2.mp4", thread_lock)
    thread3 = VideoThread(1, "thread 3", "videos/test3.mp4", thread_lock)
    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()
    returnfile1 = thread1.video_list
    returnfile2 = thread2.video_list
    returnfile3 = thread3.video_list
    writeCsv("test1.csv",returnfile1)
    writeCsv("test2.csv", returnfile2)
    writeCsv("test3.csv", returnfile3)

if __name__ == "__main__":
    main()
