# from https://www.pythonforengineers.com/image-and-video-processing-in-python/
# build a program that take a video and recognizes faces

import cv2
import sys
import math


def recognize_face(image):
    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + casc_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    return faces


def main():
    # get a predefined video or webcam
    if len(sys.argv) < 2:
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(sys.argv[1])

    # get the first 2 frames and get the face
    return_code, last_image = video_capture.read()
    faces_last = recognize_face(last_image)

    top_left = (faces_last[0][0],faces_last[0][1])
    bottom_right = (faces_last[0][0]+faces_last[0][3],faces_last[0][1]+faces_last[0][3])

    return_code, current_image = video_capture.read()
    faces_current = recognize_face(current_image)

    # calculate the different magnitude between the faces
    difference_top_left = []
    difference_bottom_right = []

    while True:
        # capture frame-by-frame
        image_last = current_image
        faces_last = faces_current

        return_code, current_image = video_capture.read()

        if not return_code:
            break

        faces_current = recognize_face(current_image)

        try:
            change_top_left = math.sqrt(((faces_last[0][0]-faces_current[0][0])**2)+(faces_last[0][1]-faces_current[0][1])**2)
            change_bottom_right = math.sqrt(((faces_last[0][0] + faces_last[0][3]-faces_current[0][0]-faces_current[0][3])**2)
                                            +(faces_last[0][1]+faces_last[0][3]-faces_current[0][1]-faces_current[0][3])**2)
            if(change_top_left < 6 or change_bottom_right < 8):
                difference_top_left.append(0)
                difference_bottom_right.append(math.sqrt(0))
            else:
                difference_top_left.append(change_top_left)
                difference_bottom_right.append(change_bottom_right)
        except:
            pass


        if (change_top_left > 6 or change_bottom_right > 8):
            try:
                top_left = (faces_current[0][0], faces_current[0][1])
                bottom_right = (faces_current[0][0] + faces_current[0][3], faces_current[0][1] + faces_current[0][3])
            except:
                pass

        cv2.rectangle(current_image, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow("Faces found", current_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

    average_change_top_left = sum(difference_top_left)/len(difference_top_left)
    average_change_bottom_right = sum(difference_bottom_right) / len(difference_bottom_right)
    print(average_change_top_left)
    print(average_change_bottom_right)


if __name__ == "__main__":
    main()



