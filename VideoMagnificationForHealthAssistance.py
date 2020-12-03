# from https://www.pythonforengineers.com/image-and-video-processing-in-python/
# build a program that take a video and recognizes faces

import cv2
import sys
import math

global change_top_left
global change_bottom_right

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
    print(difference_bottom_right)
    print(difference_top_left)
    average_change_top_left = sum(difference_top_left)/len(difference_top_left)
    average_change_bottom_right = sum(difference_bottom_right) / len(difference_bottom_right)
    print(average_change_top_left)
    print(average_change_bottom_right)


if __name__ == "__main__":
    main()



