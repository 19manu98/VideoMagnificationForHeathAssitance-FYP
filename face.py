import cv2

def recognize_face(image):
    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + casc_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(150, 150))

    return faces

def forehead(x,y,h):
    forehead_x1 = x + int(h * 0.25)
    forehead_x2 = x + int(h * 0.70)
    forehead_y1 = y + int(h * 0.10)
    forehead_y2 = y + int(h * 0.20)
    return forehead_x1, forehead_x2, forehead_y1, forehead_y2