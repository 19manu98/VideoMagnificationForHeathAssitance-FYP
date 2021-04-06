import unittest
import cv2
import src
from src.face import recognize_face,forehead
from src.rotation import check_rotation
from src.processingSignal import processing

class testPhotoplethismography(unittest.TestCase):
    def test_RecognizeFace(self):
        img = cv2.imread("Face.jpg")
        self.assertEqual(len(recognize_face(img)),1)

    def test_RegnonizeFace2(self):
        img = cv2.imread("noFace.jpg")
        self.assertEqual(len(recognize_face(img)), 0)

    def test_RecognizeForehead(self):
        img = cv2.imread("Face.jpg")
        xl, yl, hl ,h2l= (recognize_face(img)[0])
        # xl = 653
        # yl = 230
        # hl = 248
        # h2l = 248
        # Round the pixel down
        # x1 = 653 + 248 * 0.25 = 714
        # x2 = 653 + 248 * 0.70 = 827
        # y1 = 230 + 248 * 0.10 = 255
        # y2 = 230 + 248 * 0.20 = 280
        self.assertEqual(forehead(xl,yl,hl), (714,827,255,280))

    def test_checkRotation(self):
        # when manually checked the metatags, fps was equal to 0 and the rotation tag was not included
        self.assertEqual(check_rotation("../videos/test1-1.mp4"),(10.0,None))

    def test_checkRotation2(self):
        # # when manually checked the metatags, fps was equal to 19.916666666666668 and the rotation tag 0
        self.assertEqual(check_rotation("../videos/test6-1.mp4"),(19.916666666666668,0))

    def test_Processing(self):
        bufferGreen = [158,158,158,158,158,158.14]
        times = []
        buffer_size = [6]
        real_fps = None
        self.assertEqual(processing(bufferGreen,times,buffer_size,real_fps),0)

