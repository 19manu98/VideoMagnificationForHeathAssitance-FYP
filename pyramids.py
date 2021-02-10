import cv2

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