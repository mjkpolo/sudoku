# Credit
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
# https://aakashjhawar.medium.com/sudoku-solver-using-opencv-and-dl-part-1-490f08701179
# https://becominghuman.ai/sudoku-and-cell-extraction-sudokuai-opencv-38b603066066

import pytesseract
import cv2
import sw
import operator
import numpy as np
import sys


# Display Image
def displayImg(img):
    cv2.imshow("sudoku", img)
    cv2.waitKey(1)


# Preprocessing
def preproc(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # first param is histogram which we don't need
    _, img = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return img


def distance(p1, p2):
    a = p2[0]-p1[0]
    b = p2[1]-p1[1]
    return np.sqrt((a**2)+(b**2))


def warp(img, returnImg):
    # Find largest contour which would be the puzzle
    contours, h = cv2.findContours(
            img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) == 0:
        print("no countour found exiting...")
        sys.exit(1)
    polygon = contours[0]

    # key=operator.itemgetter(index) compares
    # the VALUE of the minimum NOT index
    # without itemgetter the index given by enum would be compared
    # Enum is only used to get the index to each var
    bRight, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
                    key=operator.itemgetter(1))
    bLeft, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
                   key=operator.itemgetter(1))
    tRight, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
                    key=operator.itemgetter(1))
    tLeft, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
                   key=operator.itemgetter(1))

    corners = [polygon[tLeft][0],
               polygon[tRight][0],
               polygon[bRight][0],
               polygon[bLeft][0]]

    src = np.array([corners[0],
                    corners[1],
                    corners[2],
                    corners[3]],
                   dtype='float32')

    side = max([distance(corners[0], corners[1]),
                distance(corners[1], corners[2]),
                distance(corners[2], corners[3]),
                distance(corners[3], corners[0])])

    dst = np.array([[0, 0],
                    [side - 1, 0],
                    [side - 1, side - 1],
                    [0, side - 1]],
                   dtype='float32')

    m = cv2.getPerspectiveTransform(src, dst)
    returnImg = cv2.warpPerspective(returnImg, m, (int(side), int(side)))
    return returnImg


def extractNums(img):
    # split the image into 81 squares
    hI, wI = img.shape
    hEdge = hI//9
    wEdge = wI//9

    # check CREDIT for info
    grid = []
    for i in range(hEdge, hI + 1, hEdge):
        for j in range(wEdge, wI + 1, wEdge):
            rows = img[i-hEdge:i]
            grid.append([rows[k][j - wEdge:j]
                        for k in range(len(rows))])

    finalG = []
    for i in range(0, len(grid)-8, 9):
        finalG.append(grid[i:i+9])

    cong = r'--psm 6 outputbase digits'
    matrix = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            section = np.array(finalG[i][j])
            hSec, wSec = section.shape
            tolerance = min(hSec, wSec)//6
            section = section[tolerance:hSec-tolerance,
                              tolerance:wSec-tolerance]
            section = cv2.copyMakeBorder(section, tolerance, tolerance,
                    tolerance, tolerance, cv2.BORDER_CONSTANT, value=0)

            # TODO Manipulate image so it is more legible
            # Inverts and blurs image once again
            # Saves back to finalG
            section = preproc(section)
            finalG[i][j] = section 
            displayImg(finalG)
            return matrix
#            temp = pytesseract.image_to_boxes(section, config=cong)
#            try:
#                matrix[i][j] = temp[0]
#            except IndexError:
#                matrix[i][j] = 0
#            print(matrix[i][j])
#            displayImg(section)
#    return matrix


# Import Image 0 means grayscale
# change image and see it fail with low res pictures!
img = cv2.imread('webSudoku.png', 0)
displayImg(img)

# manipulate image
img = preproc(img)
displayImg(img)

img = warp(img, img)
displayImg(img)

print(sw.sudoku(extractNums(img)))
