# CREDIT
# https://aakashjhawar.medium.com/sudoku-solver-using-opencv-and-dl-part-1-490f08701179
# https://becominghuman.ai/sudoku-and-cell-extraction-sudokuai-opencv-38b603066066
import sw
import cv2
import numpy as np
import pytesseract
import operator

imagePath = 'sudoku.png'
img = cv2.imread(imagePath)
img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (9, 9), 0)
img = cv2.adaptiveThreshold(img, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 3, 2)
img = cv2.bitwise_not(img, img)
cv2.imshow("Source", img)
cv2.waitKey(0)
contours, h = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
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


def distance(p1, p2):
    a = p2[0]-p1[0]
    b = p2[1]-p1[1]
    return np.sqrt((a**2)+(b**2))


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

# TODO
# Insert image again with dimensions given prev
# Sections must be legible to the machine learning algo
img = cv2.imread(imagePath)
img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

img = cv2.warpPerspective(img, m, (int(side), int(side)))

cv2.imshow("Source", img)
cv2.waitKey(0)

cong = r'--psm 6 outputbase digits'
hI, wI = img.shape
hEdge = hI//9
wEdge = wI//9

# Just to split the image into 81 squares
# check CREDIT for info
grid = []
for i in range(hEdge, hI + 1, hEdge):
    for j in range(wEdge, wI + 1, wEdge):
        rows = img[i-hEdge:i]
        grid.append([rows[k][j - wEdge:j] for k in range(len(rows))])

finalG = []
for i in range(0, len(grid)-8, 9):
    finalG.append(grid[i:i+9])

matrix = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        section = np.array(finalG[i][j])
        hSec, wSec = section.shape
        tolerance = min(hSec, wSec)//8
        section = section[tolerance:hSec-tolerance, tolerance:wSec-tolerance]
        temp = pytesseract.image_to_boxes(section, config=cong)
        try:
            matrix[i][j] = temp[0]
        except IndexError:
            matrix[i][j] = 0
        print(matrix[i][j])
        cv2.imshow("Source", section)
        cv2.waitKey(0)

print(np.matrix(sw.sudoku(matrix)))
