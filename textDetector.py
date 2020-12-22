# from cv2 import cv2
import sw
import cv2
import numpy as np
import pytesseract

img = cv2.imread('sudoku.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cong = r'--psm 6 outputbase digits'
hI, wI = img.shape
hEdge = hI//9
wEdge = wI//9

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
        tolerance = hSec//8

        section = section[tolerance:hSec-tolerance, tolerance:wSec-tolerance]
        temp = pytesseract.image_to_boxes(section, config=cong)
        try:
            matrix[i][j] = temp[0]
        except IndexError:
            matrix[i][j] = 0
print(np.matrix(sw.sudoku(matrix)))
