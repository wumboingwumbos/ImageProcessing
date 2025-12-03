# create a checkerboard image and save it
import cv2
import numpy as np
def create_checkerboard(square_size=50, num_squares=8):
    board_size = square_size * num_squares
    checkerboard = np.zeros((board_size, board_size), dtype=np.uint8)

    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                checkerboard[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255

    cv2.imwrite('checkerboard.bmp', checkerboard)
    return checkerboard
create_checkerboard()