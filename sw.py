def sudoku(board):
    (x, y) = find_empty_cell(board)
    if (x, y) == (-1, -1):
        return board
    for i in range(1, 10):
        if valid(x, y, i, board):
            board[x][y] = i
            if sudoku(board) is board:
                return board
            board[x][y] = 0


def valid(x, y, n, board):
    # check row and column
    for i in range(9):
        if board[x][i] == n or board[i][y] == n:
            return False

    # check box
    new_x = x//3 * 3
    new_y = y//3 * 3
    for i in range(3):
        for j in range(3):
            if board[new_x + i][new_y + j] == n:
                return False
    return True


def finished(board):
    if find_empty_cell(board) == (-1, -1):
        return True


def find_empty_cell(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return (-1, -1)
