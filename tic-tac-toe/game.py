# Board Function to create the state of the board
def BoardState(board):
    print("Current State Of Board : \n")
    for i in range(0, 9):
        if (i > 0) and (i % 3) == 0:
            print("\n")
        if board[i] == 0:
            print("- ", end=" ")
        if board[i] == 1:
            print("O ", end=" ")
        if board[i] == -1:
            print("X ", end=" ")
    print("\n")

# This function takes the user move as input and makes the required changes on the board.
def User1Turn(board):
    pos = input("Enter X's position from [1...9]: ")
    pos = int(pos)
    if board[pos-1] != 0:
        print("Wrong Move!!!")
        exit(0)
    board[pos-1] = -1

# MinMax function.
def minimax(board, player):
    x = Winningboard(board)
    if x != 0:
        return (x * player)
    pos = -1
    value = -2
    for i in range(0, 9):
        if board[i] == 0:
            board[i] = player
            score = -minimax(board, (player * -1))
            if score > value:
                value = score
                pos = i
            board[i] = 0

    if pos == -1:
        return 0
    return value

# This function makes the computer's move using the minimax algorithm.
def AiTurn(board):
    pos = -1
    value = -2
    for i in range(0, 9):
        if board[i] == 0:
            board[i] = 1
            score = -minimax(board, -1)
            board[i] = 0
            if score > value:
                value = score
                pos = i
    board[pos] = 1

# This function is used to analyze a game.
def Winningboard(board):
    cb = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

    for i in range(0, 8):
        if (board[cb[i][0]] != 0 and
            board[cb[i][0]] == board[cb[i][1]] and
            board[cb[i][0]] == board[cb[i][2]]):
            return board[cb[i][2]]
    return 0

# Main Function.
def main():
    print("Computer : O Vs. You : X")
    player = input("Enter to play 1(st) or 2(nd): ")
    player = int(player)
    
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3x3 board

    for i in range(0, 9):
       
        if Winningboard(board) != 0:
            break
       
        if (i + player) % 2 == 0:
            AiTurn(board)
        else:
            BoardState(board)
            User1Turn(board)
        
        # Display the board after each move
        BoardState(board)
    
    x = Winningboard(board)
    if x == 0:
        print("Draw!!!")
    elif x == -1:
        print("X Wins!!! O Loses!!!")
    elif x == 1:
        print("X Loses!!! O Wins!!!")

main()
