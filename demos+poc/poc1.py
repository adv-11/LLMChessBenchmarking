import chess
import chess.engine
from stockfish import Stockfish

# Initialize a chess board
board = chess.Board()

# Initialize Stockfish
stockfish = Stockfish(path="stockfish\stockfish-windows-x86-64-avx2.exe")
stockfish.set_skill_level(10)

# Play 5 moves from the starting position
for move_number in range(5):
    print(f"\nMove {move_number + 1}")
    print(board)

    # Set the board position in Stockfish
    stockfish.set_fen_position(board.fen())

    # Get best move from Stockfish
    best_move = stockfish.get_best_move()

    if best_move is None:
        print("Game over or no moves available.")
        break

    print(f"Stockfish plays: {best_move}")

    # Apply the move on the board
    move = chess.Move.from_uci(best_move)
    if move in board.legal_moves:
        board.push(move)
    else:
        print("Illegal move detected!")
        break

# Final board position
print("\nFinal Board Position:")
print(board)
