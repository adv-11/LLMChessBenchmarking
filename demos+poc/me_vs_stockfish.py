import chess
from stockfish import Stockfish

# Initialize
board = chess.Board()
stockfish = Stockfish(path="stockfish\stockfish-windows-x86-64-avx2.exe")
stockfish.set_skill_level(10)

# Play 10 moves maximum (5 each)
for move_number in range(10):
    print(f"\nMove {move_number + 1}")
    print(board)

    if board.is_game_over():
        print("\nGame Over:", board.result())
        break

    if board.turn == chess.WHITE:
        # LLM (you) plays White
        user_move = input("Enter your move (in UCI format, e.g., e2e4): ")
        move = chess.Move.from_uci(user_move)
        if move in board.legal_moves:
            board.push(move)
        else:
            print("Illegal move, try again.")
            continue
    else:
        # Stockfish plays Black
        stockfish.set_fen_position(board.fen())
        best_move = stockfish.get_best_move()
        if best_move is None:
            print("Stockfish resigns!")
            break
        move = chess.Move.from_uci(best_move)
        if move in board.legal_moves:
            print(f"Stockfish plays: {best_move}")
            board.push(move)
        else:
            print("Stockfish made an illegal move? Unlikely. Breaking.")
            break

# Final board
print("\nFinal Board Position:")
print(board)
if board.is_game_over():
    print("\nFinal Result:", board.result())
