from stockfish import Stockfish

# Give the path to your downloaded Stockfish engine
stockfish = Stockfish(path="stockfish\stockfish-windows-x86-64-avx2.exe")

# Optional: Set the skill level (0-20, 20 is strongest)
stockfish.set_skill_level(10)

# Set a position using FEN (this is the starting position)
stockfish.set_fen_position("r1bqkbnr/pppppppp/n7/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")

# Ask Stockfish for the best move
best_move = stockfish.get_best_move()
print("Best move:", best_move)