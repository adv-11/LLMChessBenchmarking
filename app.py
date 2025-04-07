import os
import time
from stockfish import Stockfish
import chess
import chess.svg
from IPython.display import display, SVG
import json
import random

# Stockfish path and initialization
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Adjust path as needed
stockfish = Stockfish(path=STOCKFISH_PATH)

# Define constants for score weights
RESULT_WEIGHT = 0.5  # Higher weight for result
MOVES_WEIGHT = 0.3   # Mid weight for number of moves
STOCKFISH_MATCH_WEIGHT = 0.2  # Mid weight for how well the LLM moves align with Stockfish
TIME_WEIGHT = 0.1  # Low weight for time taken

# Set up the board
board = chess.Board()

# Function to make a move using LLM
def make_move() -> str:
    """Random move for testing. Replace this with LLM logic."""
    moves = list(board.legal_moves)
    move = random.choice(moves)  # Replace with LLM's decision logic later
    board.push(move)
    return str(move)

# Function to get Stockfish's best move
def get_stockfish_best_move() -> str:
    """Get Stockfish's best move with 20 difficulty."""
    stockfish.set_skill_level(20)  # Set to level 20
    return stockfish.get_best_move()

# Function to evaluate the LLM's performance and calculate the final score
def evaluate_performance(llm_moves: int, result: str, matching_moves: int, time_taken: float) -> dict:
    """Evaluate the LLM's performance based on the result, number of moves, matching moves, and time taken."""
    score = 0

    # Result: 3 points for win, 1 point for draw, 0 points for loss
    result_score = 3 if result == "win" else (1 if result == "draw" else 0)

    # Number of moves: Penalize for longer games
    moves_penalty = -0.1 * (llm_moves - 20) if llm_moves > 20 else 0

    # Matching moves with Stockfish: +1 point for each match
    matching_moves_score = 0.5 * matching_moves

    # Time taken: Penalize for longer decision times
    time_penalty = -0.05 * time_taken if time_taken > 5 else 0

    # Calculate final score
    score = (RESULT_WEIGHT * result_score) + (MOVES_WEIGHT * moves_penalty) + \
            (STOCKFISH_MATCH_WEIGHT * matching_moves_score) + (TIME_WEIGHT * time_penalty)

    return {
        'score': score,
        'result': result,
        'moves': llm_moves,
        'matching_moves': matching_moves,
        'time_taken': time_taken
    }

# Function to play a game and evaluate performance
def play_game():
    """Run a full game with benchmarking."""
    board.reset()
    winner = None
    move_count = 0
    matching_moves = 0
    start_time = time.time()

    # Track moves
    game_moves = []

    while not board.is_game_over():
        # LLM makes a move (White Player)
        move_white = make_move()
        game_moves.append(('White', move_white))

        # Get Stockfish's best move
        move_black = get_stockfish_best_move()
        game_moves.append(('Black', move_black))

        # Compare LLM move with Stockfish move
        if move_white == move_black:
            matching_moves += 1

        board.push(chess.Move.from_uci(move_white))
        board.push(chess.Move.from_uci(move_black))

        move_count += 1
        print(f"Move {move_count}: {move_white} (White) vs {move_black} (Black)")

    # Game result: win/loss/draw
    result = "draw" if board.is_stalemate() else ("win" if board.turn == chess.WHITE else "loss")
    end_time = time.time()
    time_taken = end_time - start_time

    # Evaluate LLM performance
    performance = evaluate_performance(move_count, result, matching_moves, time_taken)
    
    # Prepare game details for JSON output
    game_details = {
        'moves': game_moves,
        'final_result': result,
        'performance': performance
    }

    return game_details

# Main loop for multiple games and leaderboard
def run_benchmark():
    """Run multiple games and track the leaderboard."""
    games_to_play = 2  # Adjust number of games
    final_leaderboard = []

    for game in range(games_to_play):
        print(f"Starting Game {game + 1}...")
        game_result = play_game()
        final_leaderboard.append(game_result)

        print(f"Game {game + 1} Result: {game_result['final_result']}\n")
    
    # Final leaderboard in JSON format
    final_leaderboard_json = json.dumps(final_leaderboard, indent=4)
    print("Leaderboard in JSON format:")
    print(final_leaderboard_json)

    # Save the JSON to a file
    with open('leaderboard.json', 'w') as json_file:
        json.dump(final_leaderboard, json_file, indent=4)

# Run the benchmark
run_benchmark()
