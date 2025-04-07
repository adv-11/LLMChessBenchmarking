import os
import time
import json
import chess
import chess.svg
from IPython.display import display, SVG
import random
from stockfish import Stockfish
import openai

# Set your OpenAI API key from the environment variable
openai.api_key = os.environ.get("ADV_OPENAI_DEV")

# Stockfish configuration (maximum difficulty)
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Adjust as needed
stockfish = Stockfish(path=STOCKFISH_PATH)
stockfish.set_skill_level(20)

# Scoring weights and parameters
RESULT_WEIGHT = 0.5          # Weight for game result (win/draw/loss)
MOVES_WEIGHT = 0.3           # Penalty for long games (moves beyond 20)
MATCHING_MOVES_WEIGHT = 0.2  # Bonus for each move matching Stockfish's best move
TIME_WEIGHT = 0.1           # Penalty for longer decision times

# Initialize the chess board
board = chess.Board()

def llm_make_move(conversation_history: str) -> str:
    """
    Generate the next move for White using GPT-4, using the conversation history as context.
    The prompt instructs the LLM to play to win, returning only the best move in UCI format.
    """
    prompt = (
        f"You are a chess grandmaster playing as White. Your objective is to win. "
        f"Here is the move history so far:\n{conversation_history}\n"
        "Based on the current board, provide your best next move in UCI format (e.g., e2e4) with no extra commentary."
    )
    
    # Using the new ChatCompletion API format:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a chess grandmaster. Provide only the best move in UCI format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    move = response["choices"][0]["message"]["content"].strip()
    return move

def get_stockfish_best_move() -> str:
    """
    Get Stockfish's best move (for Black) based on the current board state.
    Stockfish is set to maximum difficulty (level 20).
    """
    stockfish.set_fen_position(board.fen())
    move = stockfish.get_best_move()
    return move

def evaluate_performance(llm_moves: int, result: str, matching_moves: int, time_taken: float) -> dict:
    """
    Improved scoring function:
      - Game Result: 3 points for win, 1 for draw, 0 for loss.
      - Moves: Penalty of -0.1 for each move beyond 20 moves.
      - Matching moves: Bonus of +0.5 for each time the LLM's move matches Stockfish's best move.
      - Time: Penalty of -0.05 per second if total decision time exceeds 5 seconds.
    """
    result_score = 3 if result == "win" else (1 if result == "draw" else 0)
    moves_penalty = -0.1 * (llm_moves - 20) if llm_moves > 20 else 0
    matching_moves_score = 0.5 * matching_moves
    time_penalty = -0.05 * time_taken if time_taken > 5 else 0

    score = (RESULT_WEIGHT * result_score) + (MOVES_WEIGHT * moves_penalty) + \
            (MATCHING_MOVES_WEIGHT * matching_moves_score) + (TIME_WEIGHT * time_penalty)

    return {
        'score': score,
        'result': result,
        'moves': llm_moves,
        'matching_moves': matching_moves,
        'time_taken': time_taken
    }

def play_game():
    """
    Plays a complete game between the LLM (White) and Stockfish (Black).
    The LLM receives the conversation history (move list so far) as context.
    Returns a dictionary with the move list, final result, and performance metrics.
    """
    board.reset()
    move_count = 0
    matching_moves = 0
    start_time = time.time()
    conversation_history = ""  # For storing the move history
    game_moves = []  # List of tuples: (Color, move)

    while not board.is_game_over():
        # LLM's move (White)
        move_white = llm_make_move(conversation_history)
        if move_white is None:
            break
        # Validate move legality; if illegal, fallback to a random legal move
        if chess.Move.from_uci(move_white) not in board.legal_moves:
            legal_moves = list(board.legal_moves)
            move_white = random.choice(legal_moves).uci() if legal_moves else None
        game_moves.append(("White", move_white))
        conversation_history += f"White: {move_white}\n"
        board.push(chess.Move.from_uci(move_white))
        
        if board.is_game_over():
            break
        
        # Stockfish's move (Black)
        move_black = get_stockfish_best_move()
        game_moves.append(("Black", move_black))
        conversation_history += f"Black: {move_black}\n"
        board.push(chess.Move.from_uci(move_black))
        
        # Matching move bonus if LLM's move exactly matches Stockfish's best move
        if move_white == move_black:
            matching_moves += 1
        
        move_count += 1
        print(f"Move {move_count}: {move_white} (White) vs {move_black} (Black)")

    # Determine the final result from White's perspective.
    # Note: This simple logic assumes that if board.turn is White at game over, then Black just moved and White is winning.
    result = "draw" if board.is_stalemate() else ("win" if board.turn == chess.WHITE else "loss")
    end_time = time.time()
    time_taken = end_time - start_time
    performance = evaluate_performance(move_count, result, matching_moves, time_taken)
    
    game_details = {
        'moves': game_moves,
        'final_result': result,
        'performance': performance
    }
    return game_details

def run_benchmark():
    """
    Runs multiple games, calculates the average score for the LLM over all games,
    and outputs the results in JSON format with a top-level 'total_score_avg' field.
    """
    games_to_play = 10  # Adjust as needed
    final_leaderboard = []
    total_score = 0

    for game in range(games_to_play):
        print(f"Starting Game {game + 1}...")
        game_result = play_game()
        final_leaderboard.append(game_result)
        total_score += game_result['performance']['score']
        print(f"Game {game + 1} Result: {game_result['final_result']}\n")
    
    avg_score = total_score / games_to_play if games_to_play else 0
    output = {
        'total_score_avg': avg_score,
        'games': final_leaderboard
    }
    final_leaderboard_json = json.dumps(output, indent=4)
    print("Leaderboard in JSON format:")
    print(final_leaderboard_json)
    with open('leaderboard.json', 'w') as json_file:
        json.dump(output, json_file, indent=4)

# Run the benchmark
run_benchmark()
