import os
import time
import random
import chess
import chess.svg
from pathlib import Path
from IPython.display import display, SVG
from dotenv import load_dotenv

load_dotenv()

# --- Stockfish Setup ---
from stockfish import Stockfish

# Path to your Stockfish binary
STOCKFISH_PATH = Stockfish(path="stockfish\stockfish-windows-x86-64-avx2.exe") 

# Initialize Stockfish (set a reasonable skill level)
stockfish = Stockfish(path=STOCKFISH_PATH)
stockfish.set_skill_level(10)

# --- LLM & Autogen Setup ---

from autogen import ConversableAgent, register_function, AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Use OpenAI instead of Mistral
llm_config = [{
    "model": "gpt-3.5-turbo",
    "api_key": os.environ.get("OPENAI_API_KEY"),  # ensure this is set in your env
    "api_type": "openai"
}]

# Define a working directory
workdir = Path("coding")
workdir.mkdir(exist_ok=True)

# --- Chess Board Initialization ---
board = chess.Board()

# A simple leaderboard dictionary to track cumulative LLM move quality scores.
leaderboard = {"LLM": 0}

# --- Function: Make a move using Stockfish ---
def stockfish_make_move() -> str:
    """
    Uses Stockfish to pick the best move from the current board state.
    Pushes the move and displays the board.
    """
    # Update Stockfish with the current board position
    stockfish.set_fen_position(board.fen())
    best_move = stockfish.get_best_move()
    if best_move is None:
        return "resign"
    move = chess.Move.from_uci(best_move)
    board.push(move)
    svg_board = chess.svg.board(board=board, size=400)
    display(SVG(svg_board))
    return best_move

# --- Function: Evaluate a move quality ---
def evaluate_move(move_uci: str) -> float:
    """
    Evaluates the board after applying a move (given in UCI format) and returns a score in centipawns.
    For LLM moves, we compare its evaluation against Stockfish's best move evaluation.
    A lower absolute difference means a better move.
    """
    # Save current board state
    original_fen = board.fen()

    # Make a copy of the board and push the LLM's move
    temp_board = chess.Board(original_fen)
    move = chess.Move.from_uci(move_uci)
    if move not in temp_board.legal_moves:
        # Illegal move: assign a high penalty
        return 1000.0
    temp_board.push(move)

    # Evaluate the new board state using Stockfish:
    stockfish.set_fen_position(temp_board.fen())
    eval_result = stockfish.get_evaluation()
    # eval_result is typically a dict like {'type': 'cp', 'value': 23}
    if eval_result['type'] == "cp":
        eval_cp = eval_result['value']
    else:
        # For mate scores, convert them (for simplicity, use a high value)
        eval_cp = 10000 if eval_result['value'] > 0 else -10000

    # Now, get evaluation for the best move (Stockfish’s move) from the original board:
    stockfish.set_fen_position(original_fen)
    best_move = stockfish.get_best_move()
    if best_move is None:
        best_eval = 0
    else:
        temp_board_best = chess.Board(original_fen)
        temp_board_best.push(chess.Move.from_uci(best_move))
        stockfish.set_fen_position(temp_board_best.fen())
        best_eval_result = stockfish.get_evaluation()
        best_eval = best_eval_result['value'] if best_eval_result['type'] == "cp" else (10000 if best_eval_result['value'] > 0 else -10000)

    # The quality score is the absolute difference in evaluation.
    quality_score = abs(best_eval - eval_cp)
    return quality_score

# --- Function for LLM to make a move ---
def llm_make_move() -> str:
    """
    Simulates the LLM move by asking the LLM to call the function `make_move()`.
    For now, we let the LLM generate a move in UCI format.
    Once a move is returned, we evaluate it against Stockfish’s evaluation.
    """
    # For demonstration, we prompt the LLM to choose a move.
    # In practice, autogen would allow the agent to call the 'make_move' function.
    # Here, we simulate a call by having the LLM generate a move.
    # (You can integrate your OpenAI API call here.)
    # For now, we simply input a move from the user.
    user_move = input("LLM (OpenAI) - Enter your move in UCI format (e.g., e2e4): ")
    move = chess.Move.from_uci(user_move)
    if move not in board.legal_moves:
        print("Illegal move by LLM. Try again.")
        return llm_make_move()  # re-prompt
    board.push(move)
    svg_board = chess.svg.board(board=board, size=400)
    display(SVG(svg_board))
    # Evaluate the move quality
    score = evaluate_move(user_move)
    print(f"Move quality (difference from Stockfish best move): {score} centipawns")
    # Lower score means closer to Stockfish’s evaluation (better move)
    leaderboard["LLM"] += score
    return user_move

# --- Setup Autogen Agents ---
# The Stockfish agent will simply use the stockfish_make_move() function.
stockfish_agent = ConversableAgent(
    name='Stockfish_Agent',
    system_message='You are a chess engine. Always call stockfish_make_move() to make a move.',
    llm_config=False  # Not using LLM for this agent
)

llm_agent = ConversableAgent(
    name='LLM_Player',
    system_message='You are playing chess as an LLM. Always call llm_make_move() to make your move.',
    llm_config={"config_list": llm_config, 'cache_seed': True}
)

# Board proxy agent for function execution (if needed)
board_proxy = ConversableAgent(
    name="Board_Proxy",
    llm_config=False,
    is_termination_msg=lambda msg: "tool_calls" not in msg,
)

# Register the Stockfish move function for the Stockfish agent
register_function(
    stockfish_make_move,
    caller=stockfish_agent,
    name='stockfish_make_move',
    executor=board_proxy,
    description='Stockfish engine makes a move in chess',
)

# Register the LLM move function for the LLM agent
register_function(
    llm_make_move,
    caller=llm_agent,
    name='llm_make_move',
    executor=board_proxy,
    description='LLM makes a move in chess',
)

# (Optional) Set up nested chats if needed so that agents can communicate through the board proxy.
llm_agent.register_nested_chats(
    trigger=stockfish_agent,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": llm_agent,
    }],
)

stockfish_agent.register_nested_chats(
    trigger=llm_agent,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": stockfish_agent,
    }],
)

# --- Running a Game Loop ---
# We’ll simulate a game for a fixed number of moves (for example, 6 moves: 3 each)
num_moves = 6
for move_num in range(num_moves):
    print(f"\n=== Move {move_num + 1} ===")
    print(board)
    
    if board.is_game_over():
        print("Game Over:", board.result())
        break

    if board.turn == chess.WHITE:
        print("LLM's turn (White)")
        llm_agent.initiate_chat(
            stockfish_agent,
            message="LLM's turn: Please call llm_make_move() function.",
            max_turns=1,
        )
    else:
        print("Stockfish's turn (Black)")
        stockfish_agent.initiate_chat(
            llm_agent,
            message="Stockfish's turn: Please call stockfish_make_move() function.",
            max_turns=1,
        )

print("\nFinal Board Position:")
print(board)
if board.is_game_over():
    print("Final Result:", board.result())
    
# --- Display Leaderboard Score ---
print("\nLeaderboard - Lower cumulative score indicates moves closer to Stockfish's ideal:")
print(leaderboard)
