import os
import time
import random
import chess
import chess.svg
import subprocess

from pathlib import Path
from typing_extensions import Annotated
from IPython.display import display, SVG

import autogen
from autogen import ConversableAgent, register_function
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv
load_dotenv()

# Setup environment
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"

# LLM Configuration
llm_config = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": os.environ.get("ADV_OPENAI_DEV"),
        }
    ],
    "cache_seed": True
}

# Working directory for coding
workdir = Path("coding")
workdir.mkdir(exist_ok=True)

# Initialize chess board
board = chess.Board()

# Function to make a move
def make_move() -> Annotated[str, 'A move in UCI format']:
    moves = list(board.legal_moves)
    move = random.choice(moves)
    board.push(move)

    # Display the board
    svg_board = chess.svg.board(board=board, size=400)
    display(SVG(svg_board))
    return str(move)

# Setup players (agents)
player_white = ConversableAgent(
    name='player_white',
    system_message='You are playing as white. Always call make_move() function to make a move.',
    llm_config=llm_config
)

player_black = ConversableAgent(
    name='player_black',
    system_message='You are playing as black. Always call make_move() function to make a move.',
    llm_config=llm_config
)

# Setup Board Proxy (no LLM needed here)
board_proxy = ConversableAgent(
    name="board_proxy",
    llm_config=False,
    is_termination_msg=lambda msg: "tool_calls" not in msg,
)

# Register the make_move function for both players
register_function(
    make_move,
    caller=player_white,
    name='make_move',
    executor=board_proxy,
    description='Make a move in chess.'
)

register_function(
    make_move,
    caller=player_black,
    name='make_move',
    executor=board_proxy,
    description='Make a move in chess.'
)

# Setting up nested chats
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": player_white,
    }]
)

player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": player_black,
    }]
)

# Resetting the board before match start
board = chess.Board()

# Starting the match
# Let's simulate 50 moves or until the board is in a terminal state
max_moves = 50

for move_number in range(max_moves):
    if board.is_game_over():
        print(f"Game over! Result: {board.result()}")
        break
    if move_number % 2 == 0:
        player_white.initiate_chat(player_black, message="Your move, black!")
    else:
        player_black.initiate_chat(player_white, message="Your move, white!")
else:
    print("Maximum moves reached. Game drawn!")
