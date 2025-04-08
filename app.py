# Part 1: Imports, Constants, and Class Initialization

import os
import time
import json
import random
from typing import Optional, List, Dict, Any
import numpy as np
import chess
import chess.engine
import chess.pgn
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import matplotlib.pyplot as plt
import logging

# Load environment variables from the .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
OPENAI_API_KEY = os.environ.get("ADV_OPENAI_DEV")
if not OPENAI_API_KEY:
    print("Warning: ADV_OPENAI_DEV environment variable not set. LLM calls will fail.")
    # Or raise an error: raise ValueError("ADV_OPENAI_DEV environment variable not set.")

# Path to stockfish engine - Ensure this path is correct for your system
STOCKFISH_PATH = os.environ.get("STOCKFISH_EXECUTABLE_PATH", "stockfish/stockfish-windows-x86-64-avx2.exe")
if not os.path.exists(STOCKFISH_PATH):
    print(f"Warning: Stockfish executable not found at {STOCKFISH_PATH}. Engine calls will fail.")
    # Or raise an error: raise FileNotFoundError(f"Stockfish executable not found at {STOCKFISH_PATH}")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chess_benchmark")

# --- Position Sets for Training/Benchmarking ---

# Common opening positions (FEN strings) for standard benchmarking
OPENING_POSITIONS = {
    "Standard": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Sicilian Defense": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "French Defense": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "Caro-Kann": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "Kings Indian": "rnbqkb1r/pppppp1p/5np1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "Queens Gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
}

# Simpler positions for initial training/testing (Requirement 3)
SIMPLE_ENDGAMES = {
    "KPvK_1": "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1", # King and Pawn vs King
    "KPvK_2": "8/8/8/8/4k3/8/p7/K7 b - - 0 1", # King and Pawn vs King (Black to move)
    "KRPvK": "8/8/8/8/8/k7/8/KR6 w - - 0 1",   # King, Rook, Pawn vs King (Checkmate practice)
    "KBPvK": "8/8/8/8/k7/8/1P6/K1B5 w - - 0 1", # King, Bishop, Pawn vs King
    "KNPvK": "8/8/8/8/k7/8/1P6/K1N5 w - - 0 1", # King, Knight, Pawn vs King
}

# More complex tactical positions (Example set for Requirement 3)
TACTICAL_MIDDLEGAMES = {
    "Fork Opportunity": "r1bqkb1r/ppp2ppp/2n1pn2/3p4/3P1B2/4PN2/PPP2PPP/RN1QKB1R w KQkq - 0 4", # Potential fork tactics
    "Pin Example": "rnbqkbnr/ppp1pppp/8/8/3p4/1PN5/P1PPPPPP/R1BQKBNR b KQkq - 0 2", # Example pin situation
    "Discovered Attack Setup": "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PPQ1PPPP/R1B1KB1R w KQ - 0 6", # Setting up discovered attacks
}

# Positions requiring deeper strategic understanding (Example set for Requirement 3)
COMPLEX_OPENINGS_MIDDLEGAMES = {
    "Closed Sicilian": "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 5",
    "Benoni Defense": "rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "Grunfeld Defense": "rnbqkb1r/pp1p1ppp/5n2/2pP4/8/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 5",
}

# Combine all position dictionaries for easier reference if needed
ALL_POSITION_SETS = {
    "SIMPLE_ENDGAMES": SIMPLE_ENDGAMES,
    "TACTICAL_MIDDLEGAMES": TACTICAL_MIDDLEGAMES,
    "COMPLEX_OPENINGS_MIDDLEGAMES": COMPLEX_OPENINGS_MIDDLEGAMES,
    "STANDARD_OPENINGS": OPENING_POSITIONS
}


class ChessBenchmark:
    def __init__(self, llm_model="gpt-4", temperature=0.0, stockfish_depth=15, elo_k_factor=32):
        """
        Initialize the Chess Benchmark system.

        Args:
            llm_model (str): The LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo").
            temperature (float): Temperature parameter for the LLM (0.0 for deterministic).
            stockfish_depth (int): Default depth for Stockfish engine analysis.
            elo_k_factor (int): K-factor for Elo rating calculation.
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.stockfish_depth = stockfish_depth
        self.elo_k_factor = elo_k_factor

        # Initialize the LLM client
        if not OPENAI_API_KEY:
             logger.error("OpenAI API Key (ADV_OPENAI_DEV) is not set. Cannot initialize LLM.")
             # Depending on desired behavior, you might raise an error or allow operation without LLM
             # raise ValueError("OpenAI API Key (ADV_OPENAI_DEV) is not set.")
             self.chat = None # Set chat to None if API key is missing
        else:
            try:
                self.chat = ChatOpenAI(
                    model_name=llm_model,
                    temperature=temperature,
                    api_key=OPENAI_API_KEY
                    # Consider adding max_tokens if needed
                    # max_tokens=100
                )
                logger.info(f"Initialized ChatOpenAI with model: {llm_model}, temp: {temperature}")
            except Exception as e:
                logger.error(f"Failed to initialize ChatOpenAI: {e}")
                self.chat = None # Set chat to None on initialization failure

        # Initialize Stockfish engine check
        self.stockfish_available = False
        if os.path.exists(STOCKFISH_PATH):
            try:
                # Try opening the engine to ensure it works
                with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
                    pass # Just check if it opens
                self.stockfish_available = True
                logger.info(f"Stockfish engine found and accessible at: {STOCKFISH_PATH}")
            except Exception as e:
                logger.error(f"Stockfish engine at {STOCKFISH_PATH} could not be opened: {e}")
        else:
            logger.warning(f"Stockfish executable not found at {STOCKFISH_PATH}. Engine-related functions will be limited.")


        # Statistics tracking - enhanced
        self.stats = {
            "games_played": 0,
            "llm_wins": 0,
            "stockfish_wins": 0,
            "draws": 0,
            "illegal_moves_llm": 0,  # Track LLM illegal moves specifically
            "blunders_llm": 0,       # Track LLM blunders specifically
            "total_llm_move_time": 0.0,
            "total_llm_moves": 0,
            "opening_success": 0, # Count games where a known opening was played by LLM
            "game_history": [],      # Stores detailed data per game
            "total_game_plies": 0,   # Sum of plies (half-moves) across all games
            "eval_history_all": [],  # Stores eval history tuples (game_idx, move_num, eval)
            "elo_rating": 1500,      # Starting Elo rating for the LLM
            "elo_history": [(0, 1500)], # Track Elo changes over games (game_num, elo)
            "recent_blunders": [],   # Stores last N blunders for prompt context (Requirement 2)
            "blunder_positions": [] # Stores positions where blunders occurred (Requirement 5)
        }

        # Configuration
        self.max_recent_blunders_in_prompt = 3 # For Requirement 2
        self.blunder_threshold = 150 # Centipawn drop threshold for blunder detection
        self.max_retries_llm = 3     # Max retries for LLM generating invalid move
        self.stockfish_timeout = 1.0 # Default thinking time for Stockfish opponent moves
        self.stockfish_eval_depth = stockfish_depth # Depth for position evaluation
        self.stockfish_verification_depth = 8 # Shallow depth for hybrid verification (Requirement 5)

        # Create results directory if it doesn't exist
        self.base_results_dir = "benchmark_results"
        self.current_run_dir = None # Will be set in run_benchmark
        os.makedirs(self.base_results_dir, exist_ok=True)

        logger.info(f"Initialized ChessBenchmark. LLM: {llm_model}, Temp: {temperature}, SF Depth: {stockfish_depth}")
        if not self.chat:
            logger.warning("LLM client not initialized. LLM functionality will be unavailable.")
        if not self.stockfish_available:
             logger.warning("Stockfish engine not available. Engine functionality will be limited.")

# --- End of Part 1 ---