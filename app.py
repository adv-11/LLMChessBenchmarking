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

# Part 2: Move Generation and Context Helpers

class ChessBenchmark:
    # (Includes __init__ from Part 1)
    # ... (previous code from Part 1) ...

    def _get_visual_board(self, board: chess.Board) -> str:
        """
        Generate a text-based visual representation of the board with rank/file labels.

        Args:
            board: The current chess.Board object.

        Returns:
            A string representing the board visually.
        """
        visual = str(board)
        ranks = "87654321"
        files = "  a b c d e f g h" # Add spacing for alignment
        lines = visual.split("\n")

        # Add rank labels to the right
        for i, line in enumerate(lines):
            lines[i] = line + " " + ranks[i]

        # Add file labels at the bottom
        lines.append(files)
        return "\n".join(lines)

    def detect_position_type(self, board: chess.Board) -> str:
        """
        Classify the position type (opening, middlegame, endgame) based on simple heuristics.
        (Requirement 5)

        Args:
            board: The current chess.Board object.

        Returns:
            A string indicating the position type ("opening", "middlegame", "endgame").
        """
        piece_count = len(board.piece_map())
        # Endgame definition: King + <= 5 other pieces total (adjust as needed)
        if piece_count <= 6:
            return "endgame"
        # Opening definition: Within the first 10-12 full moves (adjust as needed)
        elif board.fullmove_number <= 10:
             # Check if major pieces are still mostly on back ranks (more robust check possible)
            back_rank_pieces = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type != chess.PAWN:
                    rank = chess.square_rank(square)
                    if (piece.color == chess.WHITE and rank == 0) or \
                       (piece.color == chess.BLACK and rank == 7):
                        back_rank_pieces += 1
            # If many major pieces still haven't moved, likely still opening phase
            if back_rank_pieces > 8: # Heuristic threshold
                 return "opening"
            else:
                 return "middlegame" # Transitioning out of opening
        else:
            return "middlegame"

    def get_position_template(self, position_type: str) -> str:
        """
        Return a position-specific prompt template snippet based on the classified type.
        (Requirement 5)

        Args:
            position_type: The type of position ("opening", "middlegame", "endgame").

        Returns:
            A string containing guidance for the LLM prompt.
        """
        templates = {
            "opening": "Strategy: Focus on rapid development of minor pieces, controlling the center (e.g., e4, d4, e5, d5 squares), and ensuring king safety, often through castling. Avoid moving the same piece multiple times unless necessary.",
            "middlegame": "Strategy: Look for tactical opportunities (forks, pins, skewers, discovered attacks). Formulate strategic plans based on pawn structures (e.g., open files for rooks, outpost squares for knights, pawn breaks). Coordinate your pieces towards attacking the opponent's king or key weaknesses.",
            "endgame": "Strategy: King activity is crucial - centralize your king. Calculate precise move sequences, especially regarding pawn promotion. Understand key endgame principles like opposition, zugzwang, and creating passed pawns. Material advantage becomes more significant."
        }
        return templates.get(position_type, "Strategy: Analyze the position carefully to identify threats, opportunities, and the best strategic plan.") # Default fallback

    def llm_make_move(self, board: chess.Board, conversation_history: str, current_eval: Optional[int], retry_count=0) -> Optional[str]:
        """
        Generate the next move for White using LLM with enhanced prompting (CoT, context)
        and optional verification. (Requirements 1, 2, 5)

        Args:
            board: Current chess board state.
            conversation_history: String containing previous moves in the game.
            current_eval: The current Stockfish evaluation (in centipawns), if available.
            retry_count: Internal counter for retry attempts.

        Returns:
            UCI move string (e.g., "e2e4") or None if unable to generate a valid move.
        """
        if not self.chat:
            logger.error("LLM client not initialized. Cannot generate LLM move.")
            return self.get_stockfish_move(board) # Fallback if LLM unavailable

        if retry_count > self.max_retries_llm:
            logger.warning(f"LLM failed to provide a valid move after {self.max_retries_llm} retries. Falling back to Stockfish.")
            self.stats["illegal_moves_llm"] += 1
            # Use default Stockfish settings for fallback
            return self.get_stockfish_move(board, thinking_time=self.stockfish_timeout, depth=self.stockfish_eval_depth)

        fen = board.fen()
        visual_board = self._get_visual_board(board)
        try:
            legal_moves = [move.uci() for move in board.legal_moves]
            if not legal_moves:
                logger.warning("No legal moves available for LLM - game should be over.")
                return None
        except Exception as e:
            logger.error(f"Error getting legal moves: {e}. Board FEN: {fen}")
            return None # Cannot proceed without legal moves

        # --- Enhanced Prompt Construction ---
        position_type = self.detect_position_type(board)
        position_guidance = self.get_position_template(position_type)

        # Context: Current evaluation
        eval_text = f"Current Stockfish evaluation: {current_eval / 100.0:.2f} pawns (positive is good for White)." if current_eval is not None else "Current Stockfish evaluation: Not available."

        # Context: Recent Blunders (Requirement 2)
        blunder_context = ""
        if self.stats["recent_blunders"]:
            blunder_context += "\nRecent significant mistakes (blunders) in previous moves to learn from:"
            # Use max_recent_blunders_in_prompt to limit context
            for blunder in self.stats["recent_blunders"][-self.max_recent_blunders_in_prompt:]:
                # Note: 'reason' is hard to generate automatically, focusing on move and impact.
                blunder_context += f"\n- Move {blunder['move']} led to an evaluation drop of approximately {-blunder['eval_drop'] / 100.0:.2f} pawns."

        # Context: Conversation History (Limit length to avoid excessive tokens)
        max_history_lines = 20 # Keep last 10 full moves (20 lines)
        history_lines = conversation_history.strip().split('\n')
        truncated_history = "\n".join(history_lines[-max_history_lines:]) if len(history_lines) > max_history_lines else conversation_history

        # Chain-of-Thought Prompt (Requirement 1)
        prompt = (
            f"You are a focused chess AI playing as White. Your goal is to find the strongest move.\n"
            f"Current board state (White=uppercase, Black=lowercase):\n{visual_board}\n"
            f"FEN notation: {fen}\n"
            f"Game phase: {position_type.capitalize()}\n"
            f"{eval_text}\n"
            f"Recent Move History (last {max_history_lines // 2} moves):\n{truncated_history}\n"
            f"Legal moves (UCI format): {', '.join(legal_moves)}\n"
            f"{blunder_context}\n\n"
            f"Strategic Guidance for {position_type}: {position_guidance}\n\n"
            "Task:\n"
            "1. **Analyze the position:** Briefly consider material balance, piece activity, king safety, pawn structure, and immediate tactical threats/opportunities.\n"
            "2. **Identify Candidate Moves:** List 2-3 promising candidate moves from the legal moves list.\n"
            "3. **Think Step-by-Step:** For each candidate, briefly outline the main idea and potential consequence (e.g., 'Nf3 develops a piece and controls e5', 'e4e5 attacks the Nf6 knight').\n"
            "4. **Select the Best Move:** Based on your analysis, choose the single best move.\n"
            "5. **Output:** Respond ONLY with the chosen move in UCI notation (e.g., 'e2e4'). Do not include your analysis or any other text in the final output."
        )

        messages = [
            SystemMessage(content=(
                "You are a chess engine assistant. Analyze the given chess position thoroughly, "
                "following the step-by-step reasoning process requested. "
                "Your final output must be ONLY the single best move in UCI format (e.g., 'a1b1', 'e7e8q'). "
                "No explanation, commentary, or analysis should accompany the final UCI move output."
            )),
            HumanMessage(content=prompt)
        ]

        start_time = time.time()
        try:
            response = self.chat.invoke(messages)
            llm_output = response.content.strip()

            # --- Strict Move Extraction and Validation ---
            # Attempt to find a legal move directly within the output
            extracted_move = None
            possible_moves = llm_output.split() # Split in case there's extra text
            potential_move = possible_moves[-1].lower() # Often the last word is the move

            if potential_move in legal_moves:
                 extracted_move = potential_move
            else:
                 # Fallback: Search the entire output for any legal move string
                 for legal_uci in legal_moves:
                    if legal_uci in llm_output.lower():
                        # Be cautious: ensure it's likely the intended move, not just a substring
                        # Check if it's a standalone word or at the end
                        if f" {legal_uci} " in f" {llm_output.lower()} " or llm_output.lower().endswith(legal_uci):
                             extracted_move = legal_uci
                             logger.info(f"Extracted move '{extracted_move}' from potentially noisy output: '{llm_output}'")
                             break

            move_time = time.time() - start_time
            self.stats["total_llm_move_time"] += move_time
            # Increment move count only when a valid move is confirmed and pushed in play_game

            if extracted_move and extracted_move in legal_moves:
                logger.debug(f"LLM proposed move: {extracted_move} (Time: {move_time:.2f}s)")

                # Optional: Shallow verification can be added here if desired,
                # e.g., evaluate position after extracted_move using evaluate_position(..., depth=self.stockfish_verification_depth)
                # and log if it seems tactically poor compared to initial eval.
                # For now, we return the validated move directly.

                return extracted_move
            else:
                logger.warning(f"LLM output '{llm_output}' did not contain a valid UCI move from the legal list: {legal_moves}. Retrying...")
                return self.llm_make_move(board, conversation_history, current_eval, retry_count + 1)

        except Exception as e:
            logger.error(f"Error during LLM call or processing: {e}", exc_info=True)
            return self.llm_make_move(board, conversation_history, current_eval, retry_count + 1)

    def get_stockfish_move(self, board: chess.Board, thinking_time: Optional[float] = None, depth: Optional[int] = None) -> Optional[str]:
        """
        Get the best move from Stockfish engine.

        Args:
            board: Current chess board state.
            thinking_time: Time limit for Stockfish (seconds). Uses self.stockfish_timeout if None.
            depth: Depth limit for Stockfish. Uses self.stockfish_eval_depth if None.

        Returns:
            UCI move string or None if no legal moves or engine fails.
        """
        if not self.stockfish_available:
            logger.error("Stockfish engine not available. Cannot get Stockfish move.")
            # Fallback: random move if possible
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves).uci() if legal_moves else None

        try:
            if not list(board.legal_moves):
                logger.warning("No legal moves available for Stockfish.")
                return None

            # Use provided limits or defaults from self
            time_limit = thinking_time if thinking_time is not None else self.stockfish_timeout
            depth_limit = depth if depth is not None else self.stockfish_eval_depth # Use eval_depth as default for opponent move

            limit = chess.engine.Limit(time=time_limit, depth=depth_limit)

            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
                # Configure skill level if needed (e.g., for weaker opponent simulation)
                # stockfish.configure({"Skill Level": 10}) # Example: Elo ~1700
                result = stockfish.play(board, limit)
                if result.move:
                    logger.debug(f"Stockfish move: {result.move.uci()}")
                    return result.move.uci()
                else:
                    logger.warning("Stockfish analysis returned no move.")
                    # Fallback to random if engine gives up unexpectedly
                    legal_moves = list(board.legal_moves)
                    return random.choice(legal_moves).uci() if legal_moves else None

        except chess.engine.EngineTerminatedError:
            logger.error("Stockfish engine terminated unexpectedly.")
            self.stockfish_available = False # Mark as unavailable
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves).uci() if legal_moves else None
        except Exception as e:
            logger.error(f"Error interacting with Stockfish: {e}", exc_info=True)
            legal_moves = list(board.legal_moves)
            return random.choice(legal_moves).uci() if legal_moves else None

# --- End of Part 2 ---