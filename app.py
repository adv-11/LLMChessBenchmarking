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
import argparse

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

    def __init__(self, llm_model="gpt-4", temperature=0.0, stockfish_depth=15,
                 elo_k_factor=32, stockfish_opponent_elo=2800):
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

# class ChessBenchmark:
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


# Part 3: Analysis, Evaluation, Saving, and Utility Methods

# class ChessBenchmark:
    # ... (previous code from Part 1 and 2) ...

    def evaluate_position(self, board: chess.Board, depth: Optional[int] = None) -> Optional[int]:
        """
        Get Stockfish's evaluation of the current position in centipawns.

        Args:
            board: The current chess.Board object.
            depth: The search depth for evaluation. Uses self.stockfish_eval_depth if None.

        Returns:
            Evaluation score in centipawns (positive is good for White),
            or None if evaluation fails.
        """
        if not self.stockfish_available:
            logger.warning("Stockfish not available, cannot evaluate position.")
            return None # Return None to indicate failure

        eval_depth = depth if depth is not None else self.stockfish_eval_depth

        try:
            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
                # Use a shorter timeout for evaluation compared to playing moves
                # Limit by depth primarily
                info = stockfish.analyse(board, chess.engine.Limit(depth=eval_depth, time=max(0.5, self.stockfish_timeout / 2)))

                # Score can be PovScore or Cp or Mate object
                score = info.get("score")
                if score is None:
                    logger.warning(f"Stockfish analysis did not return a score for FEN: {board.fen()}")
                    return None # Indicate failure

                # Get score relative to the side whose turn it is
                relative_score = score.white() if board.turn == chess.WHITE else score.black()

                # Convert to centipawns from White's perspective
                # Handle Mate scores appropriately
                final_score = relative_score.score(mate_score=10000) # Assign high value for mate

                if final_score is None:
                     # If score is Mate('-0'), score() might return None. Checkmate is maximally bad.
                     if score.is_mate() and score.mate() < 0:
                          final_score = -10000
                     # If score is Mate('+0'), score() might return None. Checkmate is maximally good.
                     elif score.is_mate() and score.mate() > 0:
                          final_score = 10000
                     else:
                          logger.warning(f"Could not convert score '{score}' to centipawns.")
                          return 0 # Default to 0 if conversion fails unexpectedly

                # Ensure score is relative to White
                if board.turn == chess.BLACK:
                     final_score = -final_score # Invert score if it's Black's turn

                return final_score

        except chess.engine.EngineTerminatedError:
             logger.error("Stockfish engine terminated unexpectedly during evaluation.")
             self.stockfish_available = False
             return None
        except Exception as e:
            logger.error(f"Error during position evaluation for FEN {board.fen()}: {e}", exc_info=True)
            return None # Indicate failure

    def detect_blunder(self, board_after_move: chess.Board, prev_eval: Optional[int], current_eval: Optional[int]) -> bool:
        """
        Detect if the last move played was a blunder based on evaluation change.
        Handles perspective correctly (White moves, eval drops; Black moves, eval increases relative to White).

        Args:
            board_after_move: The board state *after* the move was made.
            prev_eval: Evaluation (centipawns) *before* the move.
            current_eval: Evaluation (centipawns) *after* the move.

        Returns:
            True if the move qualifies as a blunder, False otherwise.
        """
        if prev_eval is None or current_eval is None:
            return False # Cannot detect blunder without evaluations

        # Ignore detection if near mate scores, as fluctuations are less meaningful
        if abs(prev_eval) > 9000 or abs(current_eval) > 9000:
            return False

        eval_delta = current_eval - prev_eval

        # Check whose turn it is *now*. If it's Black's turn, White just moved.
        if board_after_move.turn == chess.BLACK:
            # White's move is a blunder if evaluation decreased significantly (delta is negative)
            return eval_delta <= -self.blunder_threshold
        # If it's White's turn, Black just moved.
        else:
            # Black's move is a blunder if evaluation increased significantly for White (delta is positive)
            # Note: This function is primarily used to detect LLM (White's) blunders in the current setup.
            # Adjust if needed for analysing Black's blunders.
             # return eval_delta >= self.blunder_threshold # If checking Black's blunder
             return False # In current setup, only checking White's blunders

    def update_elo(self, llm_current_elo: int, opponent_elo: int, game_result: float) -> int:
        """
        Calculate the new Elo rating based on the game result.

        Args:
            llm_current_elo: The LLM's Elo rating before the game.
            opponent_elo: The opponent's Elo rating (e.g., Stockfish estimated Elo).
            game_result: Game result from LLM's perspective (1.0=win, 0.5=draw, 0.0=loss).

        Returns:
            The new Elo rating after the game.
        """
        expected_score = 1 / (1 + 10 ** ((opponent_elo - llm_current_elo) / 400))
        new_rating = llm_current_elo + self.elo_k_factor * (game_result - expected_score)
        return int(round(new_rating))

    def save_game_to_pgn(self, game_data: Dict[str, Any], game_index: int) -> Optional[str]:
        """
        Save a completed game to PGN format in the current benchmark run directory.

        Args:
            game_data: Dictionary containing game details (result, moves, starting FEN, etc.).
            game_index: The index number of the game within the current benchmark run.

        Returns:
            Path to the saved PGN file, or None on failure.
        """
        if not self.current_run_dir:
             logger.error("Current run directory not set. Cannot save PGN.")
             return None

        try:
            game = chess.pgn.Game()

            # --- Standard PGN Headers ---
            game.headers["Event"] = f"LLM Chess Benchmark ({self.llm_model})"
            game.headers["Site"] = f"Local Machine (Stockfish {self.stockfish_depth})"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(game_index)
            game.headers["White"] = f"LLM ({self.llm_model})"
            game.headers["Black"] = f"Stockfish (Depth {self.stockfish_depth})" # Or skill level if used
            game.headers["Result"] = game_data["result"]
            game.headers["WhiteElo"] = str(game_data.get("elo_before", self.stats["elo_rating"])) # Use Elo before this game
            game.headers["BlackElo"] = str(game_data.get("opponent_elo", "N/A")) # Assumed Stockfish Elo
            # Optional: Add Elo change if available
            if "elo_after" in game_data and "elo_before" in game_data:
                 white_elo_diff = game_data['elo_after'] - game_data['elo_before']
                 game.headers["WhiteRatingDiff"] = f"{white_elo_diff:+}"
                 # Assume Black Elo is stable or calculate if tracking opponent Elo
                 game.headers["BlackRatingDiff"] = "0"


            # --- Setup Specific Position (if not standard) ---
            starting_fen = game_data.get("starting_fen", chess.STARTING_FEN)
            if starting_fen != chess.STARTING_FEN:
                game.headers["SetUp"] = "1"
                game.headers["FEN"] = starting_fen

            # --- Custom Headers (Optional) ---
            game.headers["LLM_Blunders"] = str(game_data.get("llm_blunders", 0))
            game.headers["FinalEval_CP"] = str(game_data.get("final_eval")) # Centipawns
            game.headers["Opening_Detected"] = game_data.get("opening", "Unknown")

            # --- Reconstruct Game Moves ---
            # Use a temporary board starting from the correct FEN
            temp_board = chess.Board(starting_fen)
            node = game # Start at the root node

            for move_uci in game_data.get("moves", []):
                try:
                    move = chess.Move.from_uci(move_uci)
                    # Important: Ensure the move was actually legal in that position
                    if move in temp_board.legal_moves:
                        node = node.add_variation(move)
                        temp_board.push(move)
                    else:
                        # This indicates a potential issue during game play recording or an illegal fallback move was saved.
                        logger.warning(f"Skipping illegal move '{move_uci}' during PGN reconstruction for game {game_index}. FEN: {temp_board.fen()}")
                        # Optionally add a comment to the PGN node about the skipped move
                        # node.comment = f"Illegal move {move_uci} skipped here."
                        # Decide whether to stop reconstruction or continue
                        continue # Continue with next recorded move
                except ValueError:
                    logger.warning(f"Invalid UCI move string '{move_uci}' encountered during PGN reconstruction for game {game_index}.")
                    continue

            # Add final board state comment (optional)
            game.end().comment = f"Final board FEN: {temp_board.fen()}"

            # --- Save to File ---
            # Use the specific benchmark run directory
            filename = os.path.join(self.current_run_dir, f"game_{game_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn")
            with open(filename, "w", encoding="utf-8") as f:
                exporter = chess.pgn.FileExporter(f)
                game.accept(exporter)

            logger.debug(f"Game {game_index} saved to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error saving game {game_index} to PGN: {e}", exc_info=True)
            # Fallback: Save raw game data as JSON
            fallback_filename = os.path.join(self.current_run_dir, f"game_{game_index}_data_fallback.json")
            try:
                 with open(fallback_filename, "w") as f:
                     json.dump(game_data, f, indent=2)
                 return fallback_filename
            except Exception as dump_e:
                 logger.error(f"Error saving fallback JSON for game {game_index}: {dump_e}")
                 return None

    def generate_annotated_positions(self, position_set_name: str, num_positions: int = 10, depth: int = 20, output_file: Optional[str] = None) -> List[Dict]:
        """
        Generates training/analysis data by analyzing positions with Stockfish.
        (Requirement 4)

        Args:
            position_set_name: Name of the key from ALL_POSITION_SETS (e.g., "SIMPLE_ENDGAMES").
            num_positions: Maximum number of positions to annotate from the set.
            depth: Stockfish analysis depth for quality annotation.
            output_file: Optional path to save the annotations as a JSON file.

        Returns:
            A list of dictionaries, each containing annotated position data,
            or an empty list if Stockfish is unavailable or errors occur.
        """
        if not self.stockfish_available:
            logger.error("Stockfish not available. Cannot generate annotated positions.")
            return []

        if position_set_name not in ALL_POSITION_SETS:
             logger.error(f"Position set '{position_set_name}' not found in ALL_POSITION_SETS.")
             return []

        positions_to_annotate = list(ALL_POSITION_SETS[position_set_name].items())
        random.shuffle(positions_to_annotate) # Process a random subset if needed
        positions_to_annotate = positions_to_annotate[:num_positions]

        annotated_data = []
        logger.info(f"Starting annotation generation for {len(positions_to_annotate)} positions from '{position_set_name}' with depth {depth}.")

        try:
            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
                for name, fen in positions_to_annotate:
                    try:
                        board = chess.Board(fen)
                        logger.debug(f"Annotating position: {name} ({fen})")

                        # Get deep analysis
                        info = engine.analyse(board, chess.engine.Limit(depth=depth))

                        score_obj = info.get("score")
                        if score_obj:
                             eval_score = score_obj.white().score(mate_score=10000)
                             if board.turn == chess.BLACK: eval_score = -eval_score # Ensure relative to White
                        else:
                             eval_score = None

                        pv_moves = info.get("pv", [])
                        best_move = pv_moves[0].uci() if pv_moves else None

                        annotation = {
                            "name": name,
                            "fen": fen,
                            "best_move_uci": best_move,
                            "evaluation_cp": eval_score,
                            "principal_variation_uci": [move.uci() for move in pv_moves],
                            "depth": depth
                        }
                        annotated_data.append(annotation)

                    except Exception as pos_e:
                        logger.error(f"Error annotating position {name} ({fen}): {pos_e}", exc_info=True)
                        continue # Skip to next position

        except chess.engine.EngineTerminatedError:
             logger.error("Stockfish engine terminated unexpectedly during annotation.")
             self.stockfish_available = False
             return []
        except Exception as e:
            logger.error(f"General error during annotation generation: {e}", exc_info=True)
            return annotated_data # Return partial data if some succeeded


        # Save to file if requested
        if output_file:
            try:
                # Ensure directory exists if path includes folders
                output_dir = os.path.dirname(output_file)
                if output_dir:
                     os.makedirs(output_dir, exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(annotated_data, f, indent=2)
                logger.info(f"Saved {len(annotated_data)} annotated positions to {output_file}")
            except Exception as save_e:
                logger.error(f"Error saving annotated positions to {output_file}: {save_e}")

        return annotated_data

    def analyze_losing_positions(self, blunder_threshold: Optional[int] = None) -> int:
        """
        Analyzes completed games to find positions where the LLM blundered significantly.
        Populates self.stats["blunder_positions"]. (Requirement 5)

        Args:
            blunder_threshold: Optional override for the blunder centipawn threshold.

        Returns:
            The number of new blunder positions identified and added.
        """
        if not self.stats["game_history"]:
            logger.info("No game history available to analyze for blunders.")
            return 0

        threshold = blunder_threshold if blunder_threshold is not None else self.blunder_threshold
        new_blunders_found = 0
        processed_fens = {bp['fen'] for bp in self.stats["blunder_positions"]} # Avoid duplicates

        logger.info(f"Analyzing game history for blunders (threshold: {threshold}cp)...")

        for game_idx, game_data in enumerate(self.stats["game_history"]):
            moves = game_data.get("moves", [])
            eval_history = game_data.get("eval_history", [])
            starting_fen = game_data.get("starting_fen", chess.STARTING_FEN)

            # Need at least one move and two evaluations to detect a blunder
            if len(moves) < 1 or len(eval_history) < 2:
                continue

            try:
                board = chess.Board(starting_fen)
                # Iterate through LLM (White's) moves
                # White moves are at indices 0, 2, 4, ...
                # Eval history has initial eval at index 0, eval after move i is at index i+1
                for move_idx in range(0, len(moves), 2): # Step by 2 for White's moves
                    # Ensure evaluations exist for before and after this move
                    # Eval before move `move_idx` is `eval_history[move_idx]`
                    # Eval after move `move_idx` is `eval_history[move_idx + 1]`
                    if move_idx + 1 >= len(eval_history):
                         continue # Not enough evaluation data for this move

                    prev_eval = eval_history[move_idx]
                    current_eval = eval_history[move_idx + 1]

                    if prev_eval is None or current_eval is None:
                        board.push(chess.Move.from_uci(moves[move_idx]))
                        # Also push Black's response if it exists
                        if move_idx + 1 < len(moves):
                            board.push(chess.Move.from_uci(moves[move_idx + 1]))
                        continue # Skip if evals are missing

                    # Check if White's move was a blunder
                    eval_delta = current_eval - prev_eval
                    if eval_delta <= -threshold:
                        # Blunder detected! Reconstruct board *before* the blunder move
                        blunder_board = chess.Board(starting_fen)
                        for i in range(move_idx):
                            blunder_board.push(chess.Move.from_uci(moves[i]))

                        blunder_fen = blunder_board.fen()
                        blunder_move_uci = moves[move_idx]

                        # Avoid adding duplicate positions from different games/runs
                        if blunder_fen not in processed_fens:
                             blunder_info = {
                                 "game_index": game_idx + 1, # 1-based index
                                 "move_number": board.fullmove_number, # Move number when blunder occurred
                                 "fen": blunder_fen,
                                 "blunder_move_uci": blunder_move_uci,
                                 "eval_before_cp": prev_eval,
                                 "eval_after_cp": current_eval,
                                 "eval_drop_cp": eval_delta # This will be negative
                             }
                             self.stats["blunder_positions"].append(blunder_info)
                             processed_fens.add(blunder_fen)
                             new_blunders_found += 1
                             logger.debug(f"Blunder detected in game {game_idx+1}: Move {blunder_move_uci} from FEN {blunder_fen}, Eval drop {eval_delta}")

                    # Push the moves to advance the board state for the next iteration
                    board.push(chess.Move.from_uci(moves[move_idx]))
                    if move_idx + 1 < len(moves): # Check if Black's response exists
                        board.push(chess.Move.from_uci(moves[move_idx + 1]))

            except Exception as game_e:
                logger.error(f"Error analyzing game {game_idx+1} for blunders: {game_e}", exc_info=True)
                continue # Skip to next game

        logger.info(f"Blunder analysis complete. Found {new_blunders_found} new blunder positions.")
        # Optionally save self.stats["blunder_positions"] to a file here
        # blunder_file = os.path.join(self.current_run_dir or self.base_results_dir, "blunder_positions.json")
        # with open(blunder_file, "w") as f:
        #     json.dump(self.stats["blunder_positions"], f, indent=2)

        return new_blunders_found

# --- End of Part 3 ---

# Part 4: Game Play, Benchmark Orchestration, and Curriculum Learning

# class ChessBenchmark:
    # ... (previous code from Parts 1, 2, 3) ...

    # Add opponent Elo to init, default can be adjusted
    def __init__(self, llm_model="gpt-4", temperature=0.0, stockfish_depth=15,
                 elo_k_factor=32, stockfish_opponent_elo=2800): # Added opponent Elo
        # ... (rest of __init__ from Part 1, including setting self.stockfish_opponent_elo) ...
        self.stockfish_opponent_elo = stockfish_opponent_elo
        # ...

    def play_game(self, game_num: int, opening_name: Optional[str] = None, custom_fen: Optional[str] = None,
                  opponent_elo: Optional[int] = None, use_hybrid_verification: bool = False,
                  max_plies: int = 400) -> Dict[str, Any]:
        """
        Play a single game between LLM (White) and Stockfish (Black).

        Args:
            game_num: The index of this game in the benchmark run (1-based).
            opening_name: Name of the opening from OPENING_POSITIONS dict.
            custom_fen: Custom starting FEN. Overrides opening_name if provided.
            opponent_elo: Assumed Elo rating of the Stockfish opponent for this game.
            use_hybrid_verification: If True, pass flag to llm_make_move (currently informational).
            max_plies: Maximum number of half-moves before declaring a draw.

        Returns:
            Dictionary containing game results and statistics.
        """
        game_data = {
            "game_num": game_num,
            "result": "*", # Default to undecided
            "termination_reason": "Unknown",
            "moves": [],
            "eval_history": [], # Stores eval *after* each move, starting with initial pos
            "llm_blunders": 0,
            "final_eval": None,
            "opening": "Unknown",
            "starting_fen": chess.STARTING_FEN,
            "elo_before": self.stats["elo_rating"],
            "opponent_elo": opponent_elo if opponent_elo is not None else self.stockfish_opponent_elo,
            "elo_after": None,
            "total_plies": 0
        }

        # --- Board Setup ---
        board = chess.Board()
        if custom_fen:
            try:
                board.set_fen(custom_fen)
                game_data["starting_fen"] = custom_fen
                game_data["opening"] = "Custom FEN"
            except ValueError:
                logger.error(f"Invalid custom FEN provided: {custom_fen}. Starting standard game.")
                game_data["starting_fen"] = chess.STARTING_FEN
        elif opening_name and opening_name in OPENING_POSITIONS:
            try:
                board.set_fen(OPENING_POSITIONS[opening_name])
                game_data["starting_fen"] = OPENING_POSITIONS[opening_name]
                game_data["opening"] = opening_name
            except ValueError:
                 logger.error(f"Invalid FEN for opening {opening_name}: {OPENING_POSITIONS[opening_name]}. Starting standard game.")
                 game_data["starting_fen"] = chess.STARTING_FEN
        else:
            # Standard game
            game_data["starting_fen"] = chess.STARTING_FEN
            game_data["opening"] = "Standard"


        conversation_history = "" # Stores moves like "White: e2e4\nBlack: c7c5\n"
        detected_opening_name = None # For dynamic detection

        # --- Initial State ---
        current_eval = self.evaluate_position(board)
        game_data["eval_history"].append(current_eval)
        logger.info(f"Starting Game {game_num}. FEN: {board.fen()}. Initial Eval: {current_eval/100.0 if current_eval is not None else 'N/A'}")

        # --- Game Loop ---
        while not board.is_game_over(claim_draw=True):
            if game_data["total_plies"] >= max_plies:
                logger.warning(f"Game {game_num} reached max plies ({max_plies}). Declaring draw.")
                game_data["result"] = "1/2-1/2"
                game_data["termination_reason"] = f"Max plies ({max_plies}) reached"
                break

            prev_eval = current_eval # Eval before the move about to be made
            move_uci = None
            is_llm_turn = board.turn == chess.WHITE

            try:
                if is_llm_turn: # LLM's turn (White)
                    logger.debug(f"Game {game_num}, Ply {game_data['total_plies']+1} (White), Eval: {prev_eval/100.0 if prev_eval is not None else 'N/A'}")
                    move_uci = self.llm_make_move(
                        board=board,
                        conversation_history=conversation_history,
                        current_eval=prev_eval
                        # use_hybrid_verification=use_hybrid_verification # Pass flag if needed
                    )
                    if move_uci:
                        logger.info(f"Game {game_num}: White (LLM) plays {move_uci}")
                        conversation_history += f"White: {move_uci}\n"
                        self.stats["total_llm_moves"] += 1
                    else:
                        logger.error(f"Game {game_num}: LLM failed to provide a move. Ending game.")
                        game_data["result"] = "0-1" # LLM forfeits
                        game_data["termination_reason"] = "LLM failed to move"
                        break

                else: # Stockfish's turn (Black)
                     logger.debug(f"Game {game_num}, Ply {game_data['total_plies']+1} (Black), Eval: {prev_eval/100.0 if prev_eval is not None else 'N/A'}")
                     move_uci = self.get_stockfish_move(board) # Use default time/depth
                     if move_uci:
                          logger.info(f"Game {game_num}: Black (Stockfish) plays {move_uci}")
                          conversation_history += f"Black: {move_uci}\n"
                     else:
                          logger.error(f"Game {game_num}: Stockfish failed to provide a move. Ending game.")
                          # If stockfish fails, maybe draw? Or LLM win? Let's call it a draw.
                          game_data["result"] = "1/2-1/2"
                          game_data["termination_reason"] = "Stockfish failed to move"
                          break

                # --- Validate and Make Move ---
                try:
                     chess_move = chess.Move.from_uci(move_uci)
                     if chess_move in board.legal_moves:
                          board.push(chess_move)
                          game_data["moves"].append(move_uci)
                          game_data["total_plies"] += 1
                          self.stats["total_game_plies"] += 1
                     else:
                          logger.error(f"Game {game_num}: {'LLM' if is_llm_turn else 'Stockfish'} generated illegal move: {move_uci}. FEN: {board.fen()}. Legal: {[m.uci() for m in board.legal_moves]}. Ending game.")
                          game_data["result"] = "0-1" if is_llm_turn else "1-0" # Player making illegal move loses
                          game_data["termination_reason"] = f"{'LLM' if is_llm_turn else 'Stockfish'} made illegal move {move_uci}"
                          if is_llm_turn: self.stats["illegal_moves_llm"] += 1
                          break
                except ValueError:
                     logger.error(f"Game {game_num}: Invalid UCI move format '{move_uci}'. Ending game.")
                     game_data["result"] = "0-1" if is_llm_turn else "1-0"
                     game_data["termination_reason"] = f"{'LLM' if is_llm_turn else 'Stockfish'} generated invalid format {move_uci}"
                     if is_llm_turn: self.stats["illegal_moves_llm"] += 1
                     break

                # --- Post-Move Analysis ---
                current_eval = self.evaluate_position(board)
                game_data["eval_history"].append(current_eval)

                # Blunder detection for LLM's move
                if is_llm_turn:
                    if self.detect_blunder(board, prev_eval, current_eval):
                        blunder_info = {
                            "game_num": game_num,
                            "ply": game_data["total_plies"],
                            "move": move_uci,
                            "eval_before": prev_eval,
                            "eval_after": current_eval,
                            "eval_drop": current_eval - prev_eval if prev_eval is not None and current_eval is not None else 0
                        }
                        game_data["llm_blunders"] += 1
                        self.stats["blunders_llm"] += 1
                        # Add to recent blunders queue (Requirement 2)
                        self.stats["recent_blunders"].append(blunder_info)
                        if len(self.stats["recent_blunders"]) > self.max_recent_blunders_in_prompt:
                            self.stats["recent_blunders"].pop(0) # Keep queue size limited
                        logger.info(f"Game {game_num}: Blunder detected! Move: {move_uci}, Eval drop: {blunder_info['eval_drop']/100.0:.2f}")

                # Opening detection (dynamic)
                if not detected_opening_name and game_data["total_plies"] <= 10: # Check first 5 full moves
                     op_name = self.detect_opening(board) # Reuse simple detection for now
                     if op_name:
                          detected_opening_name = op_name
                          game_data["opening"] = op_name # Update if detected dynamically
                          # Only count success if LLM played into a known opening from start
                          if game_data["opening"] != "Custom FEN" and game_data["opening"] != "Standard" and game_data["opening"] != "Unknown":
                                self.stats["opening_success"] += 1
                          logger.info(f"Game {game_num}: Dynamically detected opening: {detected_opening_name}")


            except Exception as loop_e:
                 logger.error(f"Unexpected error in game loop for game {game_num}: {loop_e}", exc_info=True)
                 game_data["result"] = "*" # Mark as error/unknown
                 game_data["termination_reason"] = f"Runtime error: {loop_e}"
                 break

        # --- Game End ---
        if game_data["result"] == "*": # If not ended by loop break condition
             try:
                  # Check standard termination reasons
                  outcome = board.outcome(claim_draw=True)
                  if outcome:
                       game_data["result"] = outcome.result()
                       if outcome.termination == chess.Termination.CHECKMATE:
                            winner = "White" if outcome.winner == chess.WHITE else "Black"
                            game_data["termination_reason"] = f"Checkmate ({winner} wins)"
                       elif outcome.termination == chess.Termination.STALEMATE:
                            game_data["termination_reason"] = "Stalemate"
                       elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                            game_data["termination_reason"] = "Insufficient Material"
                       elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                            game_data["termination_reason"] = "75 Move Rule"
                       elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                            game_data["termination_reason"] = "Fivefold Repetition"
                       elif outcome.termination == chess.Termination.VARIANT_WIN: # Should not happen
                            game_data["termination_reason"] = "Variant Win"
                       elif outcome.termination == chess.Termination.VARIANT_LOSS: # Should not happen
                             game_data["termination_reason"] = "Variant Loss"
                       elif outcome.termination == chess.Termination.VARIANT_DRAW: # Should not happen
                             game_data["termination_reason"] = "Variant Draw"
                  else:
                       # Should be game over, but no standard outcome? Maybe draw?
                       game_data["result"] = "1/2-1/2"
                       game_data["termination_reason"] = "Game ended without standard outcome"
                       logger.warning(f"Game {game_num} ended but outcome object was None. FEN: {board.fen()}")

             except Exception as outcome_e:
                  logger.error(f"Error determining game outcome for game {game_num}: {outcome_e}")
                  game_data["result"] = "*" # Unknown
                  game_data["termination_reason"] = f"Error getting outcome: {outcome_e}"

        game_data["final_eval"] = self.evaluate_position(board) # Get final evaluation
        logger.info(f"Game {game_num} finished. Result: {game_data['result']}, Reason: {game_data['termination_reason']}, Final Eval: {game_data['final_eval']/100.0 if game_data['final_eval'] is not None else 'N/A'}")

        # --- Update Stats and Elo ---
        self.stats["games_played"] += 1
        elo_result = 0.0 # Default to loss for LLM
        if game_data["result"] == "1-0":
            self.stats["llm_wins"] += 1
            elo_result = 1.0
        elif game_data["result"] == "0-1":
            self.stats["stockfish_wins"] += 1
            elo_result = 0.0
        elif game_data["result"] == "1/2-1/2":
            self.stats["draws"] += 1
            elo_result = 0.5

        # Calculate and update Elo rating
        current_elo = self.stats["elo_rating"]
        new_elo = self.update_elo(current_elo, game_data["opponent_elo"], elo_result)
        self.stats["elo_rating"] = new_elo
        game_data["elo_after"] = new_elo
        self.stats["elo_history"].append((game_num, new_elo)) # Track Elo progression
        logger.info(f"Game {game_num}: Elo updated from {current_elo} to {new_elo} (Result: {elo_result}, Opponent Elo: {game_data['opponent_elo']})")

        # Save game data to history
        self.stats["game_history"].append(game_data)

        # Save game to PGN
        pgn_file = self.save_game_to_pgn(game_data, game_num)
        if pgn_file:
             logger.info(f"Game {game_num} PGN saved to {pgn_file}")
        else:
             logger.error(f"Failed to save PGN for game {game_num}")

        return game_data


    def run_benchmark(self, num_games: int = 5, openings: Optional[List[str]] = None,
                      custom_fens: Optional[List[str]] = None, opponent_elo: Optional[int] = None,
                      use_hybrid_verification: bool = False) -> Dict[str, Any]:
        """
        Run a chess benchmark playing multiple games.

        Args:
            num_games: Number of games to play.
            openings: List of opening names (from OPENING_POSITIONS) to cycle through.
                      If None and custom_fens is None, uses random standard openings.
            custom_fens: List of specific FENs to start games from. Overrides openings.
            opponent_elo: Override the default opponent Elo for this run.
            use_hybrid_verification: Flag to enable hybrid checks in llm_make_move.

        Returns:
            Dictionary containing aggregated benchmark results.
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.base_results_dir, f"benchmark_{self.llm_model}_{timestamp}")
        os.makedirs(self.current_run_dir, exist_ok=True)
        logger.info(f"--- Starting Benchmark Run ---")
        logger.info(f"Model: {self.llm_model}, Temperature: {self.temperature}")
        logger.info(f"Number of Games: {num_games}, Stockfish Depth: {self.stockfish_depth}")
        logger.info(f"Results directory: {self.current_run_dir}")

        # Reset stats relevant to a single run (keep overall history if needed across runs)
        # Or, re-initialize stats completely if each run is independent
        initial_elo = self.stats["elo_rating"] # Capture Elo at the start of the run
        run_stats = { # Temp storage for this run's aggregation
             "llm_wins": 0, "stockfish_wins": 0, "draws": 0,
             "total_blunders": 0, "total_final_eval": 0, "games_completed": 0
        }


        opponent_rating = opponent_elo if opponent_elo is not None else self.stockfish_opponent_elo

        # Determine game setups
        game_setups = [] # List of tuples (type, value) e.g., ('opening', 'Sicilian Defense') or ('fen', '...')
        if custom_fens:
             logger.info(f"Using {len(custom_fens)} custom FEN positions.")
             if len(custom_fens) < num_games:
                  logger.warning(f"Number of custom FENs ({len(custom_fens)}) is less than num_games ({num_games}). FENs will be reused.")
             for i in range(num_games):
                  game_setups.append(('fen', custom_fens[i % len(custom_fens)]))
        elif openings:
             logger.info(f"Using specified openings list: {openings}")
             if len(openings) < num_games:
                  logger.warning(f"Number of specified openings ({len(openings)}) is less than num_games ({num_games}). Openings will be reused.")
             for i in range(num_games):
                  game_setups.append(('opening', openings[i % len(openings)]))
        else:
             logger.info("Using random standard openings.")
             standard_openings = list(OPENING_POSITIONS.keys())
             if not standard_openings:
                  logger.warning("No standard openings defined. Using default starting position.")
                  standard_openings = ["Standard"] # Fallback
             selected_openings = random.choices(standard_openings, k=num_games)
             for opening_name in selected_openings:
                 game_setups.append(('opening', opening_name))

        # --- Play Games ---
        for game_idx in range(num_games):
            game_num = self.stats["games_played"] + 1 # Overall game index
            setup_type, setup_value = game_setups[game_idx]

            logger.info(f"--- Starting Game {game_num}/{self.stats['games_played'] + num_games} (Run Game {game_idx+1}/{num_games}) ---")
            if setup_type == 'opening':
                 logger.info(f"Opening: {setup_value}")
                 current_game_data = self.play_game(game_num=game_num, opening_name=setup_value, opponent_elo=opponent_rating, use_hybrid_verification=use_hybrid_verification)
            else: # setup_type == 'fen'
                 logger.info(f"Custom FEN: {setup_value}")
                 current_game_data = self.play_game(game_num=game_num, custom_fen=setup_value, opponent_elo=opponent_rating, use_hybrid_verification=use_hybrid_verification)

            # Aggregate results for this run
            if current_game_data and current_game_data["result"] != "*":
                 run_stats["games_completed"] += 1
                 if current_game_data["result"] == "1-0": run_stats["llm_wins"] += 1
                 elif current_game_data["result"] == "0-1": run_stats["stockfish_wins"] += 1
                 else: run_stats["draws"] += 1
                 run_stats["total_blunders"] += current_game_data.get("llm_blunders", 0)
                 if current_game_data.get("final_eval") is not None:
                      run_stats["total_final_eval"] += current_game_data.get("final_eval", 0)
            else:
                 logger.warning(f"Game {game_num} result was invalid or game failed. Skipping aggregation for this game.")


        # --- Finalize Run Results ---
        run_duration = time.time() - start_time
        final_elo = self.stats["elo_rating"]
        results = {
            "benchmark_start_time": timestamp,
            "benchmark_duration_seconds": run_duration,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "stockfish_depth": self.stockfish_depth,
            "opponent_elo_assumed": opponent_rating,
            "num_games_requested": num_games,
            "num_games_completed": run_stats["games_completed"],
            "llm_wins": run_stats["llm_wins"],
            "stockfish_wins": run_stats["stockfish_wins"],
            "draws": run_stats["draws"],
            "win_rate_llm_percent": (run_stats["llm_wins"] / max(1, run_stats["games_completed"])) * 100,
            "avg_blunders_per_game": run_stats["total_blunders"] / max(1, run_stats["games_completed"]),
            "avg_final_eval_cp": run_stats["total_final_eval"] / max(1, run_stats["games_completed"]),
            "initial_elo": initial_elo,
            "final_elo": final_elo,
            "elo_change": final_elo - initial_elo,
            "overall_illegal_llm_moves": self.stats["illegal_moves_llm"], # Tracks across all runs if instance reused
            "overall_blunders_llm": self.stats["blunders_llm"], # Tracks across all runs
            "results_directory": self.current_run_dir
        }

        # Generate final report and visualizations for this run
        logger.info("Generating final report and visualizations for the run...")
        report_data = self.generate_report() # Uses overall stats, might need adjustment if run-specific report needed
        results["summary_report"] = report_data

        visualization_files = self.visualize_results() # Uses overall history
        results["visualization_files"] = visualization_files

        # Save the aggregated results for this specific run
        benchmark_results_file = os.path.join(self.current_run_dir, "benchmark_summary_results.json")
        try:
            with open(benchmark_results_file, "w") as f:
                # Use a custom encoder for numpy types if necessary, but basic types should be fine here
                 json.dump(results, f, indent=2, default=lambda x: str(x)) # Basic fallback for non-serializable
            logger.info(f"Benchmark run summary saved to {benchmark_results_file}")
        except Exception as save_e:
            logger.error(f"Error saving benchmark summary results: {save_e}")

        logger.info(f"--- Benchmark Run Completed in {run_duration:.2f} seconds ---")
        logger.info(f"Final Elo: {final_elo} (Change: {results['elo_change']})")
        logger.info(f"Results: {run_stats['llm_wins']} W / {run_stats['stockfish_wins']} L / {run_stats['draws']} D")

        # Analyze blunders found during this run
        self.analyze_losing_positions()

        return results

    def run_curriculum_benchmark(self, stages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run a progressive benchmark over stages of increasing difficulty.
        (Requirement 3)

        Args:
            stages: A list of dictionaries, where each dict defines a stage.
                    Example stage: {"name": "Simple Endgames", "position_set": "SIMPLE_ENDGAMES", "games": 5, "opponent_elo": 1500}
                    If None, uses a default curriculum.

        Returns:
            A dictionary containing results aggregated per stage.
        """
        if stages is None:
            # Define a default curriculum
            stages = [
                {"name": "Simple Endgames", "position_set": "SIMPLE_ENDGAMES", "games": 3, "opponent_elo": 1500, "stockfish_depth": 10},
                {"name": "Tactical Middlegames", "position_set": "TACTICAL_MIDDLEGAMES", "games": 5, "opponent_elo": 2000, "stockfish_depth": 12},
                {"name": "Complex Openings/Middlegames", "position_set": "COMPLEX_OPENINGS_MIDDLEGAMES", "games": 7, "opponent_elo": self.stockfish_opponent_elo, "stockfish_depth": self.stockfish_depth},
                {"name": "Standard Openings", "position_set": "STANDARD_OPENINGS", "games": 10, "opponent_elo": self.stockfish_opponent_elo, "stockfish_depth": self.stockfish_depth},
            ]

        logger.info("--- Starting Curriculum Benchmark ---")
        all_stage_results = {}
        original_stockfish_depth = self.stockfish_depth # Backup original depth
        original_opponent_elo = self.stockfish_opponent_elo # Backup original Elo

        for i, stage in enumerate(stages):
            stage_name = stage.get("name", f"Stage {i+1}")
            position_set_key = stage.get("position_set")
            num_games = stage.get("games", 5)
            stage_opponent_elo = stage.get("opponent_elo", original_opponent_elo)
            stage_stockfish_depth = stage.get("stockfish_depth", original_stockfish_depth)

            logger.info(f"--- Starting Curriculum Stage: {stage_name} ---")
            logger.info(f"Games: {num_games}, Position Set: {position_set_key}, Opponent Elo: {stage_opponent_elo}, SF Depth: {stage_stockfish_depth}")

            if not position_set_key or position_set_key not in ALL_POSITION_SETS:
                logger.error(f"Invalid or missing 'position_set' key: {position_set_key} in stage {stage_name}. Skipping stage.")
                continue

            position_dict = ALL_POSITION_SETS[position_set_key]
            fens_for_stage = [fen for name, fen in position_dict.items()]
            if not fens_for_stage:
                 logger.warning(f"Position set '{position_set_key}' is empty. Skipping stage {stage_name}.")
                 continue

            # Temporarily set stage parameters
            self.stockfish_depth = stage_stockfish_depth
            # Note: self.stockfish_opponent_elo is not directly used by get_stockfish_move,
            # but passed to play_game and then update_elo. We pass stage_opponent_elo directly.

            try:
                # Run a benchmark specifically for this stage using custom FENs
                stage_results = self.run_benchmark(
                    num_games=num_games,
                    custom_fens=fens_for_stage,
                    opponent_elo=stage_opponent_elo
                    # use_hybrid_verification can also be stage-specific if needed
                )
                all_stage_results[stage_name] = stage_results
                logger.info(f"--- Completed Curriculum Stage: {stage_name} ---")
                logger.info(f"Stage Win Rate: {stage_results.get('win_rate_llm_percent', 0):.2f}%, Elo after stage: {stage_results.get('final_elo')}")

                # Optional: Add logic here to check performance and decide whether to proceed
                # e.g., if stage_results['win_rate_llm_percent'] < 20: break

            except Exception as stage_e:
                logger.error(f"Error running curriculum stage '{stage_name}': {stage_e}", exc_info=True)
                all_stage_results[stage_name] = {"error": str(stage_e)}


            # Restore original parameters for next stage if they were changed on self
            self.stockfish_depth = original_stockfish_depth

        logger.info("--- Curriculum Benchmark Completed ---")

        # Optionally save the combined stage results
        curriculum_results_file = os.path.join(self.base_results_dir, f"curriculum_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(curriculum_results_file, "w") as f:
                json.dump(all_stage_results, f, indent=2, default=lambda x: str(x))
            logger.info(f"Curriculum benchmark results saved to {curriculum_results_file}")
        except Exception as save_e:
            logger.error(f"Error saving curriculum results: {save_e}")


        return all_stage_results

# --- End of Part 4 ---


# Part 5: Reporting, Visualization, and Main Execution



# class ChessBenchmark:
    # ... (previous code from Parts 1, 2, 3, 4) ...

    def generate_report(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive benchmarking report based on accumulated statistics.

        Args:
            output_dir: Directory to save the report JSON file. Uses current_run_dir or base_results_dir if None.

        Returns:
            Dictionary containing the report data.
        """
        report = {
            "report_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "stockfish_analysis_depth": self.stockfish_depth,
            "stockfish_opponent_elo_assumed": self.stockfish_opponent_elo,
            "elo_k_factor": self.elo_k_factor,
            "total_games_played_instance": self.stats["games_played"],
        }

        if self.stats["games_played"] == 0:
            logger.warning("No games played in this benchmark instance. Report will be minimal.")
            report["status"] = "No games played."
            return report

        # --- Calculate Statistics ---
        total_games = self.stats["games_played"]
        llm_wins = self.stats["llm_wins"]
        stockfish_wins = self.stats["stockfish_wins"]
        draws = self.stats["draws"]
        total_llm_moves = self.stats["total_llm_moves"]
        illegal_llm_moves = self.stats["illegal_moves_llm"]
        blunders_llm = self.stats["blunders_llm"]
        total_plies = self.stats["total_game_plies"]
        total_move_time = self.stats["total_llm_move_time"]

        # Avoid division by zero
        report["llm_win_rate_percent"] = (llm_wins / total_games) * 100 if total_games else 0
        report["stockfish_win_rate_percent"] = (stockfish_wins / total_games) * 100 if total_games else 0
        report["draw_rate_percent"] = (draws / total_games) * 100 if total_games else 0

        report["avg_game_length_moves"] = (total_plies / 2 / total_games) if total_games else 0

        report["avg_llm_move_time_seconds"] = (total_move_time / total_llm_moves) if total_llm_moves else 0

        # Rate per LLM move made
        report["llm_blunder_rate_percent"] = (blunders_llm / total_llm_moves) * 100 if total_llm_moves else 0
        # Rate per LLM move attempt (including illegal ones)
        total_attempts = total_llm_moves + illegal_llm_moves
        report["llm_illegal_move_rate_percent"] = (illegal_llm_moves / total_attempts) * 100 if total_attempts else 0

        report["opening_success_rate_percent"] = (self.stats["opening_success"] / total_games) * 100 if total_games else 0

        # Elo Rating
        report["initial_elo"] = self.stats["elo_history"][0][1] if self.stats["elo_history"] else self.stats["elo_rating"]
        report["final_elo"] = self.stats["elo_rating"]
        report["elo_change"] = report["final_elo"] - report["initial_elo"]

        # Average final evaluation (provides a sense of typical end-game advantage/disadvantage)
        final_evals = [g["final_eval"] for g in self.stats["game_history"] if g.get("final_eval") is not None]
        report["avg_final_eval_cp"] = np.mean(final_evals) if final_evals else None

        # --- Save Report ---
        save_dir = output_dir if output_dir else self.current_run_dir if self.current_run_dir else self.base_results_dir
        report_file = os.path.join(save_dir, f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=lambda x: str(x)) # Handle non-serializable types
            logger.info(f"Benchmark report saved to {report_file}")
            report["report_file_path"] = report_file
        except Exception as e:
            logger.error(f"Failed to save benchmark report: {e}")
            report["report_file_path"] = None

        return report

    def visualize_results(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Create visualizations of benchmark results based on accumulated statistics.

        Args:
            output_dir: Directory to save the plot files. Uses current_run_dir or base_results_dir if None.

        Returns:
            List of paths to the generated visualization PNG files.
        """
        if self.stats["games_played"] == 0:
            logger.warning("No games played, cannot generate visualizations.")
            return ["No games played yet"]

        save_dir = output_dir if output_dir else self.current_run_dir if self.current_run_dir else self.base_results_dir
        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists

        visualization_files = []
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        try:
            # 1. Win/Loss/Draw Pie Chart
            plt.figure(figsize=(8, 8))
            labels = ['LLM Wins', 'Stockfish Wins', 'Draws']
            sizes = [self.stats["llm_wins"], self.stats["stockfish_wins"], self.stats["draws"]]
            # Only plot slices > 0 to avoid clutter
            labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
            sizes_filtered = [s for s in sizes if s > 0]
            if sizes_filtered:
                 colors = ['lightgreen', 'lightcoral', 'lightskyblue'][:len(sizes_filtered)]
                 plt.pie(sizes_filtered, labels=labels_filtered, colors=colors, autopct='%1.1f%%', startangle=140)
                 plt.axis('equal')
                 plt.title(f'Game Outcomes ({self.stats["games_played"]} Games)\n{self.llm_model} vs Stockfish (Elo {self.stockfish_opponent_elo})')

                 pie_chart_file = os.path.join(save_dir, f"plot_outcomes_{timestamp_str}.png")
                 plt.savefig(pie_chart_file)
                 plt.close()
                 visualization_files.append(pie_chart_file)
                 logger.info(f"Outcome pie chart saved to {pie_chart_file}")
            else:
                 logger.warning("No game outcomes to plot in pie chart.")
                 plt.close()


            # 2. Elo Rating Over Time
            if len(self.stats["elo_history"]) > 1:
                 plt.figure(figsize=(12, 6))
                 game_indices = [item[0] for item in self.stats["elo_history"]]
                 elo_ratings = [item[1] for item in self.stats["elo_history"]]
                 plt.plot(game_indices, elo_ratings, marker='o', linestyle='-', label=f'{self.llm_model} Elo')
                 plt.xlabel('Game Number')
                 plt.ylabel('Elo Rating')
                 plt.title('LLM Elo Rating Progression Over Games')
                 plt.grid(True, alpha=0.5)
                 plt.legend()
                 # Add min/max Elo labels
                 min_elo = min(elo_ratings)
                 max_elo = max(elo_ratings)
                 plt.text(game_indices[-1], elo_ratings[-1], f" Final: {elo_ratings[-1]}", va='center')
                 plt.text(game_indices[0], elo_ratings[0], f"Start: {elo_ratings[0]} ", ha='right', va='center')


                 elo_chart_file = os.path.join(save_dir, f"plot_elo_history_{timestamp_str}.png")
                 plt.savefig(elo_chart_file)
                 plt.close()
                 visualization_files.append(elo_chart_file)
                 logger.info(f"Elo history plot saved to {elo_chart_file}")
            else:
                 logger.info("Not enough Elo history data points to plot.")


            # 3. LLM Blunders per Game
            blunders_per_game = [game.get("llm_blunders", 0) for game in self.stats["game_history"]]
            if blunders_per_game:
                 plt.figure(figsize=(12, 6))
                 game_nums = list(range(1, len(blunders_per_game) + 1))
                 plt.bar(game_nums, blunders_per_game, color='indianred', label='LLM Blunders')
                 avg_blunders = np.mean(blunders_per_game)
                 plt.axhline(y=avg_blunders, color='black', linestyle='--', alpha=0.7, label=f'Avg Blunders: {avg_blunders:.2f}')
                 plt.xlabel('Game Number')
                 plt.ylabel('Number of Blunders')
                 plt.title('LLM Blunders per Game')
                 plt.legend()
                 plt.grid(True, axis='y', alpha=0.3)
                 # Set x-axis ticks to be integers
                 if len(game_nums) < 20: # Show all ticks if few games
                      plt.xticks(game_nums)
                 else: # Otherwise let matplotlib decide ticks
                      pass

                 blunder_chart_file = os.path.join(save_dir, f"plot_blunders_{timestamp_str}.png")
                 plt.savefig(blunder_chart_file)
                 plt.close()
                 visualization_files.append(blunder_chart_file)
                 logger.info(f"Blunder analysis plot saved to {blunder_chart_file}")
            else:
                 logger.info("No blunder data available to plot.")


            # 4. Optional: Final Evaluation per Game
            final_evals_per_game = [(g.get("game_num", i+1), g.get("final_eval")) for i, g in enumerate(self.stats["game_history"])]
            valid_final_evals = [(num, val) for num, val in final_evals_per_game if val is not None]
            if valid_final_evals:
                 plt.figure(figsize=(12, 6))
                 game_indices_eval = [item[0] for item in valid_final_evals]
                 eval_values = [item[1] / 100.0 for item in valid_final_evals] # Convert to pawns
                 plt.plot(game_indices_eval, eval_values, marker='x', linestyle=':', color='purple', label='Final Eval (pawns)')
                 avg_final_eval = np.mean(eval_values)
                 plt.axhline(y=avg_final_eval, color='grey', linestyle='--', alpha=0.7, label=f'Avg Final Eval: {avg_final_eval:.2f} pawns')
                 plt.axhline(y=0, color='black', linestyle='-', alpha=0.5) # Zero line for reference
                 plt.xlabel('Game Number')
                 plt.ylabel('Final Evaluation (Pawns)')
                 plt.title('Final Game Evaluation (White\'s Perspective)')
                 plt.legend()
                 plt.grid(True, alpha=0.5)

                 final_eval_chart_file = os.path.join(save_dir, f"plot_final_eval_{timestamp_str}.png")
                 plt.savefig(final_eval_chart_file)
                 plt.close()
                 visualization_files.append(final_eval_chart_file)
                 logger.info(f"Final evaluation plot saved to {final_eval_chart_file}")


        except ImportError:
             logger.error("Matplotlib not installed or display unavailable. Cannot generate visualizations.")
             return ["Matplotlib error: Could not generate plots."]
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
            # Clean up any open plots
            plt.close('all')
            return [f"Error generating visualizations: {e}"]

        return visualization_files


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLM Chess Benchmark')

    # --- Core Benchmark Arguments ---
    parser.add_argument('--model', type=str, default=os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4"), help='LLM model to use (default: gpt-4 or OPENAI_DEFAULT_MODEL env var)')
    parser.add_argument('--games', type=int, default=5, help='Number of games to play in standard benchmark')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature parameter for LLM (default: 0.0)')
    parser.add_argument('--depth', type=int, default=15, help='Stockfish analysis depth (default: 15)')
    parser.add_argument('--opponent_elo', type=int, default=2800, help='Assumed Elo rating for Stockfish opponent (default: 2800)')
    parser.add_argument('--k_factor', type=int, default=32, help='K-factor for Elo calculation (default: 32)')
    parser.add_argument('--openings', type=str, nargs='+', help='List of specific standard opening names to use')

    # --- Mode Selection Arguments ---
    parser.add_argument('--mode', type=str, default='benchmark', choices=['benchmark', 'curriculum', 'annotate'],
                        help='Operation mode: "benchmark" (standard), "curriculum" (staged), "annotate" (utility)')

    # --- Curriculum Arguments ---
    # (Curriculum uses default stages unless complex stage definitions via JSON/file needed)

    # --- Annotation Arguments ---
    parser.add_argument('--annotate_set', type=str, default='SIMPLE_ENDGAMES', choices=list(ALL_POSITION_SETS.keys()),
                        help='Position set to use for annotation generation (default: SIMPLE_ENDGAMES)')
    parser.add_argument('--annotate_count', type=int, default=10, help='Number of positions to annotate (default: 10)')
    parser.add_argument('--annotate_depth', type=int, default=20, help='Stockfish depth for annotation (default: 20)')
    parser.add_argument('--annotate_output', type=str, default='annotated_positions.json', help='Output file for annotations (default: annotated_positions.json)')


    args = parser.parse_args()

    # --- Initialize Benchmark Class ---
    logger.info("Initializing ChessBenchmark...")
    benchmark = ChessBenchmark(
        llm_model=args.model,
        temperature=args.temperature,
        stockfish_depth=args.depth,
        stockfish_opponent_elo=args.opponent_elo,
        elo_k_factor=args.k_factor
    )

    # --- Execute Selected Mode ---
    try:
        if args.mode == 'benchmark':
             logger.info(f"Running Standard Benchmark ({args.games} games)...")
             results = benchmark.run_benchmark(
                 num_games=args.games,
                 openings=args.openings, # Pass specific openings if provided
                 opponent_elo=args.opponent_elo # Pass opponent Elo override
                 # use_hybrid_verification=False # Add arg if needed
             )
             if results:
                  print("\n--- Benchmark Summary ---")
                  print(f"LLM Model: {results.get('llm_model')}")
                  print(f"Games Completed: {results.get('num_games_completed')} / {results.get('num_games_requested')}")
                  print(f"Result: {results.get('llm_wins')} W / {results.get('stockfish_wins')} L / {results.get('draws')} D")
                  print(f"Win Rate: {results.get('win_rate_llm_percent', 0):.2f}%")
                  print(f"Elo Change: {results.get('elo_change', 0):+} (Start: {results.get('initial_elo')}, End: {results.get('final_elo')})")
                  print(f"Avg Blunders/Game: {results.get('avg_blunders_per_game', 0):.2f}")
                  print(f"Report/Plots saved in: {results.get('results_directory')}")
             else:
                  print("Benchmark run failed to produce results.")

        elif args.mode == 'curriculum':
             logger.info("Running Curriculum Benchmark...")
             # Currently uses default stages, extend args if custom stages needed
             stage_results = benchmark.run_curriculum_benchmark()
             # Add summary print here if desired

        elif args.mode == 'annotate':
             logger.info(f"Running Annotation Generation for '{args.annotate_set}'...")
             annotations = benchmark.generate_annotated_positions(
                 position_set_name=args.annotate_set,
                 num_positions=args.annotate_count,
                 depth=args.annotate_depth,
                 output_file=args.annotate_output
             )
             if annotations:
                  print(f"Annotation generation complete. {len(annotations)} positions annotated and saved to {args.annotate_output}")
             else:
                  print("Annotation generation failed or produced no data.")

    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
        logger.warning("Operation interrupted by user.")
    except chess.engine.EngineTerminatedError:
         print("\nERROR: Stockfish engine terminated unexpectedly. Please check engine setup.")
         logger.critical("Stockfish engine terminated unexpectedly.", exc_info=True)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logger.critical(f"Benchmark failed with an unexpected error: {e}", exc_info=True)

    finally:
        logger.info("ChessBenchmark script finished.")


# --- End of Part 5 / End of File ---