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


# Part 3: Analysis, Evaluation, Saving, and Utility Methods

class ChessBenchmark:
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