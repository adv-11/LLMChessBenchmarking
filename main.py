import os
import time
import json
import random
from typing import Optional
import numpy as np
import chess
import chess.engine
import chess.pgn
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import matplotlib.pyplot as plt

# Load environment variables from the .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
OPENAI_API_KEY = os.environ.get("ADV_OPENAI_DEV")

# Path to stockfish engine
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chess_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chess_benchmark")

# Common opening positions (FEN strings) for benchmarking
OPENING_POSITIONS = {
    "Standard": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "Sicilian Defense": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "French Defense": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "Caro-Kann": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "Kings Indian": "rnbqkb1r/pppppp1p/5np1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "Queens Gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
}

class ChessBenchmark:
    def __init__(self, llm_model="gpt-4", temperature=0.0, stockfish_depth=15):
        """
        Initialize the Chess Benchmark system.
        
        Args:
            llm_model (str): The LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo", etc.)
            temperature (float): Temperature parameter for the LLM
            stockfish_depth (int): Depth for Stockfish engine analysis
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.stockfish_depth = stockfish_depth
        
        # Initialize the LLM
        self.chat = ChatOpenAI(
            model_name=llm_model, 
            temperature=temperature, 
            api_key=OPENAI_API_KEY
        )
        
        # Statistics tracking
        self.stats = {
            "games_played": 0,
            "llm_wins": 0,
            "stockfish_wins": 0,
            "draws": 0,
            "illegal_moves": 0,
            "blunders": 0,
            "avg_move_time": 0,
            "opening_success": 0,
            "game_history": [],
            "avg_game_length": 0,
            "total_moves": 0,
            "eval_history": [],
            "elo_rating": 1500,  # Starting Elo rating
        }
        
        # Create results directory if it doesn't exist
        self.results_dir = "benchmark_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Initialized ChessBenchmark with model: {llm_model}, temp: {temperature}")
    
    def llm_make_move(self, board, conversation_history, retry_count=0):
        """
        Generate the next move for White using LLM with improved error handling.
        
        Args:
            board (chess.Board): Current chess board
            conversation_history (str): History of moves and commentary
            retry_count (int): Number of retries attempted
            
        Returns:
            str: UCI move string
        """
        if retry_count > 3:
            # If we've retried too many times, get a stockfish move as fallback
            logger.warning("Too many LLM retries, falling back to Stockfish")
            self.stats["illegal_moves"] += 1
            return self.get_stockfish_move(board)
        
        # Current board state in FEN and visual format
        fen = board.fen()
        visual_board = self._get_visual_board(board)
        legal_moves = [move.uci() for move in board.legal_moves]
        
        if not legal_moves:
            logger.warning("No legal moves available - game should be over")
            return None
        
        # Create an improved prompt with more context and clearer instructions
        prompt = (
            f"You are a chess grandmaster playing as White. Your objective is to win. "
            f"\nCurrent board state (White = uppercase, Black = lowercase):\n{visual_board}\n"
            f"Current position in FEN notation: {fen}\n"
            f"Move history:\n{conversation_history}\n"
            f"Legal moves in UCI format: {', '.join(legal_moves)}\n\n"
            "Analyze the position carefully and provide the best move for White in UCI format (e.g., e2e4). "
            "Respond ONLY with a single valid UCI move from the provided list, with no additional text."
        )
        
        # Use LangChain's ChatOpenAI to generate a response
        messages = [
            SystemMessage(content=(
                "You are a chess grandmaster. Respond ONLY with a valid UCI move "
                "from the legal moves provided. Do not include explanations or commentary."
            )),
            HumanMessage(content=prompt)
        ]
        
        start_time = time.time()
        
        try:
            # Invoke the chat model to get a response
            response = self.chat.invoke(messages)
            
            # Extract the move and clean it
            move = response.content.strip().lower()
            # Remove any non-UCI elements (handles cases where LLM adds extra text)
            for legal_move in legal_moves:
                if legal_move in move:
                    move = legal_move
                    break
                    
            move_time = time.time() - start_time
            
            # Update the average move time
            if self.stats["total_moves"] > 0:
                self.stats["avg_move_time"] = (
                    (self.stats["avg_move_time"] * self.stats["total_moves"]) + move_time
                ) / (self.stats["total_moves"] + 1)
            else:
                self.stats["avg_move_time"] = move_time
                
            # Validate the move more strictly
            if move not in legal_moves:
                logger.warning(f"Invalid move received: '{move}'. Legal moves: {legal_moves}")
                return self.llm_make_move(board, conversation_history, retry_count + 1)
                
            return move
            
        except Exception as e:
            logger.error(f"Error in LLM move generation: {e}")
            return self.llm_make_move(board, conversation_history, retry_count + 1)
    
    def get_stockfish_move(self, board, thinking_time=1.0):
        """
        Get the best move from Stockfish.
        
        Args:
            board (chess.Board): Current chess board
            thinking_time (float): Time in seconds for Stockfish to think
            
        Returns:
            str: UCI move string or None if no legal moves
        """
        if len(list(board.legal_moves)) == 0:
            logger.warning("No legal moves available for Stockfish")
            return None
            
        try:
            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
                # Set options for more consistent play if needed
                # stockfish.configure({"Skill Level": 20})  # Maximum skill
                
                # Get the best move
                result = stockfish.play(
                    board, 
                    chess.engine.Limit(time=thinking_time, depth=self.stockfish_depth)
                )
                return result.move.uci()
        except Exception as e:
            logger.error(f"Error with Stockfish: {e}")
            # Fallback: use a random legal move if Stockfish fails
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves).uci()
            return None
    
    def evaluate_position(self, board):
        """
        Get Stockfish's evaluation of the current position.
        
        Args:
            board (chess.Board): Current chess board
            
        Returns:
            float: Evaluation score in centipawns (positive is good for White)
        """
        try:
            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
                info = stockfish.analyse(
                    board, 
                    chess.engine.Limit(depth=self.stockfish_depth)
                )
                
                # Convert score to centipawns from White's perspective
                score = info["score"].relative.score(mate_score=10000)
                return score if score is not None else 0
                
        except Exception as e:
            logger.error(f"Error in position evaluation: {e}")
            return 0
    
    def detect_blunder(self, board, prev_eval, current_eval, threshold=150):
        """
        Detect if a move was a blunder based on evaluation delta with improved logic.
        
        Args:
            board (chess.Board): Current chess board
            prev_eval (float): Previous position evaluation
            current_eval (float): Current position evaluation
            threshold (int): Centipawn threshold for blunder detection
            
        Returns:
            bool: True if the move was a blunder
        """
        # Handle extremely high evaluation scores (checkmate scenarios)
        if abs(prev_eval) > 9000 or abs(current_eval) > 9000:
            return False  # Skip blunder detection for winning/losing positions
        
        # For White's move: a significant decrease in evaluation is a blunder
        # For Black's move: a significant increase in evaluation is a blunder
        if board.turn:  # White just moved (it's now Black's turn)
            return (prev_eval - current_eval) >= threshold
        else:  # Black just moved (it's now White's turn)
            return (current_eval - prev_eval) >= threshold
            
    def detect_opening(self, board):
        """
        Detect if the current position matches a known opening.
        
        Args:
            board (chess.Board): Current chess board
            
        Returns:
            str or None: Name of the opening if detected, None otherwise
        """
        # This is a placeholder - in a real implementation you would use a
        # database of opening positions or an ECO (Encyclopedia of Chess Openings) database
        fen_without_moves = " ".join(board.fen().split(" ")[:4])
        for name, opening_fen in OPENING_POSITIONS.items():
            opening_fen_trimmed = " ".join(opening_fen.split(" ")[:4])
            if fen_without_moves == opening_fen_trimmed:
                return name
        return None
    
    def update_elo(self, result, opponent_elo=2800, k_factor=32):
        """
        Update the LLM's Elo rating based on game result.
        
        Args:
            result (float): Game result (1 for win, 0.5 for draw, 0 for loss)
            opponent_elo (int): Opponent's Elo rating (Stockfish's estimated Elo)
            k_factor (int): K-factor for Elo calculation
            
        Returns:
            int: New Elo rating
        """
        # Calculate expected score
        expected = 1 / (1 + 10 ** ((opponent_elo - self.stats["elo_rating"]) / 400))
        
        # Calculate new rating
        new_rating = self.stats["elo_rating"] + k_factor * (result - expected)
        self.stats["elo_rating"] = int(new_rating)
        
        return self.stats["elo_rating"]
    
    def _get_visual_board(self, board):
        """
        Generate a visual representation of the board.
        
        Args:
            board (chess.Board): Chess board
            
        Returns:
            str: Visual representation of the board
        """
        visual = str(board)
        # Enhance visualization with rank and file labels
        ranks = "87654321"
        files = "abcdefgh"
        
        # Add file labels at the bottom
        visual += "\n  " + " ".join(files)
        
        # Add rank labels to the right side
        lines = visual.split("\n")
        for i, line in enumerate(lines[:-1]):  # Skip the newly added files line
            if i < 8:  # Only add to the actual board rows
                lines[i] = line + " " + ranks[i]
        
        return "\n".join(lines)
    
    def save_game_to_pgn(self, board, game_data):
        """
        Save a game to PGN format with improved error handling for illegal moves.
        
        Args:
            board (chess.Board): Final board state
            game_data (dict): Game metadata
            
        Returns:
            str: Path to the saved PGN file
        """
        try:
            # Create a new game
            game = chess.pgn.Game()
            
            # Set game headers
            game.headers["Event"] = "LLM Chess Benchmark"
            game.headers["Site"] = "LLM vs Stockfish"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(self.stats["games_played"])
            game.headers["White"] = f"LLM ({self.llm_model})"
            game.headers["Black"] = f"Stockfish (Depth {self.stockfish_depth})"
            game.headers["Result"] = game_data["result"]
            
            # Add custom headers
            game.headers["LLMBlunders"] = str(game_data["llm_blunders"])
            game.headers["FinalEval"] = str(game_data["final_eval"])
            game.headers["Opening"] = game_data["opening"] if game_data["opening"] else "Unknown"
            
            # Reconstruct the game move by move from scratch
            node = game
            
            # Use a fresh board to validate moves
            temp_board = chess.Board()
            if game_data.get("starting_fen"):
                temp_board.set_fen(game_data["starting_fen"])
            
            # Safely process each move
            for move_idx, move_uci in enumerate(game_data["moves"]):
                try:
                    move = chess.Move.from_uci(move_uci)
                    
                    # Make sure the move is legal
                    if move not in temp_board.legal_moves:
                        logger.warning(f"Skipping illegal move {move_uci} at position {move_idx}")
                        continue
                    
                    # Add the move and update board
                    node = node.add_variation(move)
                    temp_board.push(move)
                    
                except Exception as e:
                    logger.error(f"Error processing move {move_uci} at index {move_idx}: {e}")
                    # Skip invalid moves but continue processing
            
            # Save to file
            filename = f"{self.results_dir}/game_{self.stats['games_played']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
            
            # Write the PGN data to file
            with open(filename, "w") as f:
                exporter = chess.pgn.FileExporter(f)
                game.accept(exporter)
            
            # Create a supplementary text file with all moves for reference
            moves_filename = f"{self.results_dir}/game_{self.stats['games_played']}_moves.txt"
            with open(moves_filename, "w") as f:
                f.write(f"Starting FEN: {game_data.get('starting_fen', 'Standard')}\n")
                f.write(f"Result: {game_data['result']}\n")
                f.write("Moves (UCI format):\n")
                for i, move in enumerate(game_data["moves"]):
                    f.write(f"{i+1}. {move}\n")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving game to PGN: {e}")
            # Create a simplified representation as fallback
            fallback_filename = f"{self.results_dir}/game_{self.stats['games_played']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_fallback.txt"
            with open(fallback_filename, "w") as f:
                f.write(f"Game data: {json.dumps(game_data, indent=2)}")
            return fallback_filename
    
    def generate_report(self):
        """
        Generate a comprehensive benchmarking report.
        
        Returns:
            dict: Report data
        """
        if self.stats["games_played"] == 0:
            return {"error": "No games played yet"}
        
        # Calculate win percentage
        win_rate = self.stats["llm_wins"] / self.stats["games_played"] * 100
        
        # Calculate average game length - use try/except to handle potential errors
        try:
            avg_game_length = sum([len(game["moves"]) for game in self.stats["game_history"]]) / (2 * max(1, self.stats["games_played"]))
        except Exception:
            avg_game_length = 0
        
        # Calculate average evaluation trend with error handling
        try:
            eval_trend = np.mean([game["eval_history"][-1] - game["eval_history"][0] 
                                for game in self.stats["game_history"] 
                                if game["eval_history"] and len(game["eval_history"]) > 1])
        except Exception:
            eval_trend = 0
        
        # Calculate blunder rate
        blunder_rate = self.stats["blunders"] / max(1, self.stats["total_moves"] / 2) * 100
        
        report = {
            "model": self.llm_model,
            "games_played": self.stats["games_played"],
            "win_rate": win_rate,
            "draw_rate": (self.stats["draws"] / self.stats["games_played"]) * 100,
            "avg_game_length": avg_game_length,
            "avg_move_time": self.stats["avg_move_time"],
            "blunder_rate": blunder_rate,
            "illegal_move_rate": self.stats["illegal_moves"] / max(1, self.stats["total_moves"] / 2) * 100,
            "opening_success_rate": (self.stats["opening_success"] / max(1, self.stats["games_played"])) * 100,
            "position_eval_trend": eval_trend,
            "elo_rating": self.stats["elo_rating"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save report to file
        report_file = f"{self.results_dir}/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def visualize_results(self):
        """
        Create visualizations of benchmark results.
        
        Returns:
            list: Paths to generated visualization files
        """
        if self.stats["games_played"] == 0:
            return ["No games played yet"]
            
        # Create a list to store generated file paths
        visualization_files = []
        
        try:
            # 1. Win/Loss/Draw pie chart
            plt.figure(figsize=(10, 6))
            labels = ['LLM Wins', 'Stockfish Wins', 'Draws']
            sizes = [self.stats["llm_wins"], self.stats["stockfish_wins"], self.stats["draws"]]
            colors = ['lightgreen', 'lightcoral', 'lightskyblue']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            plt.title(f'Game Outcomes ({self.llm_model} vs Stockfish)')
            
            pie_chart_file = f"{self.results_dir}/game_outcomes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(pie_chart_file)
            plt.close()
            visualization_files.append(pie_chart_file)
            
            # 2. Evaluation over moves (aggregated across games)
            plt.figure(figsize=(12, 6))
            
            # Plot each game's evaluation history
            for i, game in enumerate(self.stats["game_history"]):
                if game.get("eval_history") and len(game["eval_history"]) > 1:
                    x = list(range(len(game["eval_history"])))
                    y = [eval_score / 100 for eval_score in game["eval_history"]]  # Convert to pawn units
                    plt.plot(x, y, alpha=0.3, label=f"Game {i+1}" if i < 5 else "")
            
            # Plot the average evaluation trend
            all_evals = []
            max_length = max([len(game.get("eval_history", [])) for game in self.stats["game_history"]], default=0)
            
            if max_length > 0:
                for i in range(max_length):
                    valid_evals = [game["eval_history"][i] for game in self.stats["game_history"] 
                                if game.get("eval_history") and i < len(game["eval_history"])]
                    if valid_evals:
                        all_evals.append(sum(valid_evals) / len(valid_evals))
                
                if all_evals:
                    plt.plot(range(len(all_evals)), [e/100 for e in all_evals], 'k-', linewidth=2, label="Average")
            
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            plt.xlabel('Move Number')
            plt.ylabel('Evaluation (pawns)')
            plt.title('Position Evaluation Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            eval_chart_file = f"{self.results_dir}/evaluation_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(eval_chart_file)
            plt.close()
            visualization_files.append(eval_chart_file)
            
            # 3. Blunder analysis
            plt.figure(figsize=(10, 6))
            blunder_counts = [game.get("llm_blunders", 0) for game in self.stats["game_history"]]
            if blunder_counts:
                plt.bar(range(1, len(blunder_counts)+1), blunder_counts, color='indianred')
                avg_blunders = sum(blunder_counts)/max(1, len(blunder_counts))
                plt.axhline(y=avg_blunders, color='black', linestyle='--', alpha=0.7, 
                        label=f'Avg: {avg_blunders:.2f}')
                plt.xlabel('Game Number')
                plt.ylabel('Number of Blunders')
                plt.title('LLM Blunders per Game')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                blunder_chart_file = f"{self.results_dir}/blunder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(blunder_chart_file)
                plt.close()
                visualization_files.append(blunder_chart_file)
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return [f"Error generating visualizations: {e}"]
            
        return visualization_files
    
    def play_game(self, opening=None, custom_fen=None):
        """
        Play a complete game between LLM (White) and Stockfish (Black).
        
        Args:
            opening (str): Opening name to use from the OPENING_POSITIONS dict
            custom_fen (str): Custom starting position in FEN notation
            
        Returns:
            dict: Game results and statistics
        """
        # Initialize the board with the specified opening or starting position
        starting_fen = None
        if custom_fen:
            board = chess.Board(custom_fen)
            starting_fen = custom_fen
        elif opening and opening in OPENING_POSITIONS:
            board = chess.Board(OPENING_POSITIONS[opening])
            starting_fen = OPENING_POSITIONS[opening]
        else:
            board = chess.Board()  # Standard starting position
            starting_fen = board.fen()
        
        conversation_history = ""
        moves = []
        eval_history = []
        llm_blunders = 0
        detected_opening = None
        
        # Get initial evaluation
        current_eval = self.evaluate_position(board)
        eval_history.append(current_eval)
        
        # Play until game is over
        move_count = 0
        while not board.is_game_over():
            # Store the previous evaluation for blunder detection
            prev_eval = current_eval
            
            if board.turn == chess.WHITE:  # LLM's turn (White)
                logger.info(f"Position FEN: {board.fen()}")
                logger.info(f"Current evaluation: {current_eval/100:.2f}")
                
                # Check if there are legal moves
                if not list(board.legal_moves):
                    logger.warning("No legal moves for LLM - game should be over")
                    break
                
                move = self.llm_make_move(board, conversation_history)
                
                # Check if we got a move
                if not move:
                    logger.warning("No move returned from LLM, ending game")
                    break
                
                # Check if the move is valid before pushing
                try:
                    chess_move = chess.Move.from_uci(move)
                    if chess_move in board.legal_moves:
                        logger.info(f"White (LLM) move: {move}")
                        board.push(chess_move)
                        moves.append(move)
                        conversation_history += f"White: {move}\n"
                        move_count += 1
                        self.stats["total_moves"] += 1
                        
                        # Detect opening if we're early in the game
                        if move_count <= 10 and not detected_opening:
                            detected_opening = self.detect_opening(board)
                            if detected_opening:
                                logger.info(f"Opening detected: {detected_opening}")
                                self.stats["opening_success"] += 1
                        
                        # Evaluate position after move
                        current_eval = self.evaluate_position(board)
                        eval_history.append(current_eval)
                        
                        # Detect blunders
                        if self.detect_blunder(board, prev_eval, current_eval):
                            llm_blunders += 1
                            self.stats["blunders"] += 1
                            logger.info(f"Blunder detected! Eval before: {prev_eval/100:.2f}, after: {current_eval/100:.2f}")
                    else:
                        # This should rarely happen with our improved move validation
                        logger.warning(f"Invalid move received from LLM: {move}")
                        self.stats["illegal_moves"] += 1
                        # Try to use stockfish as fallback
                        fallback_move = self.get_stockfish_move(board)
                        if fallback_move:
                            logger.info(f"Using stockfish fallback move: {fallback_move}")
                            board.push(chess.Move.from_uci(fallback_move))
                            moves.append(fallback_move)
                            conversation_history += f"White (fallback): {fallback_move}\n"
                            move_count += 1
                        else:
                            break
                except Exception as e:
                    logger.error(f"Error processing LLM move: {e}")
                    self.stats["illegal_moves"] += 1
                    break
            else:  # Stockfish's turn (Black)
                # Check if there are legal moves
                if not list(board.legal_moves):
                    logger.warning("No legal moves for Stockfish - game should be over")
                    break
                    
                move = self.get_stockfish_move(board)
                
                if move:
                    try:
                        chess_move = chess.Move.from_uci(move)
                        if chess_move in board.legal_moves:
                            logger.info(f"Black (Stockfish) move: {move}")
                            board.push(chess_move)
                            moves.append(move)
                            conversation_history += f"Black: {move}\n"
                            move_count += 1
                            self.stats["total_moves"] += 1
                            
                            # Evaluate position after move
                            current_eval = self.evaluate_position(board)
                            eval_history.append(current_eval)
                        else:
                            logger.error(f"Invalid move from Stockfish: {move}")
                            break
                    except Exception as e:
                        logger.error(f"Error processing Stockfish move: {e}")
                        break
                else:
                    logger.error("No move returned from Stockfish")
                    break
            
            # Safety check to prevent infinite games
            if move_count > 200:
                logger.warning("Game reached move limit (200), ending.")
                break
        
        # Game finished - determine result
        try:
            result = board.result()
        except Exception:
            # Default to draw if we can't determine the result
            logger.warning("Could not determine game result, defaulting to draw")
            result = "1/2-1/2"
        
        final_eval = self.evaluate_position(board)
        
        # Update statistics
        self.stats["games_played"] += 1
        if result == "1-0":
            self.stats["llm_wins"] += 1
            elo_result = 1.0
        elif result == "0-1":
            self.stats["stockfish_wins"] += 1
            elo_result = 0.0
        else:  # Draw
            self.stats["draws"] += 1
            elo_result = 0.5
        
        # Update Elo rating
        new_elo = self.update_elo(elo_result)
        logger.info(f"New Elo rating: {new_elo}")
        
        # Prepare game data
        game_data = {
            "result": result,
            "moves": moves,
            "eval_history": eval_history,
            "llm_blunders": llm_blunders,
            "final_eval": final_eval,
            "opening": detected_opening or opening or "Standard",
            "elo_change": new_elo - (self.stats["elo_rating"] - (new_elo - self.stats["elo_rating"])),
            "starting_fen": starting_fen
        }
        
        # Save game data
        self.stats["game_history"].append(game_data)
        
        # Save the game in PGN format
        pgn_file = self.save_game_to_pgn(board, game_data)
        logger.info(f"Game saved to {pgn_file}")
        
        return game_data
    
    def run_benchmark(self, num_games=5, openings=None):
        """
        Run a chess benchmark of multiple games.
        
        Args:
            num_games (int): Number of games to play
            openings (list): List of openings to use. If None, random openings will be selected.
            
        Returns:
            dict: Benchmark results
        """
        logger.info(f"Starting benchmark: {num_games} games with {self.llm_model}")
        
        # Initialize results
        results = {
            "games": [],
            "llm_wins": 0,
            "stockfish_wins": 0,
            "draws": 0,
            "avg_blunders": 0,
            "avg_eval": 0,
            "elo_rating": self.stats["elo_rating"],
            "initial_elo": self.stats["elo_rating"]
        }
        
        # Create a folder for this benchmark run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = f"{self.results_dir}/benchmark_{timestamp}"
        os.makedirs(benchmark_dir, exist_ok=True)
        self.results_dir = benchmark_dir  # Update the results directory for this run
        
        # If openings is specified, use them; otherwise randomly select from available openings
        opening_list = openings if openings else list(OPENING_POSITIONS.keys())
        selected_openings = random.choices(opening_list, k=num_games) if len(opening_list) > 0 else ["Standard"] * num_games
        
        total_blunders = 0
        total_eval = 0
        
        # Play the specified number of games
        for game_idx in range(num_games):
            opening = selected_openings[game_idx]
            logger.info(f"Game {game_idx+1}/{num_games} with opening: {opening}")
            
            try:
                # Play the game
                game_data = self.play_game(opening=opening)
                
                # Record game results
                results["games"].append(game_data)
                
                if game_data["result"] == "1-0":
                    results["llm_wins"] += 1
                elif game_data["result"] == "0-1":
                    results["stockfish_wins"] += 1
                else:
                    results["draws"] += 1
                    
                total_blunders += game_data["llm_blunders"]
                total_eval += game_data["final_eval"]
                
                # Save individual game results
                game_json = f"{self.results_dir}/game_{game_idx+1}_data.json"
                with open(game_json, "w") as f:
                    json.dump(game_data, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error playing game {game_idx+1}: {e}")
                # Continue with the next game
        
        # Calculate averages
        if num_games > 0:
            results["avg_blunders"] = total_blunders / num_games
            results["avg_eval"] = total_eval / num_games
        
        # Finalize Elo rating after all games
        results["elo_rating"] = self.stats["elo_rating"]
        results["elo_change"] = results["elo_rating"] - results["initial_elo"]
        
        # Generate a report and visualizations
        report = self.generate_report()
        results["report"] = report
        
        visualization_files = self.visualize_results()
        results["visualizations"] = visualization_files
        
        # Save the final benchmark results
        benchmark_results_file = f"{self.results_dir}/benchmark_results.json"
        with open(benchmark_results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark completed. Results saved to {benchmark_results_file}")
        return results


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run LLM Chess Benchmark')
    parser.add_argument('--model', type=str, default="gpt-4", help='LLM model to use')
    parser.add_argument('--games', type=int, default=5, help='Number of games to play')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature parameter for LLM')
    parser.add_argument('--depth', type=int, default=15, help='Stockfish depth')
    parser.add_argument('--openings', type=str, nargs='+', help='List of openings to use')
    args = parser.parse_args()
    
    # Select openings to use
    selected_openings = args.openings if args.openings else None
    
    # Configure and run the benchmark
    benchmark = ChessBenchmark(
        llm_model=args.model,
        temperature=args.temperature,
        stockfish_depth=args.depth
    )
    
    # Set number of games
    num_games = args.games
    
    # Run the benchmark
    try:
        results = benchmark.run_benchmark(num_games=num_games, openings=selected_openings)
        print(f"Benchmark completed. Final Elo rating: {results['elo_rating']}")
    except KeyboardInterrupt:
        print("Benchmark interrupted by user.")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")