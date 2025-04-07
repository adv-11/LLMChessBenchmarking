import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import time
import chess
import chess.engine  # Ensure this import works
from stockfish import Stockfish

# Load environment variables from the .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
OPENAI_API_KEY = os.environ.get("ADV_OPENAI_DEV")

# Instantiate the LangChain ChatOpenAI model (using GPT-4)
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0, api_key=OPENAI_API_KEY)

STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Set the path to your Stockfish binary here

def llm_make_move(conversation_history: str) -> str:
    """
    Generate the next move for White using LangChain's ChatOpenAI (GPT-4),
    providing the conversation history as context.
    """
    # Improved prompt to specify only legal UCI moves
    prompt = (
        f"You are a chess grandmaster playing as White. Your objective is to win. "
        f"Here is the move history so far (White moves first):\n"
        f"{conversation_history}\n"
        "Please provide the best next move for White in UCI format (e.g., e2e4). "
        "Only respond with the move in UCI format, without any additional text or explanation. "
        "Make sure the move is legal according to the current board state."
    )
    
    # Use LangChain's ChatOpenAI to generate a response
    messages = [
        SystemMessage(content="You are a chess grandmaster. Respond only with a valid UCI move."),
        HumanMessage(content=prompt)
    ]
    
    # Invoke the chat model to get a response
    response = chat.invoke(messages)
    
    # Extract the move from the response content
    move = response.content.strip()

    # Ensure the move is in UCI format (length 4 or 5)
    if len(move) not in [4, 5]:
        print(f"Invalid move received: {move}. Asking LLM to regenerate...")
        return llm_make_move(conversation_history)  # Retry if the move is invalid
    
    return move

def get_stockfish_move(board):
    """
    Get the best move from Stockfish.
    This function uses Stockfish to evaluate the board and return the best move in UCI format.
    """
    try:
        # Open the Stockfish engine and get the best move
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as stockfish:
            result = stockfish.play(board, chess.engine.Limit(time=1.0))  # 1-second thinking time
            return result.move.uci()
    except Exception as e:
        print(f"Error occurred while interacting with Stockfish: {e}")
        return None

def is_legal_move(move: str, board: chess.Board) -> bool:
    """
    Check if the given move is legal according to the current board state.
    """
    try:
        chess.Move.from_uci(move)  # Try to parse the move using UCI format
        if move not in [m.uci() for m in board.legal_moves]:
            return False
        return True
    except ValueError:
        return False

def play_game():
    """
    Simulate a chess game between White (AI-powered LLM) and Black (Stockfish).
    """
    board = chess.Board()  # Initialize the chess board
    conversation_history = ""  # Initialize conversation history
    
    # Play the game until it's over
    while not board.is_game_over():
        # Get move for White (LLM)
        move_white = llm_make_move(conversation_history)
        
        # Ensure the move is legal
        if not is_legal_move(move_white, board):
            print(f"Illegal move received: {move_white}. Asking LLM to regenerate...")
            conversation_history += f"LLM generated an illegal move: {move_white}. Regenerating...\n"
            continue  # Skip this move and retry
        
        print(f"White Move: {move_white}")
        
        # Try to push the move to the board
        try:
            board.push(chess.Move.from_uci(move_white))
        except chess.InvalidMoveError:
            print(f"Invalid move received from LLM: {move_white}. Retrying...")
            continue  # Skip this move and retry
        
        conversation_history += f"White: {move_white}\n"
        
        if board.is_game_over():
            break
        
        # Get move for Black (Stockfish)
        move_black = get_stockfish_move(board)
        
        if not move_black:
            print("Error with Stockfish. Exiting game.")
            break  # Exit the game if there's an issue with Stockfish
        
        print(f"Black Move: {move_black}")
        
        # Ensure the move is legal for Black
        if not is_legal_move(move_black, board):
            print(f"Illegal move received from Stockfish: {move_black}. Retrying...")
            continue  # Skip this move and retry
        
        # Try to push the move to the board
        try:
            board.push(chess.Move.from_uci(move_black))
        except chess.InvalidMoveError:
            print(f"Invalid move received from Stockfish: {move_black}. Retrying...")
            continue  # Skip this move and retry
        
        conversation_history += f"Black: {move_black}\n"
    
    return board.result()

def run_benchmark():
    """
    Run the chess benchmarking process.
    """
    total_score = 0
    total_games = 5  # Example, you can adjust based on your needs
    for game_num in range(1, total_games + 1):
        print(f"Starting Game {game_num}...")
        game_result = play_game()
        print(f"Game {game_num} result: {game_result}")
        
        # Update score or any other metrics based on the game result
        if game_result == "1-0":
            total_score += 1  # White wins
        elif game_result == "0-1":
            total_score -= 1  # Black wins
    
    avg_score = total_score / total_games
    print(f"Average Score after {total_games} games: {avg_score}")

if __name__ == "__main__":
    run_benchmark()
