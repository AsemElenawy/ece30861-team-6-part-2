import sys
import os
import time
from typing import Tuple

# --- Import Setup ---
# This block gets the project root onto the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Now we can import the function from the other file in the 'metrics' directory
from .ai_llm_generic_call import process_file_and_get_response

def performance_claims_metric(filename: str, verbosity: int, log_queue) -> Tuple[float, float]:
    """
    Calls an LLM to rate performance claims in a file, logging its progress to a queue.

    Args:
        filename (str): The absolute path to the input file (.md or .txt).
        verbosity (int): The verbosity level (0=silent, 1=INFO, 2=DEBUG).
        log_queue (multiprocessing.Queue): The queue to send log messages to.

    Returns:
        A tuple containing:
        - The score from the LLM as a float (0.0 on error).
        - The total time spent (float).
    """
    start_time = time.time()
    pid = os.getpid() # Get process ID for clear log messages

    instruction = """Evaluate the performance claims in this model's README and provide a score from 0 to 1.0:
- 1.0 = Explicit performance metrics with numbers (accuracy, F1, benchmarks, speed, etc.)
- 0.5-0.9 = General performance descriptions or comparisons ("better than", "faster", "improved")
- 0.1-0.4 = Mentions use cases or capabilities without specific metrics
- 0.0 = No performance information at all

Look for: accuracy scores, benchmark results, speed measurements, comparisons to other models, dataset performance, use case descriptions.

ONLY RESPOND WITH A SINGLE NUMBER (e.g., 0.75). NO OTHER TEXT.\n\n"""

    score = 0.0  # Default to 0.0 for failure cases
    llm_response_str = None

    try:
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [INFO] Calling LLM for performance claims on '{os.path.basename(filename)}'...")
            
        llm_response_str = process_file_and_get_response(filename, instruction, "gemma3:1b")

        # Safely convert the LLM's string response to a float
        if llm_response_str is not None:
            score = float(llm_response_str.strip())
            if verbosity >= 2: # Debug
                log_queue.put(f"[{pid}] [DEBUG] Successfully converted LLM response to score: {score}")
        else:
             if verbosity >= 1: # Informational
                log_queue.put(f"[{pid}] [WARNING] Received no response from LLM for performance claims metric.")

    except (ValueError, TypeError):
        if verbosity >= 1: # Informational
            log_queue.put(f"[{pid}] [WARNING] Could not convert LLM response '{llm_response_str}' to a float.")
        score = 0.0 # Ensure score is 0 on conversion failure
    except Exception as e:
        # Log any other critical error before the process terminates
        if verbosity >0:
            log_queue.put(f"[{pid}] [CRITICAL ERROR] in performance_claims_metric: {e}")
        raise # Re-raise the exception to be caught by the worker

    end_time = time.time()
    time_taken = end_time - start_time
    
    return score, time_taken