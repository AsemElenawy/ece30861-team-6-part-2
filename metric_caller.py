import multiprocessing
import re
import os
import importlib
import importlib.util
import time
import inspect
import sys
import threading

def parse_keys_from_string(key_string: str) -> list[str]:
    """Parses a comma-separated string of keys into a clean list."""
    if not key_string.strip():
        return []
    return [key.strip() for key in key_string.split(',')]

def logger_process(queue, log_file_path: str, log_level: int = 0) -> None:
    """
    Dedicated process that reads messages from `queue` and writes them to `log_file_path`.
    A None message is the sentinel to stop.
    """
    # Disable any inherited tracing/profiling in this child (avoid recursion/stack overflow)
    try:
        sys.settrace(lambda *a, **k: None)
        sys.setprofile(lambda *a, **k: None)
        if hasattr(threading, "settrace"):
            threading.settrace(lambda *a, **k: None)
        if hasattr(threading, "setprofile"):
            threading.setprofile(lambda *a, **k: None)
    except Exception:
        pass

    # Ensure the log directory exists
    try:
        dir_name = os.path.dirname(os.path.abspath(log_file_path)) or "."
        os.makedirs(dir_name, exist_ok=True)
    except Exception:
        pass

    # Open the log file and consume messages until sentinel None
    try:
        f = open(log_file_path, "w", encoding="ascii", errors="ignore")
    except Exception:
        # If the file can't be opened, still drain the queue so producers don't block
        while True:
            msg = queue.get()
            if msg is None:
                break
        return

    with f:
        while True:
            try:
                message = queue.get()
            except (EOFError, OSError):
                break
            if message is None:
                break
            try:
                f.write(f"{message}\n")
                f.flush()
            except Exception:
                # Ignore transient write/encoding errors to keep the logger robust
                pass

def process_worker(target_func, result_queue, log_queue, weight, func_name, *args):
    """
    Worker that executes the target function and handles any exceptions,
    returning a score of 0.0 upon failure.
    """
    # Disable any inherited tracing/profiling in worker processes as well
    try:
        sys.settrace(lambda *a, **k: None)
        sys.setprofile(lambda *a, **k: None)
        if hasattr(threading, "settrace"):
            threading.settrace(lambda *a, **k: None)
        if hasattr(threading, "setprofile"):
            threading.setprofile(lambda *a, **k: None)
    except Exception:
        pass

    start_time = time.perf_counter()
    try:
        score, time_taken = target_func(*args)
        result_queue.put((score, float(time_taken), float(weight), func_name))
    except Exception:
        time_taken = time.perf_counter() - start_time
        result_queue.put((0.0, time_taken, float(weight), func_name))

def load_available_functions(directory: str, *_, **__) -> dict:
    """
    Discovers and loads metric functions, sending output to the provided log queue.
    """
    functions = {}
    if os.path.isdir(directory):
        # load by filesystem
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                file_path = os.path.join(directory, filename)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        func = getattr(module, module_name)
                        functions[module_name] = func
                except Exception:
                    pass
        return functions
    try:
        pkg = importlib.import_module(directory)
        filesystem_dir = pkg.__path__[0]
    except Exception:
        return {}
    for filename in os.listdir(filesystem_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f"{directory}.{module_name}")
                func = getattr(module, module_name)
                functions[module_name] = func
            except (ImportError, AttributeError):
                pass
    return functions

def run_concurrently_from_file(tasks_filename: str, all_args_dict: dict, available_functions: dict, log_file: str):
    """
    Parses a file, runs functions concurrently, and directs all status updates to the log file.
    """
    script_verbosity = int(all_args_dict.get("verbosity", 0))
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()
    
    logger = multiprocessing.Process(target=logger_process, args=(log_queue, log_file))
    logger.start()

    if not isinstance(available_functions, dict):
        available_functions = load_available_functions(available_functions)
    
    # Load available functions and log the process

    all_args_dict['log_queue'] = log_queue

    line_pattern = re.compile(r'(\w+)\((.*)\)\s*([\d.]+)')
    processes = []
    results_queue = multiprocessing.Queue()
    total_weight = 0.0

    if script_verbosity > 0:
        log_queue.put(f"[INFO] Reading and parsing tasks from '{tasks_filename}'...")
    
    with open(tasks_filename, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            match = line_pattern.match(line)
            if not match:
                #log_queue.put(f"[WARNING] Skipped line {i}: Could not parse syntax: '{line}'.")
                continue

            func_name, keys_str, weight_str = match.groups()
            
            if func_name not in available_functions:
                #log_queue.put(f"[WARNING] Skipped line {i}: Function '{func_name}' not found.")
                continue

            target_func = available_functions[func_name]
            required_keys = parse_keys_from_string(keys_str)
            
            sig = inspect.signature(target_func)
            expected_count = len(sig.parameters)
            provided_count = len(required_keys)

            if provided_count != expected_count:
                #log_queue.put(f"[WARNING] Skipped line {i}: '{func_name}' expects {expected_count} args, but {provided_count} keys were provided.")
                continue
            
            if not all(key in all_args_dict for key in required_keys):
                missing = [key for key in required_keys if key not in all_args_dict]
                #log_queue.put(f"[WARNING] Skipped line {i}: Missing required keys in input dictionary: {missing}")
                continue

            resolved_args = [all_args_dict[key] for key in required_keys]
            weight = float(weight_str)
            process_args = (target_func, results_queue, log_queue, weight, func_name) + tuple(resolved_args)
            process = multiprocessing.Process(target=process_worker, args=process_args)
            processes.append(process)
            total_weight += weight
            if script_verbosity > 0:
                log_queue.put(f"[INFO] Queued: {func_name}(...) with weight {weight}")

    if not processes:
        if script_verbosity > 0:
            log_queue.put("[INFO] No valid tasks to run.")
        log_queue.put(None)
        logger.join()
        return {'net_score': 0.0}, {'net_score_latency': 0}
    
    if script_verbosity > 0:
        log_queue.put("[INFO] --- Starting all processes ---")
    concurrent_start_time = time.perf_counter()
    for p in processes: p.start()
    
    if script_verbosity > 0:
        log_queue.put("[INFO] --- Collecting results ---")
    times_dictionary = { "net_score_latency": 0.0 }
    scores_dictionary = {}
    weighted_score_sum = 0.0
    
    for _ in range(len(processes)):
        score, time_taken, weight, func_name = results_queue.get()
        scores_dictionary[func_name] = score
        times_dictionary[func_name] = round(time_taken * 1000)   
        if isinstance(score, dict):
            # e.g., size_score {device: score}
            avg = (sum(score.values()) / len(score)) if score else 0.0
            weighted_score_sum += avg * weight
        else:
            weighted_score_sum += float(score) * weight
    
    if total_weight > 0:
        net_score = weighted_score_sum / total_weight
    else:
        net_score = 0.0
            
    scores_dictionary['net_score'] = round(net_score,2)

    concurrent_end_time = time.perf_counter()
    times_dictionary["net_score_latency"] = round((concurrent_end_time - concurrent_start_time)*1000)

    for p in processes: p.join()
    
    if script_verbosity > 0:
        log_queue.put("[INFO] --- All processes have completed ---")
    
    
    log_queue.put(None)
    logger.join()
    
    return scores_dictionary, times_dictionary