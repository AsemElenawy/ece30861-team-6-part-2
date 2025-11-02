import tempfile
import os
from metric_caller import run_concurrently_from_file

def test_run_concurrently_from_file_minimal():
    # Create a temporary directory to act as the metrics directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary tasks file with a single task name
        tasks_file = os.path.join(tmpdir, "tasks.txt")
        with open(tasks_file, "w") as f:
            f.write("dummy_task\n")

        # Provide a matching available_functions mapping for the signature
        def dummy_fn(task_name, all_args, log_queue):
            return {"dummy_task": 1.0}, 0.01

        available_functions = {"dummy_task": dummy_fn}

        # Call the function with correct named arguments
        scores, times = run_concurrently_from_file(
            tasks_filename=tasks_file,
            all_args_dict={},
            available_functions=available_functions,
            log_file=os.path.join(tmpdir, "dummy.log"),
        )

        # Basic assertions
        assert isinstance(scores, dict)
        assert isinstance(times, dict)