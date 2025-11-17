#!/usr/bin/env python3
import json
import os
import sys
import unittest
import coverage
import io

def main():
    # Set environment variables
    os.environ["LOG_FILE"] = os.path.join(os.getcwd(), "log.txt")
    os.environ["LOG_LEVEL"] = "2"

    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Buffer stdout and stderr
    buffer = io.StringIO()
    sys.stdout = buffer
    sys.stderr = buffer

    # Start coverage
    cov = coverage.Coverage(branch=True, omit=["/usr/lib/python3/*"])
    cov.start()

    # Discover and run tests in the "tests" directory
    loader = unittest.TestLoader()
    suite = loader.discover("tests")

    # Make TextTestRunner write to our buffer
    runner = unittest.TextTestRunner(stream=buffer, verbosity=2)
    result = runner.run(suite)

    # Stop coverage
    cov.stop()
    cov.save()

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    # Suppress coverage report output
    try:
        cov.report(file=io.StringIO())
        coverage_percent = cov.report(file=io.StringIO())
    except Exception:
        coverage_percent = 0.0

    # Restore stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Print JSON summary
    output = f"{passed}/{total} test cases passed. {int(coverage_percent)}% line coverage achieved."
    print(json.dumps({"summary": output}))

    # Exit code = 0 if all tests pass, nonzero otherwise
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main())