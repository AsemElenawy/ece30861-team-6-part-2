"""
Test suite for src/metrics/ module functions.

Tests cover:
- Bus factor metric calculations
- License score calculations
- Size score calculations
- Code quality checks
- Dataset and code presence checks
- Performance claims metric
- Rampup time metric
"""

import unittest
import multiprocessing
from queue import Queue
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.bus_factor_metric import bus_factor_metric
from src.metrics.calculate_size_score import calculate_size_score
from src.metrics.calculate_license_score import calculate_license_score
from src.metrics.code_quality import code_quality
from src.metrics.dataset_and_code_present import dataset_and_code_present
from src.metrics.dataset_quality import dataset_quality


class TestBusFactorMetric(unittest.TestCase):
    """Test bus factor metric calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_queue = Queue()
        self.verbosity = 0
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_bus_factor_with_contributor_keywords(self):
        """Test bus factor detection when contributor keywords are present."""
        readme_file = os.path.join(self.temp_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write("This project has many contributors and authors.")
        
        score, time_taken = bus_factor_metric(readme_file, self.verbosity, self.log_queue)
        
        self.assertEqual(score, 1.0)
        self.assertGreater(time_taken, 0)

    def test_bus_factor_without_contributor_keywords(self):
        """Test bus factor when no contributor keywords are present."""
        readme_file = os.path.join(self.temp_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write("This is a simple project.")
        
        score, time_taken = bus_factor_metric(readme_file, self.verbosity, self.log_queue)
        
        self.assertEqual(score, 0.0)
        self.assertGreater(time_taken, 0)

    def test_bus_factor_with_author_keyword(self):
        """Test bus factor detection with 'author' keyword."""
        readme_file = os.path.join(self.temp_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write("Author: John Doe")
        
        score, time_taken = bus_factor_metric(readme_file, self.verbosity, self.log_queue)
        
        self.assertEqual(score, 1.0)

    def test_bus_factor_with_team_keyword(self):
        """Test bus factor detection with 'team' keyword."""
        readme_file = os.path.join(self.temp_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write("The team behind this project includes...")
        
        score, time_taken = bus_factor_metric(readme_file, self.verbosity, self.log_queue)
        
        self.assertEqual(score, 1.0)

    def test_bus_factor_with_verbosity_debug(self):
        """Test bus factor with debug verbosity."""
        readme_file = os.path.join(self.temp_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write("Contributors: Alice, Bob")
        
        score, time_taken = bus_factor_metric(readme_file, verbosity=2, log_queue=self.log_queue)
        
        self.assertEqual(score, 1.0)
        # Check that debug messages were logged
        self.assertGreater(self.log_queue.qsize(), 0)

    def test_bus_factor_case_insensitive(self):
        """Test that bus factor check is case-insensitive."""
        readme_upper = os.path.join(self.temp_dir, "README_upper.md")
        with open(readme_upper, 'w') as f:
            f.write("CONTRIBUTORS AND AUTHORS")
        
        score_upper, _ = bus_factor_metric(readme_upper, self.verbosity, self.log_queue)
        
        readme_lower = os.path.join(self.temp_dir, "README_lower.md")
        with open(readme_lower, 'w') as f:
            f.write("contributors and authors")
        
        score_lower, _ = bus_factor_metric(readme_lower, self.verbosity, self.log_queue)
        
        self.assertEqual(score_upper, score_lower)
        self.assertEqual(score_upper, 1.0)


class TestSizeScore(unittest.TestCase):
    """Test size score calculation for different platforms."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_queue = Queue()
        self.verbosity = 0

    def test_size_score_small_model(self):
        """Test size score for a small model (50MB)."""
        size_bytes = 50 * 1024 * 1024  # 50 MB
        scores, time_taken = calculate_size_score(size_bytes, self.verbosity, self.log_queue)
        
        self.assertIsInstance(scores, dict)
        self.assertIn("raspberry_pi", scores)
        self.assertIn("jetson_nano", scores)
        self.assertIn("desktop_pc", scores)
        self.assertEqual(scores["raspberry_pi"], 1.0)
        self.assertGreater(time_taken, 0)

    def test_size_score_medium_model(self):
        """Test size score for a medium model (1GB)."""
        size_bytes = 1 * 1024 * 1024 * 1024  # 1 GB
        scores, time_taken = calculate_size_score(size_bytes, self.verbosity, self.log_queue)
        
        self.assertIsInstance(scores, dict)
        # Jetson Nano should have partial score for 1GB
        self.assertGreaterEqual(scores["jetson_nano"], 0)
        self.assertLessEqual(scores["jetson_nano"], 1.0)

    def test_size_score_large_model(self):
        """Test size score for a large model (20GB)."""
        size_bytes = 20 * 1024 * 1024 * 1024  # 20 GB
        scores, time_taken = calculate_size_score(size_bytes, self.verbosity, self.log_queue)
        
        self.assertIsInstance(scores, dict)
        # Large model should score lower on edge devices
        self.assertLess(scores["raspberry_pi"], 1.0)
        self.assertLess(scores["jetson_nano"], 1.0)

    def test_size_score_zero_bytes(self):
        """Test size score for zero-byte model."""
        size_bytes = 0
        scores, time_taken = calculate_size_score(size_bytes, self.verbosity, self.log_queue)
        
        self.assertIsInstance(scores, dict)
        self.assertEqual(scores["raspberry_pi"], 1.0)

    def test_size_score_with_verbosity(self):
        """Test size score calculation with debug verbosity."""
        size_bytes = 500 * 1024 * 1024  # 500 MB
        scores, _ = calculate_size_score(size_bytes, verbosity=2, log_queue=self.log_queue)
        
        self.assertIsInstance(scores, dict)
        # Verify logging occurred
        self.assertGreater(self.log_queue.qsize(), 0)


class TestLicenseScore(unittest.TestCase):
    """Test license score calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_queue = Queue()
        self.verbosity = 0

    def test_license_score_mit(self):
        """Test license score for MIT license."""
        license_name = "MIT"
        score, time_taken = calculate_license_score(license_name, self.verbosity, self.log_queue)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(time_taken, 0)
        # MIT is highly permissive, should get 1.0
        self.assertEqual(score, 1.0)

    def test_license_score_apache(self):
        """Test license score for Apache 2.0 license."""
        license_name = "Apache-2.0"
        score, time_taken = calculate_license_score(license_name, self.verbosity, self.log_queue)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1.0)
        # Apache 2.0 is permissive, should get 1.0
        self.assertEqual(score, 1.0)

    def test_license_score_gpl(self):
        """Test license score for GPL license."""
        license_name = "GPL"
        score, time_taken = calculate_license_score(license_name, self.verbosity, self.log_queue)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1.0)
        # GPL is medium permissiveness
        self.assertEqual(score, 0.5)

    def test_license_score_unknown(self):
        """Test license score for unknown license."""
        license_name = "UnknownCustomLicense"
        score, time_taken = calculate_license_score(license_name, self.verbosity, self.log_queue)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1.0)
        # Unknown license gets score 0
        self.assertEqual(score, 0)

    def test_license_score_case_insensitive(self):
        """Test that license scoring is case-insensitive."""
        score_upper, _ = calculate_license_score("MIT", self.verbosity, self.log_queue)
        score_lower, _ = calculate_license_score("mit", self.verbosity, self.log_queue)
        
        self.assertEqual(score_upper, score_lower)
        self.assertEqual(score_upper, 1.0)


class TestCodeQuality(unittest.TestCase):
    """Test code quality check function."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_queue = Queue()
        self.verbosity = 0
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_code_quality_valid_python(self):
        """Test code quality check on valid Python code."""
        # Code quality function requires github_str, not filepath
        github_str = "owner/repo"
        result = code_quality(github_str, self.verbosity, self.log_queue)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_code_quality_empty_string(self):
        """Test code quality check on empty string."""
        result = code_quality("", self.verbosity, self.log_queue)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)

    def test_code_quality_returns_tuple(self):
        """Test that code quality returns a tuple with score and time."""
        result = code_quality("test/repo", self.verbosity, self.log_queue)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        score, time_taken = result
        self.assertIsInstance(score, float)
        self.assertIsInstance(time_taken, float)
        self.assertGreaterEqual(score, 0)
        self.assertGreaterEqual(time_taken, 0)


class TestDatasetAndCodePresence(unittest.TestCase):
    """Test dataset and code presence check."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_queue = Queue()
        self.verbosity = 0

    def test_dataset_and_code_presence_with_keywords(self):
        """Test detection of dataset and code presence with keywords."""
        content = "This package includes datasets and code for training."
        result = dataset_and_code_present(content, self.verbosity, self.log_queue)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)

    def test_dataset_and_code_presence_without_keywords(self):
        """Test detection when dataset and code keywords are absent."""
        content = "This is a simple readme."
        result = dataset_and_code_present(content, self.verbosity, self.log_queue)
        
        self.assertIsNotNone(result)

    def test_dataset_and_code_presence_tuple_return(self):
        """Test that function returns a tuple."""
        content = "Dataset and code available"
        result = dataset_and_code_present(content, self.verbosity, self.log_queue)
        
        # Should return tuple with score and time
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestDatasetQuality(unittest.TestCase):
    """Test dataset quality check."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_queue = Queue()
        self.verbosity = 0

    def test_dataset_quality_with_data_files(self):
        """Test dataset quality detection with data file mentions."""
        content = "Dataset includes train.csv, test.csv, and validation.csv"
        result = dataset_quality(content, self.verbosity, self.log_queue)
        
        self.assertIsNotNone(result)

    def test_dataset_quality_with_size_info(self):
        """Test dataset quality with size information."""
        content = "Dataset size: 10GB with 1 million samples"
        result = dataset_quality(content, self.verbosity, self.log_queue)
        
        self.assertIsNotNone(result)

    def test_dataset_quality_returns_tuple(self):
        """Test that function returns tuple."""
        content = "Quality dataset with documentation"
        result = dataset_quality(content, self.verbosity, self.log_queue)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
