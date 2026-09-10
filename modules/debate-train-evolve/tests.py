#!/usr/bin/env python3
"""
Simple validation tests for DTE Framework.

This provides basic functionality tests to ensure the codebase works correctly.
Run with: python tests.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all core modules import correctly."""
    print("Testing imports...")
    try:
        from dte.core.config import DTEConfig
        from dte.core.logger import DTELogger
        from dte.core.pipeline import DTEPipeline
        from dte.data.dataset_manager import DatasetManager
        from dte.debate.manager import DebateManager
        from dte.utils.helpers import format_time, DTEError
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    try:
        from dte.core.config import DTEConfig

        config = DTEConfig()
        errors = config.validate()

        if len(errors) == 0:
            print("✅ Configuration validation passed")
            return True
        else:
            print(f"❌ Configuration validation failed: {errors}")
            return False
    except Exception as e:
        print(f"❌ Configuration validation error: {e}")
        return False

def test_dataset_manager():
    """Test dataset manager basic functionality."""
    print("Testing dataset manager...")
    try:
        from dte.data.dataset_manager import DatasetManager

        manager = DatasetManager()
        datasets = manager.list_supported_datasets()

        if len(datasets) > 0 and "gsm8k" in datasets:
            print(f"✅ Dataset manager working ({len(datasets)} datasets supported)")
            return True
        else:
            print("❌ Dataset manager not working properly")
            return False
    except Exception as e:
        print(f"❌ Dataset manager error: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("Testing utility functions...")
    try:
        from dte.utils.helpers import format_time, calculate_model_size
        from dte.utils.answer_extraction import clean_numeric_string, extract_final_answer

        # Test time formatting
        time_str = format_time(65.5)
        assert "1.1m" in time_str or "65.5s" in time_str

        # Test model size calculation
        size = calculate_model_size("Qwen2.5-1.5B-Instruct")
        assert size == 1.5

        # Test answer extraction
        answer = extract_final_answer("The answer is 42")
        assert answer == "42"

        # Test numeric cleaning
        num = clean_numeric_string("42.5")
        assert num == 42.5

        print("✅ Utility functions working")
        return True
    except Exception as e:
        print(f"❌ Utility functions error: {e}")
        return False

def test_cli_basic():
    """Test basic CLI functionality."""
    print("Testing CLI functionality...")
    try:
        import subprocess
        import os

        # Test help command
        result = subprocess.run([
            sys.executable, "main.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0 and "DTE Framework" in result.stdout:
            print("✅ CLI help working")
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running DTE Framework Validation Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_config_validation,
        test_dataset_manager,
        test_utilities,
        test_cli_basic
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"🏆 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! DTE Framework is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())