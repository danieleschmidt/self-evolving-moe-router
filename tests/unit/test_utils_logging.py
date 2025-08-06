"""Unit tests for logging utilities."""

import pytest
import logging
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from contextlib import contextmanager

from src.self_evolving_moe.utils.logging import (
    ColoredFormatter,
    JSONFormatter,
    EvolutionLogger,
    setup_logging,
    get_logger,
    get_evolution_logger,
    log_execution_time,
    log_memory_usage,
    LogCapture
)


class TestColoredFormatter:
    """Test ColoredFormatter functionality."""

    @pytest.mark.unit
    def test_colored_formatter_basic(self):
        """Test basic colored formatting."""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "Test message" in formatted
        # Colors should be applied to level name
        assert "\033[92mINFO\033[0m" in formatted

    @pytest.mark.unit
    def test_colored_formatter_all_levels(self):
        """Test colored formatting for all log levels."""
        formatter = ColoredFormatter('%(levelname)s')
        
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL")
        ]
        
        for level_num, level_name in levels:
            record = logging.LogRecord(
                name="test", level=level_num, pathname="test.py",
                lineno=1, msg="Test", args=(), exc_info=None
            )
            
            formatted = formatter.format(record)
            assert level_name in formatted
            assert "\033[" in formatted  # Contains color codes


class TestJSONFormatter:
    """Test JSONFormatter functionality."""

    @pytest.mark.unit
    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/path/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_func",
            module="test_module"
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_func"
        assert log_data["line"] == 42
        assert "timestamp" in log_data

    @pytest.mark.unit
    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=1, msg="Error occurred", args=(), exc_info=exc_info
            )
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            assert "exception" in log_data
            assert "ValueError" in log_data["exception"]
            assert "Test exception" in log_data["exception"]

    @pytest.mark.unit
    def test_json_formatter_extra_fields(self):
        """Test JSON formatter with extra fields."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Test", args=(), exc_info=None
        )
        
        # Add extra fields
        record.user_id = "123"
        record.request_id = "abc-def"
        record.generation = 42
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["user_id"] == "123"
        assert log_data["request_id"] == "abc-def"
        assert log_data["generation"] == 42


class TestEvolutionLogger:
    """Test EvolutionLogger functionality."""

    @pytest.mark.unit
    def test_evolution_logger_creation(self):
        """Test EvolutionLogger creation."""
        logger = EvolutionLogger("test_evolution")
        
        assert logger.logger.name == "test_evolution"
        assert logger.generation_times == []
        assert logger.fitness_history == []
        assert logger.best_fitness == float('-inf')

    @pytest.mark.unit
    def test_log_generation_cycle(self, capture_logs):
        """Test complete generation logging cycle."""
        logger = EvolutionLogger("test_evolution")
        
        # Start generation
        logger.log_generation_start(1, 50)
        
        # Wait a bit to simulate processing time
        import time
        time.sleep(0.01)
        
        # End generation
        logger.log_generation_end(1, 0.85, 0.75, 0.9)
        
        logs = capture_logs.getvalue()
        assert "Generation 1 started" in logs
        assert "Generation 1 complete" in logs

    @pytest.mark.unit
    def test_best_fitness_tracking(self):
        """Test best fitness tracking across generations."""
        logger = EvolutionLogger("test_evolution")
        
        logger.log_generation_start(1, 10)
        logger.log_generation_end(1, 0.7, 0.65, 0.8)
        
        logger.log_generation_start(2, 10)
        logger.log_generation_end(2, 0.9, 0.85, 0.75)  # Better fitness
        
        logger.log_generation_start(3, 10)
        logger.log_generation_end(3, 0.85, 0.8, 0.9)  # Worse fitness
        
        assert logger.best_fitness == 0.9
        assert len(logger.fitness_history) == 3
        assert len(logger.generation_times) == 3

    @pytest.mark.unit
    def test_topology_mutation_logging(self, capture_logs):
        """Test topology mutation logging."""
        logger = EvolutionLogger("test_evolution")
        
        logger.log_topology_mutation("topo_001", "swap_experts", True)
        logger.log_topology_mutation("topo_002", "add_connection", False)
        
        logs = capture_logs.getvalue()
        assert "Mutation swap_experts on topology topo_001: ✅ success" in logs
        assert "Mutation add_connection on topology topo_002: ❌ failed" in logs

    @pytest.mark.unit
    def test_expert_selection_logging(self, capture_logs):
        """Test expert selection logging."""
        logger = EvolutionLogger("test_evolution")
        
        expert_ids = [1, 5, 12]
        weights = [0.4, 0.3, 0.3]
        
        logger.log_expert_selection(expert_ids, weights)
        
        logs = capture_logs.getvalue()
        assert "Selected experts: [1, 5, 12]" in logs

    @pytest.mark.unit
    def test_performance_metrics_logging(self, capture_logs):
        """Test performance metrics logging."""
        logger = EvolutionLogger("test_evolution")
        
        metrics = {
            "accuracy": 0.87,
            "latency": 15.2,
            "memory_usage": 2.1
        }
        
        logger.log_performance_metrics(metrics)
        
        logs = capture_logs.getvalue()
        assert "Performance metrics" in logs

    @pytest.mark.unit 
    def test_resource_usage_logging(self, capture_logs):
        """Test resource usage logging."""
        logger = EvolutionLogger("test_evolution")
        
        logger.log_resource_usage(1024.5, 512.3)
        logger.log_resource_usage(2048.0)  # Without GPU memory
        
        logs = capture_logs.getvalue()
        assert "Memory: 1024.5MB" in logs
        assert "GPU: 512.3MB" in logs
        assert "Memory: 2048.0MB" in logs

    @pytest.mark.unit
    def test_evolution_summary(self):
        """Test evolution summary generation."""
        logger = EvolutionLogger("test_evolution")
        
        # Simulate several generations
        for gen in range(1, 6):
            logger.log_generation_start(gen, 20)
            import time
            time.sleep(0.001)  # Small delay
            fitness = 0.5 + (gen * 0.05)  # Improving fitness
            logger.log_generation_end(gen, fitness, fitness - 0.1, 0.8)
        
        summary = logger.get_evolution_summary()
        
        assert summary["generations_completed"] == 5
        assert summary["best_fitness_achieved"] == 0.75
        assert summary["fitness_improvement"] > 0
        assert summary["avg_generation_time"] > 0
        assert 0 <= summary["convergence_rate"] <= 1


class TestSetupLogging:
    """Test setup_logging function."""

    @pytest.mark.unit
    def test_setup_basic_logging(self):
        """Test basic logging setup."""
        logger = setup_logging(level="DEBUG")
        
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1  # At least console handler

    @pytest.mark.unit
    def test_setup_file_logging(self, temp_dir):
        """Test logging setup with file output."""
        log_file = temp_dir / "test.log"
        
        logger = setup_logging(level="INFO", log_file=str(log_file))
        
        # Test logging
        logger.info("Test message")
        
        # File should exist and contain message
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    @pytest.mark.unit
    def test_setup_json_logging(self, temp_dir):
        """Test logging setup with JSON format."""
        log_file = temp_dir / "test.json"
        
        logger = setup_logging(
            level="INFO", 
            log_file=str(log_file),
            json_format=True
        )
        
        logger.info("Test JSON message")
        
        # Read and parse JSON log
        content = log_file.read_text().strip()
        log_data = json.loads(content)
        
        assert log_data["message"] == "Test JSON message"
        assert log_data["level"] == "INFO"

    @pytest.mark.unit
    def test_setup_log_dir(self, temp_dir):
        """Test logging setup with log directory."""
        log_dir = temp_dir / "logs"
        
        logger = setup_logging(level="INFO", log_dir=str(log_dir))
        
        assert log_dir.exists()
        assert log_dir.is_dir()

    @pytest.mark.unit
    def test_invalid_log_level(self):
        """Test handling of invalid log level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="INVALID")


class TestLogExecutionTime:
    """Test log_execution_time context manager."""

    @pytest.mark.unit
    def test_log_execution_time_success(self, capture_logs):
        """Test successful execution time logging."""
        logger = get_logger("test")
        
        with log_execution_time(logger, "test operation"):
            import time
            time.sleep(0.01)
        
        logs = capture_logs.getvalue()
        assert "Starting test operation" in logs
        assert "test operation completed" in logs

    @pytest.mark.unit
    def test_log_execution_time_exception(self, capture_logs):
        """Test execution time logging with exception."""
        logger = get_logger("test")
        
        with pytest.raises(ValueError):
            with log_execution_time(logger, "failing operation"):
                raise ValueError("Test error")
        
        logs = capture_logs.getvalue()
        assert "Starting failing operation" in logs
        assert "failing operation failed" in logs
        assert "Test error" in logs


class TestLogMemoryUsage:
    """Test log_memory_usage context manager."""

    @pytest.mark.unit
    def test_log_memory_usage_success(self, capture_logs):
        """Test memory usage logging."""
        logger = get_logger("test")
        
        with log_memory_usage(logger, "memory test"):
            # Allocate some memory
            data = [0] * 1000
        
        logs = capture_logs.getvalue()
        if "psutil not available" not in logs:
            assert "Memory before memory test" in logs
            assert "Memory after memory test" in logs

    @pytest.mark.unit
    def test_log_memory_usage_without_psutil(self, capture_logs):
        """Test memory usage logging without psutil."""
        logger = get_logger("test")
        
        with patch('src.self_evolving_moe.utils.logging.psutil', None):
            with log_memory_usage(logger, "memory test"):
                pass
        
        # Should work without errors even without psutil


class TestLogCapture:
    """Test LogCapture utility."""

    @pytest.mark.unit
    def test_log_capture_basic(self):
        """Test basic log capture functionality."""
        with LogCapture("test_logger") as capture:
            logger = logging.getLogger("test_logger")
            logger.info("Captured message")
        
        messages = capture.get_messages()
        assert any("Captured message" in msg for msg in messages)

    @pytest.mark.unit
    def test_log_capture_filter_level(self):
        """Test log capture with level filtering."""
        with LogCapture("test_logger") as capture:
            logger = logging.getLogger("test_logger")
            logger.debug("Debug message")
            logger.info("Info message")
            logger.error("Error message")
        
        error_messages = capture.get_messages("ERROR")
        assert any("Error message" in msg for msg in error_messages)
        assert not any("Info message" in msg for msg in error_messages)

    @pytest.mark.unit
    def test_log_capture_clear(self):
        """Test log capture clear functionality."""
        with LogCapture("test_logger") as capture:
            logger = logging.getLogger("test_logger")
            logger.info("First message")
            
            assert len(capture.get_messages()) > 0
            
            capture.clear()
            assert len(capture.get_messages()) == 0
            
            logger.info("Second message")
            messages = capture.get_messages()
            assert any("Second message" in msg for msg in messages)
            assert not any("First message" in msg for msg in messages)