"""
Logging utilities for Self-Evolving MoE-Router.

This module provides structured logging configuration and utilities
for better debugging and monitoring of evolution processes.
"""

import logging
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, IO
from contextlib import contextmanager

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED + Colors.BOLD,
    }
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{Colors.END}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class EvolutionLogger:
    """Specialized logger for evolution processes."""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.log_dir = log_dir
        self.start_time = time.time()
        
        # Evolution-specific metrics
        self.generation_times = []
        self.fitness_history = []
        self.best_fitness = float('-inf')
        
    def log_generation_start(self, generation: int, population_size: int):
        """Log start of evolution generation."""
        self.generation_start_time = time.time()
        self.logger.info(
            f"🧬 Generation {generation} started",
            extra={
                'event_type': 'generation_start',
                'generation': generation,
                'population_size': population_size
            }
        )
    
    def log_generation_end(self, generation: int, best_fitness: float, avg_fitness: float, diversity: float):
        """Log end of evolution generation."""
        generation_time = time.time() - self.generation_start_time
        self.generation_times.append(generation_time)
        self.fitness_history.append(best_fitness)
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            improvement = "🎯 NEW BEST"
        else:
            improvement = "📈 continuing"
        
        self.logger.info(
            f"✅ Generation {generation} complete ({generation_time:.2f}s) - {improvement}",
            extra={
                'event_type': 'generation_end',
                'generation': generation,
                'generation_time': generation_time,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity,
                'is_new_best': best_fitness > self.best_fitness
            }
        )
    
    def log_topology_mutation(self, topology_id: str, mutation_type: str, success: bool):
        """Log topology mutation events."""
        status = "✅ success" if success else "❌ failed"
        self.logger.debug(
            f"🔬 Mutation {mutation_type} on topology {topology_id}: {status}",
            extra={
                'event_type': 'mutation',
                'topology_id': topology_id,
                'mutation_type': mutation_type,
                'success': success
            }
        )
    
    def log_expert_selection(self, expert_ids: list, weights: list):
        """Log expert selection events."""
        self.logger.debug(
            f"🎯 Selected experts: {expert_ids}",
            extra={
                'event_type': 'expert_selection',
                'expert_ids': expert_ids,
                'weights': weights
            }
        )
    
    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.logger.info(
            f"📊 Performance metrics: {metrics}",
            extra={
                'event_type': 'performance_metrics',
                **metrics
            }
        )
    
    def log_resource_usage(self, memory_mb: float, gpu_memory_mb: float = None):
        """Log resource usage."""
        resource_info = f"💾 Memory: {memory_mb:.1f}MB"
        if gpu_memory_mb is not None:
            resource_info += f", GPU: {gpu_memory_mb:.1f}MB"
        
        self.logger.debug(
            resource_info,
            extra={
                'event_type': 'resource_usage', 
                'memory_mb': memory_mb,
                'gpu_memory_mb': gpu_memory_mb
            }
        )
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution run."""
        total_time = time.time() - self.start_time
        
        return {
            'total_time': total_time,
            'generations_completed': len(self.generation_times),
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
            'best_fitness_achieved': self.best_fitness,
            'fitness_improvement': self.fitness_history[-1] - self.fitness_history[0] if len(self.fitness_history) > 1 else 0,
            'convergence_rate': len([f for f in self.fitness_history if f == self.best_fitness]) / len(self.fitness_history) if self.fitness_history else 0
        }
    
    # Delegate standard logging methods to internal logger
    def debug(self, message, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
        
    def info(self, message, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
        
    def warning(self, message, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
        
    def error(self, message, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
        
    def critical(self, message, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    use_colors: bool = True,
    json_format: bool = False,
    log_dir: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        use_colors: Whether to use colored output for console
        json_format: Whether to use JSON formatting
        log_dir: Directory for log files
        
    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if json_format:
        console_formatter = JSONFormatter()
    elif use_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"moe_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party library log levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for a module."""
    return logging.getLogger(name)


def get_evolution_logger(name: str, log_dir: Optional[Path] = None) -> EvolutionLogger:
    """Get specialized evolution logger."""
    return EvolutionLogger(name, log_dir)


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """Context manager to log execution time of operations."""
    start_time = time.time()
    logger.log(level, f"⏱️  Starting {operation}...")
    
    try:
        yield
        execution_time = time.time() - start_time
        logger.log(level, f"✅ {operation} completed in {execution_time:.2f}s")
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ {operation} failed after {execution_time:.2f}s: {e}")
        raise


@contextmanager
def log_memory_usage(logger: logging.Logger, operation: str):
    """Context manager to log memory usage during operations."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.debug(f"🧠 Memory before {operation}: {start_memory:.1f}MB")
        
        yield
        
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = end_memory - start_memory
        
        if memory_diff > 0:
            logger.debug(f"🧠 Memory after {operation}: {end_memory:.1f}MB (+{memory_diff:.1f}MB)")
        else:
            logger.debug(f"🧠 Memory after {operation}: {end_memory:.1f}MB ({memory_diff:.1f}MB)")
            
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        yield
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {e}")
        yield


class LogCapture:
    """Utility to capture log messages for testing."""
    
    def __init__(self, logger_name: str = None, level: int = logging.DEBUG):
        self.logger_name = logger_name
        self.level = level
        self.records = []
        self.handler = None
    
    def __enter__(self):
        self.handler = logging.StreamHandler(self)
        self.handler.setLevel(self.level)
        
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.root
        
        logger.addHandler(self.handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.root
        
        logger.removeHandler(self.handler)
    
    def write(self, msg):
        """Handle write calls from StreamHandler."""
        if msg.strip():
            self.records.append(msg.strip())
    
    def flush(self):
        """Handle flush calls from StreamHandler."""
        pass
    
    def get_messages(self, level: str = None) -> list:
        """Get captured log messages, optionally filtered by level."""
        if level:
            return [record for record in self.records if level.upper() in record]
        return self.records.copy()
    
    def clear(self):
        """Clear captured messages."""
        self.records.clear()