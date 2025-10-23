"""
Checkpoint Utilities and Performance Monitoring
================================================

Utilities for checkpoint management and training performance monitoring.

Author: GitHub Copilot
Date: 2025-10-23
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict
import torch


class PerformanceMonitor:
    """
    Monitor training performance and bottlenecks.
    
    Tracks timing for:
    - Data collection
    - Forward pass
    - Loss computation
    - Backward pass
    - Optimizer step
    - Checkpoint saving
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> with monitor.timer('data_collection'):
        ...     data = collector.collect()
        >>> stats = monitor.get_stats()
        >>> print(f"Data collection: {stats['data_collection']['mean']:.3f}s")
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
    
    def timer(self, name: str):
        """Context manager for timing code blocks."""
        return _TimerContext(self, name)
    
    def start_timer(self, name: str):
        """Start a timer (manual timing)."""
        self.current_timers[name] = time.time()
    
    def end_timer(self, name: str):
        """End a timer and record elapsed time."""
        if name in self.current_timers:
            elapsed = time.time() - self.current_timers[name]
            self.timings[name].append(elapsed)
            del self.current_timers[name]
            return elapsed
        return None
    
    def record(self, name: str, value: float):
        """Manually record a timing value."""
        self.timings[name].append(value)
    
    def get_stats(self, window: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics.
        
        Args:
            window: Number of recent samples to compute stats over
        
        Returns:
            Dict mapping timer names to {mean, std, min, max, total}
        """
        stats = {}
        for name, values in self.timings.items():
            recent = values[-window:] if len(values) > window else values
            if recent:
                stats[name] = {
                    'mean': sum(recent) / len(recent),
                    'std': torch.tensor(recent).std().item() if len(recent) > 1 else 0.0,
                    'min': min(recent),
                    'max': max(recent),
                    'total': sum(recent),
                    'count': len(recent)
                }
        return stats
    
    def get_summary(self, window: int = 100) -> str:
        """Get formatted summary string."""
        stats = self.get_stats(window)
        lines = ["\n=== Performance Summary ==="]
        
        for name, values in stats.items():
            lines.append(
                f"{name:25s}: {values['mean']*1000:7.2f} ms "
                f"(±{values['std']*1000:5.2f} ms) "
                f"[{values['min']*1000:6.2f}, {values['max']*1000:6.2f}]"
            )
        
        return "\n".join(lines)
    
    def reset(self):
        """Clear all timing data."""
        self.timings.clear()
        self.current_timers.clear()


class _TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.monitor.record(self.name, elapsed)


def format_checkpoint_name(
    base_path: str,
    iteration: int,
    metric_name: Optional[str] = None,
    metric_value: Optional[float] = None
) -> str:
    """
    Format checkpoint filename with iteration and optional metric.
    
    Args:
        base_path: Base directory for checkpoints
        iteration: Training iteration number
        metric_name: Optional metric name (e.g., 'reward')
        metric_value: Optional metric value
    
    Returns:
        Formatted checkpoint path
    
    Example:
        >>> format_checkpoint_name('/tmp/ckpts', 1000, 'reward', 123.45)
        '/tmp/ckpts/checkpoint_iter_1000_reward_123.45.pt'
    """
    import os
    
    filename = f"checkpoint_iter_{iteration}"
    if metric_name and metric_value is not None:
        filename += f"_{metric_name}_{metric_value:.2f}"
    filename += ".pt"
    
    return os.path.join(base_path, filename)


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dict with total_params, trainable_params, size_mb
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in MB (assuming float32)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb
    }


def print_model_info(model: torch.nn.Module, name: str = "Model"):
    """Print model information."""
    info = get_model_size(model)
    print(f"\n=== {name} Info ===")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Estimated size: {info['size_mb']:.2f} MB")


if __name__ == "__main__":
    # Test PerformanceMonitor
    print("Testing PerformanceMonitor...")
    monitor = PerformanceMonitor()
    
    # Simulate training loop
    for i in range(10):
        with monitor.timer('data_collection'):
            time.sleep(0.01)
        
        with monitor.timer('training'):
            time.sleep(0.02)
        
        if i % 5 == 0:
            with monitor.timer('checkpoint_save'):
                time.sleep(0.05)
    
    # Get stats
    print(monitor.get_summary())
    
    # Test checkpoint naming
    print("\nTesting checkpoint naming...")
    name = format_checkpoint_name('/tmp/ckpts', 1000, 'reward', 123.45)
    print(f"Checkpoint name: {name}")
    
    # Test model size
    print("\nTesting model size calculation...")
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    print_model_info(dummy_model, "DummyModel")
    
    print("\n✅ All tests passed!")
