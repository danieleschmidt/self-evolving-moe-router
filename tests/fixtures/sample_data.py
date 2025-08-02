"""Sample data fixtures for testing."""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json


class SampleDataGenerator:
    """Generate sample data for testing purposes."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def generate_text_classification_data(
        self, 
        num_samples: int, 
        vocab_size: int, 
        sequence_length: int,
        num_classes: int,
        device: torch.device = torch.device("cpu")
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic text classification dataset."""
        dataset = []
        
        for _ in range(num_samples):
            # Random input sequence
            input_ids = torch.randint(
                0, vocab_size, (sequence_length,), device=device
            )
            
            # Attention mask (all ones for simplicity)
            attention_mask = torch.ones(sequence_length, device=device)
            
            # Random label
            label = torch.randint(0, num_classes, (1,), device=device)
            
            sample = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label
            }
            dataset.append(sample)
        
        return dataset
    
    def generate_language_modeling_data(
        self,
        num_samples: int,
        vocab_size: int,
        sequence_length: int,
        device: torch.device = torch.device("cpu")
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic language modeling dataset."""
        dataset = []
        
        for _ in range(num_samples):
            # Input sequence
            input_ids = torch.randint(
                0, vocab_size, (sequence_length,), device=device
            )
            
            # Labels are shifted input_ids
            labels = torch.cat([
                input_ids[1:],
                torch.randint(0, vocab_size, (1,), device=device)
            ])
            
            # Attention mask
            attention_mask = torch.ones(sequence_length, device=device)
            
            sample = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            dataset.append(sample)
        
        return dataset
    
    def generate_routing_topologies(
        self,
        num_topologies: int,
        num_tokens: int,
        num_experts: int,
        sparsity_range: Tuple[float, float] = (0.7, 0.95),
        device: torch.device = torch.device("cpu")
    ) -> List[Dict[str, Any]]:
        """Generate sample routing topologies."""
        topologies = []
        
        for _ in range(num_topologies):
            # Random sparsity level
            sparsity = np.random.uniform(*sparsity_range)
            
            # Create sparse routing matrix
            routing_matrix = torch.zeros((num_tokens, num_experts), device=device)
            
            # Calculate number of connections
            total_connections = int(routing_matrix.numel() * (1 - sparsity))
            
            # Randomly place connections
            flat_indices = torch.randperm(routing_matrix.numel())[:total_connections]
            routing_matrix.view(-1)[flat_indices] = torch.rand(total_connections)
            
            # Normalize each row to sum to 1 (if any connections exist)
            row_sums = routing_matrix.sum(dim=1, keepdim=True)
            routing_matrix = torch.where(
                row_sums > 0,
                routing_matrix / row_sums,
                routing_matrix
            )
            
            # Random routing parameters
            routing_params = {
                "temperature": np.random.uniform(0.5, 2.0),
                "top_k": np.random.choice([1, 2, 3, 4]),
                "load_balancing_weight": np.random.uniform(0.001, 0.1),
                "diversity_weight": np.random.uniform(0.01, 0.2)
            }
            
            topology = {
                "routing_matrix": routing_matrix,
                "routing_params": routing_params,
                "sparsity": sparsity,
                "num_tokens": num_tokens,
                "num_experts": num_experts
            }
            
            topologies.append(topology)
        
        return topologies
    
    def generate_expert_outputs(
        self,
        batch_size: int,
        num_experts: int,
        expert_dim: int,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Generate sample expert outputs."""
        return torch.randn(
            (batch_size, num_experts, expert_dim), 
            device=device
        )
    
    def generate_fitness_scores(
        self,
        population_size: int,
        num_objectives: int = 1,
        score_range: Tuple[float, float] = (0.0, 1.0)
    ) -> np.ndarray:
        """Generate sample fitness scores."""
        if num_objectives == 1:
            return np.random.uniform(*score_range, population_size)
        else:
            return np.random.uniform(
                *score_range, 
                (population_size, num_objectives)
            )
    
    def generate_evolution_history(
        self,
        num_generations: int,
        population_size: int,
        convergence_trend: str = "improving"  # "improving", "declining", "stable"
    ) -> Dict[str, List]:
        """Generate sample evolution history."""
        history = {
            "generations": list(range(num_generations)),
            "best_fitness": [],
            "average_fitness": [],
            "worst_fitness": [],
            "diversity": [],
            "population_size": [population_size] * num_generations
        }
        
        # Starting fitness
        current_best = 0.3
        current_avg = 0.2
        current_worst = 0.1
        
        for gen in range(num_generations):
            # Apply trend
            if convergence_trend == "improving":
                improvement = np.random.uniform(0.0, 0.02)
                current_best = min(0.98, current_best + improvement)
                current_avg = min(current_best - 0.05, current_avg + improvement * 0.7)
                current_worst = min(current_avg - 0.05, current_worst + improvement * 0.5)
            elif convergence_trend == "declining":
                decline = np.random.uniform(0.0, 0.01)
                current_best = max(0.1, current_best - decline)
                current_avg = max(0.05, current_avg - decline)
                current_worst = max(0.01, current_worst - decline)
            else:  # stable
                noise = np.random.uniform(-0.01, 0.01)
                current_best += noise
                current_avg += noise * 0.7
                current_worst += noise * 0.5
            
            # Add some randomness
            best_fitness = current_best + np.random.normal(0, 0.01)
            avg_fitness = current_avg + np.random.normal(0, 0.02)
            worst_fitness = current_worst + np.random.normal(0, 0.01)
            
            # Ensure ordering
            best_fitness = max(best_fitness, avg_fitness + 0.01)
            worst_fitness = min(worst_fitness, avg_fitness - 0.01)
            
            # Diversity typically decreases over time
            diversity = max(0.1, 0.9 - (gen / num_generations) * 0.6 + np.random.uniform(-0.1, 0.1))
            
            history["best_fitness"].append(best_fitness)
            history["average_fitness"].append(avg_fitness)
            history["worst_fitness"].append(worst_fitness)
            history["diversity"].append(diversity)
        
        return history
    
    def generate_benchmark_results(
        self,
        model_names: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Generate sample benchmark results."""
        if metrics is None:
            metrics = ["accuracy", "latency", "memory", "throughput"]
        
        results = {}
        
        for model_name in model_names:
            model_results = {}
            
            for metric in metrics:
                if metric == "accuracy":
                    # Higher is better
                    if "evolved" in model_name.lower():
                        value = np.random.uniform(0.85, 0.95)
                    elif "dense" in model_name.lower():
                        value = np.random.uniform(0.80, 0.90)
                    else:
                        value = np.random.uniform(0.75, 0.88)
                
                elif metric == "latency":
                    # Lower is better (milliseconds)
                    if "evolved" in model_name.lower():
                        value = np.random.uniform(8.0, 20.0)
                    elif "dense" in model_name.lower():
                        value = np.random.uniform(30.0, 60.0)
                    else:
                        value = np.random.uniform(15.0, 40.0)
                
                elif metric == "memory":
                    # Lower is better (GB)
                    if "evolved" in model_name.lower():
                        value = np.random.uniform(1.5, 4.0)
                    elif "dense" in model_name.lower():
                        value = np.random.uniform(6.0, 12.0)
                    else:
                        value = np.random.uniform(3.0, 8.0)
                
                elif metric == "throughput":
                    # Higher is better (samples/sec)
                    if "evolved" in model_name.lower():
                        value = np.random.uniform(40.0, 80.0)
                    elif "dense" in model_name.lower():
                        value = np.random.uniform(15.0, 30.0)
                    else:
                        value = np.random.uniform(25.0, 50.0)
                
                else:
                    # Generic metric
                    value = np.random.uniform(0.0, 1.0)
                
                model_results[metric] = round(value, 3)
            
            results[model_name] = model_results
        
        return results
    
    def save_sample_data(self, data: Any, filepath: Path):
        """Save sample data to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.json':
            # Convert tensors to lists for JSON serialization
            serializable_data = self._make_json_serializable(data)
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        elif filepath.suffix in ['.pt', '.pth']:
            torch.save(data, filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def load_sample_data(self, filepath: Path, device: torch.device = torch.device("cpu")):
        """Load sample data from file."""
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return self._restore_tensors(data, device)
        elif filepath.suffix in ['.pt', '.pth']:
            return torch.load(filepath, map_location=device)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _make_json_serializable(self, obj):
        """Convert PyTorch tensors to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return {
                "_type": "tensor",
                "data": obj.cpu().numpy().tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return {
                "_type": "numpy",
                "data": obj.tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype)
            }
        else:
            return obj
    
    def _restore_tensors(self, obj, device: torch.device):
        """Restore PyTorch tensors from JSON-serializable format."""
        if isinstance(obj, dict):
            if obj.get("_type") == "tensor":
                data = torch.tensor(obj["data"], device=device)
                return data.view(obj["shape"])
            elif obj.get("_type") == "numpy":
                return np.array(obj["data"]).reshape(obj["shape"])
            else:
                return {k: self._restore_tensors(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_tensors(item, device) for item in obj]
        else:
            return obj


# Convenience functions for common use cases
def create_sample_dataset(
    dataset_type: str = "classification",
    num_samples: int = 100,
    vocab_size: int = 1000,
    sequence_length: int = 64,
    num_classes: int = 2,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
):
    """Create a sample dataset for testing."""
    generator = SampleDataGenerator(seed=seed)
    
    if dataset_type == "classification":
        return generator.generate_text_classification_data(
            num_samples, vocab_size, sequence_length, num_classes, device
        )
    elif dataset_type == "language_modeling":
        return generator.generate_language_modeling_data(
            num_samples, vocab_size, sequence_length, device
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_sample_population(
    population_size: int = 20,
    num_tokens: int = 64,
    num_experts: int = 8,
    sparsity_range: Tuple[float, float] = (0.8, 0.95),
    device: torch.device = torch.device("cpu"),
    seed: int = 42
):
    """Create a sample population for evolution testing."""
    generator = SampleDataGenerator(seed=seed)
    return generator.generate_routing_topologies(
        population_size, num_tokens, num_experts, sparsity_range, device
    )


def create_evolution_history(
    num_generations: int = 100,
    population_size: int = 50,
    trend: str = "improving",
    seed: int = 42
):
    """Create sample evolution history."""
    generator = SampleDataGenerator(seed=seed)
    return generator.generate_evolution_history(
        num_generations, population_size, trend
    )