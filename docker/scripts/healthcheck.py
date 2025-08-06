#!/usr/bin/env python3
"""
Health check script for Self-Evolving MoE-Router production deployment.
Validates system health, dependencies, and service availability.
"""

import sys
import os
import argparse
import time
import json
import subprocess
import socket
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add src to path
sys.path.insert(0, '/app/src')

# Health check results
class HealthStatus:
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Comprehensive health check for MoE-Router system."""
    
    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results: Dict[str, Any] = {}
        self.overall_status = HealthStatus.HEALTHY
        
    def check_python_environment(self) -> Tuple[str, Dict[str, Any]]:
        """Check Python environment and dependencies."""
        try:
            import torch
            import numpy as np
            
            # Try importing main package
            import self_evolving_moe
            from self_evolving_moe import EvolvingMoERouter, ExpertPool
            
            details = {
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__,
                'package_version': getattr(self_evolving_moe, '__version__', 'unknown'),
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            return HealthStatus.HEALTHY, details
            
        except ImportError as e:
            return HealthStatus.UNHEALTHY, {'error': f'Import failed: {e}'}
        except Exception as e:
            return HealthStatus.DEGRADED, {'error': f'Environment check failed: {e}'}
    
    def check_system_resources(self) -> Tuple[str, Dict[str, Any]]:
        """Check system resource availability."""
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            details = {
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent_used': memory.percent,
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'disk_percent_used': round((disk.used / disk.total) * 100, 1),
                'cpu_percent': cpu_percent,
                'process_memory_mb': round(process_memory.rss / (1024**2), 1),
                'process_cpu_percent': process.cpu_percent()
            }
            
            # Determine status based on resource usage
            status = HealthStatus.HEALTHY
            
            if memory.percent > 90 or disk.percent > 95 or cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
            elif memory.percent > 80 or disk.percent > 85 or cpu_percent > 85:
                status = HealthStatus.DEGRADED
            
            return status, details
            
        except Exception as e:
            return HealthStatus.DEGRADED, {'error': f'Resource check failed: {e}'}
    
    def check_file_system(self) -> Tuple[str, Dict[str, Any]]:
        """Check file system and required directories."""
        required_dirs = ['/app/logs', '/app/checkpoints', '/app/data', '/app/config']
        required_files = ['/app/src/self_evolving_moe/__init__.py']
        
        details = {
            'directories': {},
            'files': {},
            'permissions': {}
        }
        
        status = HealthStatus.HEALTHY
        
        # Check directories
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                details['directories'][dir_path] = {
                    'exists': True,
                    'writable': os.access(path, os.W_OK),
                    'readable': os.access(path, os.R_OK)
                }
                
                if not os.access(path, os.W_OK):
                    status = HealthStatus.DEGRADED
            else:
                details['directories'][dir_path] = {'exists': False}
                status = HealthStatus.DEGRADED
        
        # Check files
        for file_path in required_files:
            path = Path(file_path)
            details['files'][file_path] = {
                'exists': path.exists(),
                'readable': os.access(path, os.R_OK) if path.exists() else False
            }
            
            if not path.exists():
                status = HealthStatus.UNHEALTHY
        
        return status, details
    
    def check_network_connectivity(self) -> Tuple[str, Dict[str, Any]]:
        """Check network connectivity and port availability."""
        if self.quick_mode:
            return HealthStatus.HEALTHY, {'skipped': 'Quick mode enabled'}
        
        ports_to_check = [8000, 9090]  # API and metrics ports
        details = {'ports': {}, 'connectivity': {}}
        status = HealthStatus.HEALTHY
        
        # Check if ports are available
        for port in ports_to_check:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', port))
                    details['ports'][port] = {
                        'listening': result == 0,
                        'available': result != 0
                    }
            except Exception as e:
                details['ports'][port] = {'error': str(e)}
                status = HealthStatus.DEGRADED
        
        # Test basic HTTP connectivity
        try:
            response = urllib.request.urlopen('http://localhost:8000/health', timeout=5)
            details['connectivity']['api_health'] = {
                'status_code': response.getcode(),
                'accessible': True
            }
        except Exception as e:
            details['connectivity']['api_health'] = {
                'accessible': False,
                'error': str(e)
            }
            if not self.quick_mode:
                status = HealthStatus.DEGRADED
        
        return status, details
    
    def check_model_functionality(self) -> Tuple[str, Dict[str, Any]]:
        """Check basic model functionality."""
        if self.quick_mode:
            return HealthStatus.HEALTHY, {'skipped': 'Quick mode enabled'}
        
        try:
            from self_evolving_moe import ExpertPool, EvolvingMoERouter, EvolutionConfig
            import torch
            
            # Create small test components
            expert_pool = ExpertPool(
                num_experts=2,
                expert_dim=8,
                expert_type="mlp",
                device="cpu"
            )
            
            config = EvolutionConfig(
                population_size=3,
                generations=1,
                mutation_rate=0.1
            )
            
            # Test basic initialization
            evolver = EvolvingMoERouter(
                expert_pool=expert_pool,
                config=config,
                device="cpu"
            )
            
            # Test topology operations
            topology = evolver.population[0]
            sparsity = topology.compute_sparsity()
            
            # Test basic forward pass
            test_input = torch.randn(2, 8)
            routing_weights, selected_experts = topology.get_routing_weights(test_input.unsqueeze(1))
            
            details = {
                'expert_pool_created': True,
                'evolver_initialized': True,
                'population_size': len(evolver.population),
                'topology_sparsity': float(sparsity),
                'routing_test_passed': True,
                'routing_weights_shape': list(routing_weights.shape),
                'selected_experts_shape': list(selected_experts.shape)
            }
            
            return HealthStatus.HEALTHY, details
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, {
                'error': f'Model functionality test failed: {e}',
                'test_passed': False
            }
    
    def check_gpu_availability(self) -> Tuple[str, Dict[str, Any]]:
        """Check GPU availability and CUDA setup."""
        try:
            import torch
            
            details = {
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
            
            if torch.cuda.is_available():
                details['gpu_devices'] = []
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    details['gpu_devices'].append({
                        'id': i,
                        'name': gpu_props.name,
                        'total_memory_gb': round(gpu_props.total_memory / (1024**3), 2),
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                    })
                
                # Test basic GPU operation
                try:
                    test_tensor = torch.randn(10, 10).cuda()
                    result = torch.matmul(test_tensor, test_tensor.t())
                    details['gpu_test_passed'] = True
                except Exception as e:
                    details['gpu_test_passed'] = False
                    details['gpu_test_error'] = str(e)
                    return HealthStatus.DEGRADED, details
            
            return HealthStatus.HEALTHY, details
            
        except Exception as e:
            return HealthStatus.DEGRADED, {'error': f'GPU check failed: {e}'}
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive report."""
        checks = [
            ('python_environment', self.check_python_environment),
            ('system_resources', self.check_system_resources),
            ('file_system', self.check_file_system),
            ('network_connectivity', self.check_network_connectivity),
            ('model_functionality', self.check_model_functionality),
            ('gpu_availability', self.check_gpu_availability)
        ]
        
        overall_status = HealthStatus.HEALTHY
        
        for check_name, check_func in checks:
            try:
                status, details = check_func()
                self.results[check_name] = {
                    'status': status,
                    'details': details
                }
                
                # Update overall status
                if status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                self.results[check_name] = {
                    'status': HealthStatus.UNHEALTHY,
                    'details': {'error': f'Check failed: {e}'}
                }
                overall_status = HealthStatus.UNHEALTHY
        
        self.overall_status = overall_status
        
        return {
            'overall_status': overall_status,
            'timestamp': time.time(),
            'checks': self.results,
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of health check results."""
        healthy_checks = sum(1 for check in self.results.values() if check['status'] == HealthStatus.HEALTHY)
        degraded_checks = sum(1 for check in self.results.values() if check['status'] == HealthStatus.DEGRADED)
        unhealthy_checks = sum(1 for check in self.results.values() if check['status'] == HealthStatus.UNHEALTHY)
        
        return {
            'total_checks': len(self.results),
            'healthy_checks': healthy_checks,
            'degraded_checks': degraded_checks,
            'unhealthy_checks': unhealthy_checks,
            'health_percentage': round((healthy_checks / len(self.results)) * 100, 1) if self.results else 0
        }


def main():
    """Main health check entry point."""
    parser = argparse.ArgumentParser(description='Self-Evolving MoE-Router Health Check')
    parser.add_argument('--quick', action='store_true', help='Run quick health check')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run health checks
    health_checker = HealthCheck(quick_mode=args.quick)
    results = health_checker.run_all_checks()
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        # Human-readable output
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.DEGRADED: "⚠️",
            HealthStatus.UNHEALTHY: "❌"
        }
        
        print(f"\n{status_emoji[results['overall_status']]} Overall Status: {results['overall_status'].upper()}")
        print(f"Health Score: {results['summary']['health_percentage']}%")
        print(f"Checks: {results['summary']['healthy_checks']} healthy, "
              f"{results['summary']['degraded_checks']} degraded, "
              f"{results['summary']['unhealthy_checks']} unhealthy")
        
        if args.verbose:
            print("\nDetailed Results:")
            for check_name, check_result in results['checks'].items():
                status = check_result['status']
                print(f"\n{status_emoji[status]} {check_name}: {status}")
                
                if 'error' in check_result['details']:
                    print(f"   Error: {check_result['details']['error']}")
                elif check_result['details']:
                    for key, value in check_result['details'].items():
                        if isinstance(value, dict):
                            print(f"   {key}:")
                            for sub_key, sub_value in value.items():
                                print(f"     {sub_key}: {sub_value}")
                        else:
                            print(f"   {key}: {value}")
    
    # Exit with appropriate code
    if results['overall_status'] == HealthStatus.HEALTHY:
        sys.exit(0)
    elif results['overall_status'] == HealthStatus.DEGRADED:
        sys.exit(1)
    else:  # UNHEALTHY
        sys.exit(2)


if __name__ == "__main__":
    main()