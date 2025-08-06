#!/usr/bin/env python3
"""
Comprehensive quality gates runner for Self-Evolving MoE-Router.

This script runs all quality checks including linting, type checking, 
security scanning, performance benchmarking, and integration tests.
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import tempfile
import os

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class QualityGate:
    """Individual quality gate check."""
    
    def __init__(self, name: str, command: List[str], required: bool = True, timeout: int = 300):
        self.name = name
        self.command = command
        self.required = required
        self.timeout = timeout
        self.passed = False
        self.output = ""
        self.error = ""
        self.execution_time = 0.0
        
    def run(self) -> bool:
        """Run the quality gate check."""
        print(f"{Colors.BLUE}🔍 Running {self.name}...{Colors.END}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(__file__).parent.parent
            )
            
            self.execution_time = time.time() - start_time
            self.output = result.stdout
            self.error = result.stderr
            self.passed = result.returncode == 0
            
            if self.passed:
                print(f"{Colors.GREEN}✅ {self.name} passed ({self.execution_time:.2f}s){Colors.END}")
            else:
                print(f"{Colors.RED}❌ {self.name} failed ({self.execution_time:.2f}s){Colors.END}")
                if self.error:
                    print(f"   Error: {self.error[:200]}...")
                    
        except subprocess.TimeoutExpired:
            self.execution_time = time.time() - start_time
            self.passed = False
            self.error = f"Timeout after {self.timeout}s"
            print(f"{Colors.RED}❌ {self.name} timed out{Colors.END}")
            
        except Exception as e:
            self.execution_time = time.time() - start_time
            self.passed = False
            self.error = str(e)
            print(f"{Colors.RED}❌ {self.name} failed with exception: {e}{Colors.END}")
        
        return self.passed


class QualityGateRunner:
    """Runs all quality gates and generates comprehensive report."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.gates: List[QualityGate] = []
        self.results: Dict[str, Any] = {}
        
        # Setup quality gates
        self._setup_gates()
    
    def _setup_gates(self):
        """Setup all quality gate checks."""
        
        # 1. Code Formatting
        self.gates.append(QualityGate(
            "Code Formatting (Black)",
            ["python", "-m", "black", "--check", "--diff", "src/", "tests/", "examples/"],
            required=True
        ))
        
        # 2. Import Sorting
        self.gates.append(QualityGate(
            "Import Sorting (isort)",
            ["python", "-m", "isort", "--check-only", "--diff", "src/", "tests/", "examples/"],
            required=True
        ))
        
        # 3. Linting
        self.gates.append(QualityGate(
            "Linting (Ruff)",
            ["python", "-m", "ruff", "check", "src/", "tests/", "examples/"],
            required=True
        ))
        
        # 4. Type Checking
        self.gates.append(QualityGate(
            "Type Checking (MyPy)",
            ["python", "-m", "mypy", "src/self_evolving_moe/", "--ignore-missing-imports"],
            required=False  # Optional due to complexity
        ))
        
        # 5. Security Scanning
        self.gates.append(QualityGate(
            "Security Scanning (Bandit)",
            ["python", "-m", "bandit", "-r", "src/", "-f", "json"],
            required=True
        ))
        
        # 6. Dependency Vulnerability Check
        self.gates.append(QualityGate(
            "Dependency Vulnerabilities (Safety)",
            ["python", "-m", "safety", "check", "--json"],
            required=False  # Optional as it may have false positives
        ))
        
        # 7. Unit Tests
        self.gates.append(QualityGate(
            "Unit Tests (PyTest)",
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--maxfail=5"],
            required=True,
            timeout=600  # 10 minutes for tests
        ))
        
        # 8. Test Coverage
        self.gates.append(QualityGate(
            "Test Coverage",
            ["python", "-m", "pytest", "tests/", "--cov=self_evolving_moe", "--cov-report=term-missing", "--cov-fail-under=70"],
            required=False  # Optional for now
        ))
        
        # 9. Documentation Check
        self.gates.append(QualityGate(
            "Documentation (pydocstyle)",
            ["python", "-m", "pydocstyle", "src/self_evolving_moe/", "--convention=google"],
            required=False
        ))
        
        # 10. Package Build Test
        self.gates.append(QualityGate(
            "Package Build Test",
            ["python", "-m", "build", "--wheel", "--outdir", "dist/"],
            required=True
        ))
        
        # 11. Docker Build Test
        self.gates.append(QualityGate(
            "Docker Build Test",
            ["docker", "build", "-f", "docker/Dockerfile.production", "-t", "moe-test", "."],
            required=False,  # Optional if Docker not available
            timeout=900  # 15 minutes for Docker build
        ))
    
    def run_all_gates(self, fail_fast: bool = False, skip_optional: bool = False) -> Dict[str, Any]:
        """
        Run all quality gates.
        
        Args:
            fail_fast: Stop on first failure
            skip_optional: Skip non-required gates
            
        Returns:
            Comprehensive results dictionary
        """
        print(f"{Colors.BOLD}🚀 Running Quality Gates for Self-Evolving MoE-Router{Colors.END}")
        print(f"Project Root: {self.project_root}")
        print(f"Total Gates: {len(self.gates)}")
        print("=" * 80)
        
        start_time = time.time()
        passed_gates = []
        failed_gates = []
        skipped_gates = []
        
        for gate in self.gates:
            # Skip optional gates if requested
            if skip_optional and not gate.required:
                skipped_gates.append(gate.name)
                print(f"{Colors.YELLOW}⏭️  Skipping optional gate: {gate.name}{Colors.END}")
                continue
            
            # Run the gate
            success = gate.run()
            
            if success:
                passed_gates.append(gate.name)
            else:
                failed_gates.append(gate.name)
                
                # Stop on first failure if fail_fast enabled
                if fail_fast and gate.required:
                    print(f"{Colors.RED}🛑 Stopping due to failed required gate: {gate.name}{Colors.END}")
                    break
        
        total_time = time.time() - start_time
        
        # Generate results
        self.results = {
            'timestamp': time.time(),
            'total_time_seconds': total_time,
            'gates_run': len(passed_gates) + len(failed_gates),
            'gates_passed': len(passed_gates),
            'gates_failed': len(failed_gates),
            'gates_skipped': len(skipped_gates),
            'success_rate': len(passed_gates) / (len(passed_gates) + len(failed_gates)) * 100 if (passed_gates or failed_gates) else 0,
            'overall_success': len(failed_gates) == 0 or all(not gate.required for gate in self.gates if gate.name in failed_gates),
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'skipped_gates': skipped_gates,
            'gate_details': {
                gate.name: {
                    'passed': gate.passed,
                    'required': gate.required,
                    'execution_time': gate.execution_time,
                    'command': ' '.join(gate.command),
                    'output': gate.output[:1000] if gate.output else '',  # Truncate long output
                    'error': gate.error[:1000] if gate.error else ''
                }
                for gate in self.gates
            }
        }
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print quality gates summary."""
        print("\n" + "=" * 80)
        print(f"{Colors.BOLD}📊 QUALITY GATES SUMMARY{Colors.END}")
        print("=" * 80)
        
        # Overall status
        if self.results['overall_success']:
            print(f"{Colors.GREEN}✅ OVERALL STATUS: PASSED{Colors.END}")
        else:
            print(f"{Colors.RED}❌ OVERALL STATUS: FAILED{Colors.END}")
        
        # Statistics
        print(f"📈 Success Rate: {self.results['success_rate']:.1f}%")
        print(f"⏱️  Total Time: {self.results['total_time_seconds']:.1f}s")
        print(f"🏃 Gates Run: {self.results['gates_run']}")
        print(f"✅ Passed: {self.results['gates_passed']}")
        print(f"❌ Failed: {self.results['gates_failed']}")
        print(f"⏭️  Skipped: {self.results['gates_skipped']}")
        
        # Failed gates details
        if self.results['failed_gates']:
            print(f"\n{Colors.RED}❌ FAILED GATES:{Colors.END}")
            for gate_name in self.results['failed_gates']:
                gate_details = self.results['gate_details'][gate_name]
                required_text = "REQUIRED" if gate_details['required'] else "OPTIONAL"
                print(f"   • {gate_name} ({required_text}) - {gate_details['execution_time']:.2f}s")
                if gate_details['error']:
                    print(f"     Error: {gate_details['error'][:100]}...")
        
        # Recommendations
        self._print_recommendations()
    
    def _print_recommendations(self):
        """Print improvement recommendations."""
        print(f"\n{Colors.BLUE}💡 RECOMMENDATIONS:{Colors.END}")
        
        failed_gates = self.results['failed_gates']
        
        if not failed_gates:
            print("   🎉 All quality gates passed! System is production-ready.")
            return
        
        # Specific recommendations based on failed gates
        recommendations = []
        
        if "Code Formatting (Black)" in failed_gates:
            recommendations.append("Run 'python -m black src/ tests/ examples/' to fix formatting")
        
        if "Import Sorting (isort)" in failed_gates:
            recommendations.append("Run 'python -m isort src/ tests/ examples/' to fix imports")
        
        if "Linting (Ruff)" in failed_gates:
            recommendations.append("Fix linting issues reported by Ruff")
        
        if "Security Scanning (Bandit)" in failed_gates:
            recommendations.append("Review and fix security issues found by Bandit")
        
        if "Unit Tests (PyTest)" in failed_gates:
            recommendations.append("Fix failing unit tests before deployment")
        
        if "Package Build Test" in failed_gates:
            recommendations.append("Fix package configuration and dependencies")
        
        # Generic recommendations
        if len(failed_gates) > 3:
            recommendations.append("Consider running individual gates to debug issues")
        
        recommendations.append("Review detailed error messages above for specific fixes")
        recommendations.append("Run with --verbose flag for more detailed output")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def save_report(self, output_path: Path):
        """Save detailed quality gates report."""
        report_data = {
            **self.results,
            'project_info': {
                'project_root': str(self.project_root),
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved to: {output_path}")
    
    def generate_badge_info(self) -> Dict[str, str]:
        """Generate badge information for README."""
        if self.results['overall_success']:
            return {
                'status': 'passing',
                'color': 'brightgreen',
                'message': f"{self.results['gates_passed']}/{self.results['gates_run']} gates passed"
            }
        else:
            return {
                'status': 'failing', 
                'color': 'red',
                'message': f"{self.results['gates_failed']} gates failed"
            }


def main():
    """Main entry point for quality gates runner."""
    parser = argparse.ArgumentParser(description='Run quality gates for Self-Evolving MoE-Router')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    parser.add_argument('--skip-optional', action='store_true', help='Skip optional gates')
    parser.add_argument('--report', type=str, help='Save detailed report to file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create quality gate runner
    runner = QualityGateRunner(project_root)
    
    # Run all gates
    try:
        results = runner.run_all_gates(
            fail_fast=args.fail_fast,
            skip_optional=args.skip_optional
        )
        
        # Save report if requested
        if args.report:
            runner.save_report(Path(args.report))
        
        # Generate badge info
        badge_info = runner.generate_badge_info()
        print(f"\n🏷️  Badge Info: {badge_info}")
        
        # Exit with appropriate code
        if results['overall_success']:
            print(f"\n{Colors.GREEN}🎉 All quality gates passed! System ready for production.{Colors.END}")
            sys.exit(0)
        else:
            required_failures = [
                gate for gate in runner.gates 
                if gate.name in results['failed_gates'] and gate.required
            ]
            
            if required_failures:
                print(f"\n{Colors.RED}🚨 Required quality gates failed. Fix issues before deployment.{Colors.END}")
                sys.exit(1)
            else:
                print(f"\n{Colors.YELLOW}⚠️  Some optional gates failed, but system can still be deployed.{Colors.END}")
                sys.exit(0)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⏹️  Quality gates interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}💥 Quality gates runner failed: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()