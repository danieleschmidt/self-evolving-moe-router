#!/usr/bin/env python3
"""
Simplified Quality Gates for Self-Evolving MoE Router.

This script performs essential quality checks without external dependencies.
"""

import os
import sys
import ast
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class QualityResult:
    """Result of a quality check."""
    
    def __init__(self, name: str, passed: bool, score: float = 0.0):
        self.name = name
        self.passed = passed
        self.score = score
        self.details = {}
        self.errors = []
        self.warnings = []
        self.recommendations = []


class QualityGateRunner:
    """Simplified quality gate runner."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results: List[QualityResult] = []
        self.overall_score = 0.0
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🚀 Starting Quality Gate Validation")
        print("=" * 60)
        
        gates = [
            ("Code Structure", self.check_code_structure),
            ("Import Dependencies", self.check_imports),
            ("Configuration Validation", self.check_configurations),
            ("Security Scanning", self.check_security),
            ("Documentation", self.check_documentation),
            ("Test Coverage", self.check_test_coverage),
            ("Production Readiness", self.check_production_readiness),
        ]
        
        for gate_name, gate_func in gates:
            print(f"\n📋 Running {gate_name}...")
            try:
                result = gate_func()
                self.results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"   {status} - Score: {result.score:.1f}/100")
                
                if result.errors:
                    print(f"   Errors: {len(result.errors)}")
                if result.warnings:
                    print(f"   Warnings: {len(result.warnings)}")
                    
            except Exception as e:
                error_result = QualityResult(
                    name=gate_name,
                    passed=False,
                    score=0.0
                )
                error_result.errors.append(f"Gate execution failed: {str(e)}")
                self.results.append(error_result)
                print(f"   ❌ FAIL - Gate execution error: {e}")
        
        # Calculate overall score
        self.overall_score = sum(r.score for r in self.results) / len(self.results)
        
        return self.generate_report()
    
    def check_code_structure(self) -> QualityResult:
        """Check code structure and organization."""
        result = QualityResult("Code Structure", True, 100.0)
        
        # Check required directories
        required_dirs = [
            "src/self_evolving_moe",
            "src/self_evolving_moe/evolution",
            "src/self_evolving_moe/experts",
            "src/self_evolving_moe/routing",
            "src/self_evolving_moe/monitoring",
            "src/self_evolving_moe/utils",
            "tests",
            "deployment"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.repo_path / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            result.errors.extend([f"Missing directory: {d}" for d in missing_dirs])
            result.score -= len(missing_dirs) * 10
            result.passed = result.score > 50
        
        # Check key files
        key_files = [
            "src/self_evolving_moe/__init__.py",
            "src/self_evolving_moe/evolution/router.py",
            "src/self_evolving_moe/experts/pool.py",
            "src/self_evolving_moe/routing/topology.py",
            "pyproject.toml",
            "README.md"
        ]
        
        missing_files = []
        for file_path in key_files:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            result.errors.extend([f"Missing file: {f}" for f in missing_files])
            result.score -= len(missing_files) * 5
            result.passed = result.score > 50
        
        # Count Python files
        py_files = list(self.repo_path.rglob("*.py"))
        result.details['python_files_count'] = len(py_files)
        result.details['lines_of_code'] = self._count_lines_of_code(py_files)
        
        return result
    
    def check_imports(self) -> QualityResult:
        """Check import dependencies."""
        result = QualityResult("Import Dependencies", True, 100.0)
        
        py_files = list((self.repo_path / "src").rglob("*.py"))
        
        import_errors = []
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic syntax check
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    import_errors.append(f"Syntax error in {py_file}: {e}")
                    
            except Exception as e:
                import_errors.append(f"Failed to parse {py_file}: {e}")
        
        if import_errors:
            result.errors.extend(import_errors)
            result.score -= min(len(import_errors) * 10, 80)
            result.passed = False
        
        result.details['analyzed_files'] = len(py_files)
        result.details['syntax_errors'] = len(import_errors)
        
        return result
    
    def check_configurations(self) -> QualityResult:
        """Check configuration files and settings."""
        result = QualityResult("Configuration Validation", True, 100.0)
        
        # Check pyproject.toml
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    
                # Check for required sections
                required_sections = ['build-system', 'project']
                for section in required_sections:
                    if f'[{section}]' not in content:
                        result.errors.append(f"Missing [{section}] in pyproject.toml")
                        result.score -= 10
                
            except Exception as e:
                result.errors.append(f"Failed to read pyproject.toml: {e}")
                result.score -= 20
        else:
            result.errors.append("Missing pyproject.toml")
            result.score -= 25
            result.passed = False
        
        # Check for configuration files
        config_files = [
            "deployment/docker-compose.yml",
            "deployment/kubernetes/deployment.yaml",
            ".gitignore"
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not (self.repo_path / config_file).exists():
                missing_configs.append(config_file)
        
        if missing_configs:
            result.warnings.extend([f"Missing config: {c}" for c in missing_configs])
            result.score -= len(missing_configs) * 5
        
        result.details['missing_configurations'] = missing_configs
        
        return result
    
    def check_security(self) -> QualityResult:
        """Basic security checks."""
        result = QualityResult("Security Scanning", True, 100.0)
        
        py_files = list(self.repo_path.rglob("*.py"))
        security_issues = []
        
        # Basic security pattern matching
        security_patterns = [
            (r'eval\s*\(', "Use of eval() function"),
            (r'exec\s*\(', "Use of exec() function"),
            (r'os\.system\s*\(', "Use of os.system()"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
        ]
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, issue_desc in security_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        security_issues.append(f"{py_file}:{line_num} - {issue_desc}")
                
            except Exception as e:
                result.warnings.append(f"Could not scan {py_file}: {e}")
        
        if security_issues:
            result.errors.extend(security_issues)
            result.score -= min(len(security_issues) * 15, 90)
            if len(security_issues) > 0:
                result.passed = False
        
        result.details['security_issues'] = len(security_issues)
        result.details['files_scanned'] = len(py_files)
        
        return result
    
    def check_documentation(self) -> QualityResult:
        """Check documentation completeness."""
        result = QualityResult("Documentation", True, 85.0)
        
        # Check README
        readme_path = self.repo_path / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                required_sections = [
                    "Installation",
                    "Usage", 
                    "Features",
                ]
                
                missing_sections = []
                for section in required_sections:
                    if section.lower() not in readme_content.lower():
                        missing_sections.append(section)
                
                if missing_sections:
                    result.warnings.extend([f"README missing: {s}" for s in missing_sections])
                    result.score -= len(missing_sections) * 5
                
                result.details['readme_length'] = len(readme_content)
                
            except Exception as e:
                result.errors.append(f"Failed to read README: {e}")
                result.score -= 15
        else:
            result.errors.append("Missing README.md")
            result.score -= 25
        
        # Check docstrings
        py_files = list((self.repo_path / "src").rglob("*.py"))
        functions_with_docstrings = 0
        total_functions = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        if ast.get_docstring(node):
                            functions_with_docstrings += 1
                
            except Exception:
                continue
        
        if total_functions > 0:
            docstring_coverage = (functions_with_docstrings / total_functions) * 100
            result.details['docstring_coverage'] = docstring_coverage
            
            if docstring_coverage < 60:
                result.warnings.append(f"Low docstring coverage: {docstring_coverage:.1f}%")
                result.score -= (60 - docstring_coverage) * 0.5
        
        result.details['total_functions'] = total_functions
        result.details['functions_with_docstrings'] = functions_with_docstrings
        
        return result
    
    def check_test_coverage(self) -> QualityResult:
        """Check test coverage."""
        result = QualityResult("Test Coverage", True, 80.0)
        
        # Count test files
        test_files = list(self.repo_path.rglob("test_*.py")) + list(self.repo_path.rglob("*_test.py"))
        source_files = list((self.repo_path / "src").rglob("*.py"))
        
        if not test_files:
            result.errors.append("No test files found")
            result.score = 0
            result.passed = False
            return result
        
        # Basic coverage estimation
        test_coverage_ratio = len(test_files) / max(len(source_files), 1)
        estimated_coverage = min(test_coverage_ratio * 100, 85)  # Cap at 85% for estimation
        
        result.details['test_files'] = len(test_files)
        result.details['source_files'] = len(source_files)
        result.details['estimated_coverage'] = estimated_coverage
        
        if estimated_coverage < 60:
            result.errors.append(f"Low test coverage estimated: {estimated_coverage:.1f}%")
            result.score = estimated_coverage * 0.8
            result.passed = False
        else:
            result.score = min(90, estimated_coverage)
        
        return result
    
    def check_production_readiness(self) -> QualityResult:
        """Check production readiness."""
        result = QualityResult("Production Readiness", True, 85.0)
        
        # Check deployment files
        deployment_files = [
            "deployment/docker-compose.yml",
            "deployment/Dockerfile",
            "deployment/kubernetes/deployment.yaml",
        ]
        
        missing_deployment = []
        for deploy_file in deployment_files:
            if not (self.repo_path / deploy_file).exists():
                missing_deployment.append(deploy_file)
        
        if missing_deployment:
            result.warnings.extend([f"Missing deployment: {d}" for d in missing_deployment])
            result.score -= len(missing_deployment) * 8
        
        # Check for monitoring and logging
        monitoring_files = [
            "src/self_evolving_moe/monitoring/health_monitor.py",
            "src/self_evolving_moe/utils/logging.py"
        ]
        
        missing_monitoring = []
        for mon_file in monitoring_files:
            if not (self.repo_path / mon_file).exists():
                missing_monitoring.append(mon_file)
        
        if missing_monitoring:
            result.warnings.extend([f"Missing monitoring: {m}" for m in missing_monitoring])
            result.score -= len(missing_monitoring) * 10
        
        # Check for security features
        security_files = [
            "src/self_evolving_moe/utils/validation.py",
            "src/self_evolving_moe/utils/exceptions.py"
        ]
        
        missing_security = []
        for sec_file in security_files:
            if not (self.repo_path / sec_file).exists():
                missing_security.append(sec_file)
        
        if missing_security:
            result.warnings.extend([f"Missing security: {s}" for s in missing_security])
            result.score -= len(missing_security) * 8
        
        result.details['missing_deployment'] = missing_deployment
        result.details['missing_monitoring'] = missing_monitoring
        result.details['missing_security'] = missing_security
        
        return result
    
    def _count_lines_of_code(self, py_files: List[Path]) -> int:
        """Count lines of code."""
        total_lines = 0
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Count non-empty, non-comment lines
                    code_lines = [line for line in lines 
                                 if line.strip() and not line.strip().startswith('#')]
                    total_lines += len(code_lines)
            except Exception:
                pass
        return total_lines
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        passed_gates = sum(1 for r in self.results if r.passed)
        total_gates = len(self.results)
        
        report = {
            'timestamp': time.time(),
            'overall_score': self.overall_score,
            'overall_status': 'PASS' if self.overall_score >= 75 else 'FAIL',
            'gates_passed': passed_gates,
            'total_gates': total_gates,
            'pass_rate': (passed_gates / total_gates) * 100 if total_gates > 0 else 0,
            'results': []
        }
        
        for result in self.results:
            report['results'].append({
                'name': result.name,
                'passed': result.passed,
                'score': result.score,
                'errors': result.errors,
                'warnings': result.warnings,
                'recommendations': result.recommendations,
                'details': result.details
            })
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print quality gate summary."""
        print(f"\n{'='*60}")
        print("🏁 QUALITY GATE SUMMARY")
        print(f"{'='*60}")
        
        status = report['overall_status']
        status_emoji = "✅" if status == "PASS" else "❌"
        print(f"\nOverall Status: {status_emoji} {status}")
        print(f"Overall Score: {report['overall_score']:.1f}/100")
        print(f"Gates Passed: {report['gates_passed']}/{report['total_gates']} ({report['pass_rate']:.1f}%)")
        
        print(f"\n📊 Individual Gate Results:")
        for result in report['results']:
            status_emoji = "✅" if result['passed'] else "❌"
            print(f"  {status_emoji} {result['name']}: {result['score']:.1f}/100")
            
            if result['errors']:
                print(f"    ❌ Errors: {len(result['errors'])}")
                for error in result['errors'][:2]:  # Show first 2
                    print(f"      • {error}")
                if len(result['errors']) > 2:
                    print(f"      ... and {len(result['errors']) - 2} more")
            
            if result['warnings']:
                print(f"    ⚠️  Warnings: {len(result['warnings'])}")
                for warning in result['warnings'][:1]:  # Show first 1
                    print(f"      • {warning}")
                if len(result['warnings']) > 1:
                    print(f"      ... and {len(result['warnings']) - 1} more")
        
        print(f"\n{'='*60}")
        
        if status == "PASS":
            print("🎉 Quality gates validation completed successfully!")
        else:
            print("⚠️  Some quality gates need attention. Review issues above.")
        
        print(f"{'='*60}")
        
        return status == "PASS"


def main():
    """Main entry point."""
    runner = QualityGateRunner()
    
    # Run all quality gates
    report = runner.run_all_gates()
    
    # Print summary
    success = runner.print_summary(report)
    
    # Save detailed report
    report_path = Path("/root/repo/quality_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()