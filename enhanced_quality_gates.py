#!/usr/bin/env python3
"""
Enhanced Quality Gates System - Generation 1 Implementation
Intelligent testing, security scanning, performance validation with context awareness
"""

import subprocess
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import tempfile
import ast
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class QualityResult:
    """Enhanced quality gate result with detailed metrics."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    duration: float
    error_message: Optional[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class QualityReport:
    """Comprehensive quality report with actionable insights."""
    timestamp: str
    overall_passed: bool
    overall_score: float
    results: List[QualityResult]
    summary: Dict[str, Any]
    global_recommendations: List[str] = None
    
    def __post_init__(self):
        if self.global_recommendations is None:
            self.global_recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper type handling."""
        def convert_value(v):
            if isinstance(v, (bool, int, float, str)):
                return v
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            elif hasattr(v, '__dict__'):
                return {k: convert_value(val) for k, val in v.__dict__.items()}
            return str(v)
        
        return convert_value({
            'timestamp': self.timestamp,
            'overall_passed': self.overall_passed,
            'overall_score': self.overall_score,
            'results': [asdict(r) for r in self.results],
            'summary': self.summary,
            'global_recommendations': self.global_recommendations
        })


class EnhancedQualityGate:
    """Base class for enhanced quality gates with intelligence."""
    
    def __init__(self, name: str, required: bool = True, min_score: float = 0.8):
        self.name = name
        self.required = required
        self.min_score = min_score
        self.logger = logging.getLogger(f"QualityGate.{name}")
    
    def run(self) -> QualityResult:
        """Run the enhanced quality gate."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running {self.name}...")
            passed, score, details, recommendations = self._execute()
            duration = time.time() - start_time
            
            result = QualityResult(
                name=self.name,
                passed=passed,
                score=score,
                details=details,
                duration=duration,
                recommendations=recommendations
            )
            
            status = "✅ PASSED" if passed else "❌ FAILED"
            self.logger.info(f"{self.name}: {status} (Score: {score:.2f}, Duration: {duration:.2f}s)")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"{self.name} failed with exception: {e}")
            
            return QualityResult(
                name=self.name,
                passed=False,
                score=0.0,
                details={"error": str(e), "traceback": traceback.format_exc()},
                duration=duration,
                error_message=str(e),
                recommendations=[f"Fix {self.name} execution error: {str(e)}"]
            )
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Execute the quality gate (to be implemented by subclasses)."""
        raise NotImplementedError


class CodeComplexityGate(EnhancedQualityGate):
    """Enhanced code complexity analysis with recommendations."""
    
    def __init__(self):
        super().__init__("Code Complexity Analysis", min_score=0.7)
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Analyze code complexity with intelligent recommendations."""
        
        def calculate_complexity(node):
            """Calculate cyclomatic complexity of AST node."""
            complexity = 1  # Base complexity
            
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                                     ast.Try, ast.With, ast.AsyncWith)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
                elif isinstance(child, (ast.ExceptHandler)):
                    complexity += 1
                    
            return complexity
        
        def analyze_function(node, file_path):
            """Analyze individual function."""
            func_name = node.name
            complexity = calculate_complexity(node)
            lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
            
            return {
                'name': func_name,
                'complexity': complexity,
                'lines': lines,
                'file': str(file_path),
                'line_no': node.lineno,
                'issue_level': 'high' if complexity > 15 else 'medium' if complexity > 10 else 'low'
            }
        
        python_files = list(Path('/root/repo').rglob('*.py'))
        
        total_complexity = 0
        total_functions = 0
        complex_functions = []
        file_stats = {}
        
        for file_path in python_files[:50]:  # Limit for performance
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue
                
                file_complexity = 0
                file_functions = 0
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_data = analyze_function(node, file_path)
                        total_complexity += func_data['complexity']
                        total_functions += 1
                        file_complexity += func_data['complexity']
                        file_functions += 1
                        
                        if func_data['complexity'] > 10:
                            complex_functions.append(func_data)
                
                if file_functions > 0:
                    file_stats[str(file_path)] = {
                        'functions': file_functions,
                        'avg_complexity': file_complexity / file_functions,
                        'total_complexity': file_complexity
                    }
                    
            except Exception as e:
                self.logger.warning(f"Could not analyze {file_path}: {e}")
                continue
        
        avg_complexity = total_complexity / max(total_functions, 1)
        
        # Calculate score based on complexity metrics
        score = max(0, min(100, 100 - (avg_complexity - 5) * 5))  # Target complexity < 10
        passed = score >= (self.min_score * 100) and len(complex_functions) < 10
        
        # Generate intelligent recommendations
        recommendations = []
        if avg_complexity > 8:
            recommendations.append(f"Average function complexity ({avg_complexity:.1f}) exceeds recommended limit of 8")
        
        if len(complex_functions) > 0:
            recommendations.append(f"Found {len(complex_functions)} functions with high complexity (>10)")
            
        high_complexity_files = [f for f, stats in file_stats.items() 
                               if stats['avg_complexity'] > 12]
        if high_complexity_files:
            recommendations.append(f"Refactor high-complexity files: {len(high_complexity_files)} files need attention")
        
        recommendations.extend([
            "Break down complex functions into smaller, focused functions",
            "Extract common logic into utility functions",
            "Consider using design patterns to reduce conditional complexity"
        ])
        
        details = {
            'total_functions': total_functions,
            'average_complexity': avg_complexity,
            'complex_functions': complex_functions[:10],  # Top 10 most complex
            'files_analyzed': len(python_files),
            'high_complexity_count': len(complex_functions)
        }
        
        return passed, score, details, recommendations


class IntelligentSecurityGate(EnhancedQualityGate):
    """Intelligent security analysis with context awareness."""
    
    def __init__(self):
        super().__init__("Intelligent Security Analysis", min_score=0.9)
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Enhanced security scanning with context awareness."""
        
        # Import our advanced security scanner
        try:
            sys.path.append('/root/repo/src')
            from self_evolving_moe.utils.security_scanner import SecurityScanner, SecurityLevel
            
            scanner = SecurityScanner()
            scan_results = scanner.scan_directory(Path('/root/repo'))
            report = scanner.generate_report(scan_results)
            
            # Calculate enhanced scoring
            score = report['security_score']
            total_issues = report['total_issues']
            
            # Enhanced passing criteria
            critical_issues = report['level_breakdown'].get('critical', 0)
            high_issues = report['level_breakdown'].get('high', 0)
            
            passed = (critical_issues == 0 and high_issues <= 2 and score >= 85)
            
            # Generate intelligent recommendations
            recommendations = report['recommendations']
            if critical_issues > 0:
                recommendations.insert(0, f"CRITICAL: Fix {critical_issues} critical security issues immediately")
            if high_issues > 0:
                recommendations.insert(0, f"HIGH PRIORITY: Address {high_issues} high-severity security issues")
            
            details = {
                'security_score': score,
                'total_issues': total_issues,
                'files_scanned': report['files_scanned'],
                'level_breakdown': report['level_breakdown'],
                'category_breakdown': report['category_breakdown'],
                'top_issues': self._get_top_issues(scan_results, 5)
            }
            
            return passed, score, details, recommendations
            
        except Exception as e:
            self.logger.error(f"Security scanning failed: {e}")
            # Fallback to basic security check
            return self._basic_security_check()
    
    def _basic_security_check(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Fallback basic security check."""
        issues = []
        python_files = list(Path('/root/repo').rglob('*.py'))
        
        dangerous_patterns = [
            (r'\beval\s*\((?!.*model\.eval)', "eval() function usage"),
            (r'\bexec\s*\(', "exec() function usage"),
            (r'os\.system\s*\(', "os.system() usage"),
            (r'shell=True', "shell=True parameter")
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in dangerous_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip PyTorch model.eval() calls
                        line = content[:match.start()].count('\n') + 1
                        context = content.split('\n')[line-1] if line <= len(content.split('\n')) else ""
                        
                        if 'model.eval()' not in context and 'self.eval()' not in context:
                            issues.append({
                                'file': str(file_path),
                                'line': line,
                                'issue': description,
                                'context': context.strip()
                            })
                            
            except Exception as e:
                continue
        
        score = max(0, 100 - len(issues) * 10)
        passed = len(issues) == 0
        
        recommendations = [
            "Replace dangerous function calls with safer alternatives",
            "Implement input validation and sanitization",
            "Use subprocess instead of os.system()",
            "Avoid eval() and exec() - use ast.literal_eval() for safe evaluation"
        ]
        
        details = {
            'security_issues': issues[:10],  # Top 10 issues
            'total_issues': len(issues),
            'files_scanned': len(python_files)
        }
        
        return passed, score, details, recommendations
    
    def _get_top_issues(self, scan_results: Dict, limit: int = 5) -> List[Dict]:
        """Extract top security issues for reporting."""
        all_issues = []
        
        for file_path, issues in scan_results.items():
            for issue in issues:
                all_issues.append({
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'level': issue.level.value,
                    'category': issue.category,
                    'message': issue.message,
                    'context': issue.context
                })
        
        # Sort by severity level
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
        all_issues.sort(key=lambda x: severity_order.get(x['level'], 5))
        
        return all_issues[:limit]


class TestCoverageGate(EnhancedQualityGate):
    """Enhanced test coverage analysis with intelligent recommendations."""
    
    def __init__(self):
        super().__init__("Test Coverage Analysis", min_score=0.7)
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Analyze test coverage with smart recommendations."""
        
        # Count source and test files
        src_files = list(Path('/root/repo/src').rglob('*.py')) if Path('/root/repo/src').exists() else []
        test_files = list(Path('/root/repo/tests').rglob('*.py')) if Path('/root/repo/tests').exists() else []
        test_files.extend(list(Path('/root/repo').rglob('test_*.py')))
        test_files.extend(list(Path('/root/repo').rglob('*_test.py')))
        
        # Remove duplicates
        test_files = list(set(test_files))
        
        # Analyze test patterns and coverage
        src_modules = self._analyze_source_modules(src_files)
        test_coverage = self._analyze_test_coverage(test_files, src_modules)
        
        # Calculate enhanced scoring
        coverage_ratio = len(test_coverage['covered_modules']) / max(len(src_modules), 1)
        test_quality_score = self._assess_test_quality(test_files)
        
        # Combined score: coverage ratio (60%) + test quality (40%)
        score = (coverage_ratio * 60 + test_quality_score * 40)
        passed = score >= (self.min_score * 100) and len(test_files) >= max(5, len(src_files) * 0.3)
        
        # Generate intelligent recommendations
        recommendations = self._generate_test_recommendations(
            src_modules, test_coverage, test_files, coverage_ratio
        )
        
        details = {
            'source_modules': len(src_modules),
            'test_files': len(test_files),
            'coverage_ratio': coverage_ratio,
            'test_quality_score': test_quality_score,
            'uncovered_modules': test_coverage['uncovered_modules'][:10],
            'test_patterns_found': test_coverage['test_patterns']
        }
        
        return passed, score, details, recommendations
    
    def _analyze_source_modules(self, src_files: List[Path]) -> List[str]:
        """Analyze source modules for test coverage mapping."""
        modules = []
        
        for file_path in src_files:
            if file_path.name != '__init__.py':
                # Convert file path to module name
                rel_path = file_path.relative_to(Path('/root/repo'))
                module_name = str(rel_path.with_suffix('')).replace('/', '.')
                modules.append(module_name)
        
        return modules
    
    def _analyze_test_coverage(self, test_files: List[Path], src_modules: List[str]) -> Dict[str, Any]:
        """Analyze which modules have corresponding tests."""
        covered_modules = set()
        test_patterns = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for import statements to identify tested modules
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if any(module in alias.name for module in src_modules):
                                    covered_modules.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and any(module in node.module for module in src_modules):
                                covered_modules.add(node.module)
                
                    # Analyze test patterns
                    if 'def test_' in content:
                        test_patterns.append('pytest_style')
                    if 'class Test' in content:
                        test_patterns.append('unittest_class')
                    if 'assert ' in content:
                        test_patterns.append('assertions')
                    if '@pytest.fixture' in content:
                        test_patterns.append('fixtures')
                
                except SyntaxError:
                    pass
                    
            except Exception:
                continue
        
        uncovered_modules = [m for m in src_modules if not any(m in covered for covered in covered_modules)]
        
        return {
            'covered_modules': list(covered_modules),
            'uncovered_modules': uncovered_modules,
            'test_patterns': list(set(test_patterns))
        }
    
    def _assess_test_quality(self, test_files: List[Path]) -> float:
        """Assess the quality of existing tests."""
        if not test_files:
            return 0.0
        
        quality_indicators = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_score = 0
                
                # Quality indicators
                if 'assert' in content: file_score += 20
                if 'mock' in content.lower(): file_score += 15
                if 'fixture' in content.lower(): file_score += 15
                if 'parametrize' in content: file_score += 15
                if 'setUp' in content or 'tearDown' in content: file_score += 10
                if len(re.findall(r'def test_\w+', content)) >= 3: file_score += 15
                if 'docstring' in content or '"""' in content: file_score += 10
                
                quality_indicators.append(min(100, file_score))
                
            except Exception:
                continue
        
        return sum(quality_indicators) / max(len(quality_indicators), 1)
    
    def _generate_test_recommendations(self, src_modules: List[str], test_coverage: Dict,
                                     test_files: List[Path], coverage_ratio: float) -> List[str]:
        """Generate intelligent test recommendations."""
        recommendations = []
        
        if coverage_ratio < 0.5:
            recommendations.append(f"Low test coverage ({coverage_ratio:.1%}). Create tests for uncovered modules.")
        
        if len(test_coverage['uncovered_modules']) > 0:
            critical_modules = [m for m in test_coverage['uncovered_modules'][:5]]
            recommendations.append(f"Priority: Add tests for critical modules: {', '.join(critical_modules)}")
        
        if 'fixtures' not in test_coverage['test_patterns']:
            recommendations.append("Implement pytest fixtures for better test organization")
        
        if 'assertions' not in test_coverage['test_patterns']:
            recommendations.append("Add more assertion-based tests for validation")
        
        recommendations.extend([
            "Implement integration tests for end-to-end workflows",
            "Add property-based testing with hypothesis",
            "Set up continuous test coverage monitoring"
        ])
        
        return recommendations


class EnhancedQualityGateRunner:
    """Enhanced quality gate runner with comprehensive analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger("QualityGateRunner")
        
        # Initialize enhanced quality gates
        self.gates = [
            CodeComplexityGate(),
            IntelligentSecurityGate(), 
            TestCoverageGate(),
        ]
    
    def run_all(self) -> QualityReport:
        """Run all enhanced quality gates."""
        self.logger.info("Starting Enhanced Quality Gate Analysis...")
        
        start_time = time.time()
        results = []
        
        for gate in self.gates:
            result = gate.run()
            results.append(result)
        
        # Calculate overall metrics
        total_score = sum(r.score for r in results) / len(results)
        passed_gates = sum(1 for r in results if r.passed)
        total_gates = len(results)
        overall_passed = passed_gates == total_gates
        
        # Generate global recommendations
        global_recommendations = self._generate_global_recommendations(results)
        
        # Create comprehensive summary
        summary = {
            'execution_time': time.time() - start_time,
            'gates_passed': passed_gates,
            'total_gates': total_gates,
            'pass_rate': (passed_gates / total_gates) * 100,
            'improvement_areas': self._identify_improvement_areas(results)
        }
        
        report = QualityReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            overall_passed=overall_passed,
            overall_score=total_score,
            results=results,
            summary=summary,
            global_recommendations=global_recommendations
        )
        
        return report
    
    def _generate_global_recommendations(self, results: List[QualityResult]) -> List[str]:
        """Generate comprehensive project-wide recommendations."""
        recommendations = []
        
        # Analyze cross-gate patterns
        failed_gates = [r for r in results if not r.passed]
        low_score_gates = [r for r in results if r.score < 70]
        
        if len(failed_gates) > 0:
            recommendations.append(f"Priority: Fix {len(failed_gates)} failing quality gates")
        
        if len(low_score_gates) > 1:
            recommendations.append("Implement systematic code quality improvements")
        
        # Security-first recommendations
        security_results = [r for r in results if 'Security' in r.name]
        if any(not r.passed for r in security_results):
            recommendations.insert(0, "URGENT: Address security vulnerabilities before deployment")
        
        # Add strategic recommendations
        recommendations.extend([
            "Implement automated quality gates in CI/CD pipeline",
            "Set up code quality metrics dashboard",
            "Establish quality gates as deployment blockers",
            "Schedule regular code quality reviews"
        ])
        
        return recommendations
    
    def _identify_improvement_areas(self, results: List[QualityResult]) -> List[str]:
        """Identify key areas needing improvement."""
        areas = []
        
        for result in results:
            if result.score < 80:
                area_name = result.name.replace(' Gate', '').replace(' Analysis', '')
                areas.append(f"{area_name} (Score: {result.score:.1f})")
        
        return areas
    
    def save_report(self, report: QualityReport, output_file: str = None):
        """Save comprehensive quality report."""
        if output_file is None:
            output_file = f"/root/repo/enhanced_quality_report_{int(time.time())}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Enhanced quality report saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")


def main():
    """Main execution function."""
    runner = EnhancedQualityGateRunner()
    report = runner.run_all()
    
    # Save report
    runner.save_report(report)
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED QUALITY GATES REPORT")
    print("="*80)
    print(f"Overall Status: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}")
    print(f"Overall Score: {report.overall_score:.2f}/100")
    print(f"Gates Passed: {report.summary['gates_passed']}/{report.summary['total_gates']}")
    print(f"Execution Time: {report.summary['execution_time']:.2f}s")
    
    if report.summary['improvement_areas']:
        print("\n📊 Areas for Improvement:")
        for area in report.summary['improvement_areas']:
            print(f"  • {area}")
    
    if report.global_recommendations:
        print("\n🎯 Key Recommendations:")
        for rec in report.global_recommendations[:5]:
            print(f"  • {rec}")
    
    print("\n" + "="*80)
    
    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())