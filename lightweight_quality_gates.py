#!/usr/bin/env python3
"""
Lightweight Quality Gates System - Generation 1 Implementation
Fast, dependency-minimal quality analysis with intelligent recommendations
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
import ast
import re


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Quality gate result with actionable insights."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    duration: float
    recommendations: List[str]
    error_message: Optional[str] = None


@dataclass 
class QualityReport:
    """Comprehensive quality report."""
    timestamp: str
    overall_passed: bool
    overall_score: float
    results: List[QualityResult]
    summary: Dict[str, Any]
    global_recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'timestamp': self.timestamp,
            'overall_passed': self.overall_passed,
            'overall_score': self.overall_score,
            'results': [asdict(r) for r in self.results],
            'summary': self.summary,
            'global_recommendations': self.global_recommendations
        }


class LightweightQualityGate:
    """Base class for lightweight quality gates."""
    
    def __init__(self, name: str, min_score: float = 70.0):
        self.name = name
        self.min_score = min_score
    
    def run(self) -> QualityResult:
        """Run the quality gate."""
        start_time = time.time()
        
        try:
            logger.info(f"Running {self.name}...")
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
            logger.info(f"{self.name}: {status} (Score: {score:.1f}, Duration: {duration:.2f}s)")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{self.name} failed: {e}")
            
            return QualityResult(
                name=self.name,
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=duration,
                recommendations=[f"Fix execution error: {str(e)}"],
                error_message=str(e)
            )
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Execute the gate logic (implemented by subclasses)."""
        raise NotImplementedError


class CodeQualityGate(LightweightQualityGate):
    """Code quality analysis with complexity and style checks."""
    
    def __init__(self):
        super().__init__("Code Quality Analysis", min_score=70.0)
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Analyze code quality metrics."""
        
        python_files = list(Path('/root/repo').rglob('*.py'))
        if not python_files:
            return False, 0.0, {}, ["No Python files found"]
        
        # Analyze code quality factors
        total_score = 0
        file_count = 0
        complex_functions = []
        long_functions = []
        issues = []
        
        for file_path in python_files[:30]:  # Limit for performance
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                file_score = self._analyze_file(file_path, content, complex_functions, long_functions, issues)
                total_score += file_score
                file_count += 1
                
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
                continue
        
        if file_count == 0:
            return False, 0.0, {}, ["No analyzable files found"]
        
        # Calculate overall score
        avg_score = total_score / file_count
        score = max(0, min(100, avg_score))
        passed = score >= self.min_score and len(issues) < 15
        
        # Generate recommendations
        recommendations = self._generate_code_recommendations(
            complex_functions, long_functions, issues, score
        )
        
        details = {
            'files_analyzed': file_count,
            'average_score': avg_score,
            'complex_functions': len(complex_functions),
            'long_functions': len(long_functions),
            'total_issues': len(issues),
            'top_issues': issues[:10]
        }
        
        return passed, score, details, recommendations
    
    def _analyze_file(self, file_path, content, complex_functions, long_functions, issues):
        """Analyze individual file for quality metrics."""
        file_score = 100  # Start with perfect score
        
        try:
            tree = ast.parse(content)
            
            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check function complexity (simplified)
                    complexity = self._calculate_complexity(node)
                    lines = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                    
                    if complexity > 15:
                        complex_functions.append({
                            'name': node.name,
                            'file': str(file_path),
                            'complexity': complexity,
                            'line': node.lineno
                        })
                        file_score -= 5
                    
                    if lines > 50:
                        long_functions.append({
                            'name': node.name,
                            'file': str(file_path), 
                            'lines': lines,
                            'line': node.lineno
                        })
                        file_score -= 3
            
            # Check for code smells
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Long lines
                if len(line) > 120:
                    issues.append(f"{file_path}:{i} - Long line ({len(line)} chars)")
                    file_score -= 1
                
                # TODO/FIXME comments
                if 'TODO' in line or 'FIXME' in line:
                    issues.append(f"{file_path}:{i} - TODO/FIXME found")
                    file_score -= 0.5
        
        except SyntaxError:
            file_score -= 20
            issues.append(f"{file_path} - Syntax error")
        
        return max(0, file_score)
    
    def _calculate_complexity(self, node):
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _generate_code_recommendations(self, complex_functions, long_functions, issues, score):
        """Generate code quality recommendations."""
        recommendations = []
        
        if score < 60:
            recommendations.append("CRITICAL: Code quality is below acceptable threshold")
        elif score < 80:
            recommendations.append("Code quality needs improvement")
        
        if len(complex_functions) > 0:
            recommendations.append(f"Refactor {len(complex_functions)} complex functions (complexity > 15)")
        
        if len(long_functions) > 0:
            recommendations.append(f"Break down {len(long_functions)} long functions (> 50 lines)")
        
        if len(issues) > 10:
            recommendations.append("Address code style issues and technical debt")
        
        recommendations.extend([
            "Use automated code formatting (black, isort)",
            "Implement code review checklist",
            "Add function and class docstrings",
            "Consider using type hints for better code clarity"
        ])
        
        return recommendations


class SecurityGate(LightweightQualityGate):
    """Lightweight security analysis."""
    
    def __init__(self):
        super().__init__("Security Analysis", min_score=85.0)
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Perform security analysis."""
        
        python_files = list(Path('/root/repo').rglob('*.py'))
        security_issues = []
        files_scanned = 0
        
        # Security patterns to check
        patterns = [
            (r'\beval\s*\((?!.*model\.eval)', "HIGH", "eval() function usage"),
            (r'\bexec\s*\(', "HIGH", "exec() function usage"), 
            (r'os\.system\s*\(', "MEDIUM", "os.system() usage"),
            (r'shell=True', "MEDIUM", "shell=True in subprocess"),
            (r'password\s*=\s*["\'][^"\']+["\']', "MEDIUM", "Hardcoded password"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "HIGH", "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "HIGH", "Hardcoded secret"),
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                files_scanned += 1
                
                for pattern, severity, description in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_no = content[:match.start()].count('\n') + 1
                        lines = content.split('\n')
                        context = lines[line_no - 1] if line_no <= len(lines) else ""
                        
                        # Skip false positives
                        if self._is_false_positive(context, description):
                            continue
                        
                        security_issues.append({
                            'file': str(file_path),
                            'line': line_no,
                            'severity': severity,
                            'issue': description,
                            'context': context.strip()
                        })
                
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
                continue
        
        # Calculate security score
        if files_scanned == 0:
            return False, 0.0, {}, ["No files could be scanned"]
        
        # Scoring based on severity
        high_issues = len([i for i in security_issues if i['severity'] == 'HIGH'])
        medium_issues = len([i for i in security_issues if i['severity'] == 'MEDIUM'])
        
        # Deduct points based on severity
        score = 100 - (high_issues * 20) - (medium_issues * 10)
        score = max(0, score)
        
        passed = high_issues == 0 and score >= self.min_score
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(security_issues, score)
        
        details = {
            'files_scanned': files_scanned,
            'security_score': score,
            'total_issues': len(security_issues),
            'high_severity': high_issues,
            'medium_severity': medium_issues,
            'issues': security_issues[:10]  # Top 10
        }
        
        return passed, score, details, recommendations
    
    def _is_false_positive(self, context, description):
        """Check if security issue is a false positive."""
        context_lower = context.lower()
        
        if "eval() function" in description:
            # Allow PyTorch model.eval() calls
            if 'model.eval()' in context or 'self.eval()' in context or '.eval()' in context:
                return True
        
        if "password" in description or "secret" in description or "api" in description:
            # Skip test/example values
            test_indicators = ['test', 'example', 'placeholder', 'sample', 'dummy', 'fake']
            if any(indicator in context_lower for indicator in test_indicators):
                return True
        
        return False
    
    def _generate_security_recommendations(self, issues, score):
        """Generate security recommendations."""
        recommendations = []
        
        if score < 70:
            recommendations.append("CRITICAL: Security vulnerabilities need immediate attention")
        
        high_issues = [i for i in issues if i['severity'] == 'HIGH']
        if high_issues:
            recommendations.append(f"HIGH PRIORITY: Fix {len(high_issues)} critical security issues")
        
        medium_issues = [i for i in issues if i['severity'] == 'MEDIUM']
        if medium_issues:
            recommendations.append(f"Address {len(medium_issues)} medium-severity security issues")
        
        # Specific recommendations based on issue types
        issue_types = [i['issue'] for i in issues]
        if any('eval' in issue for issue in issue_types):
            recommendations.append("Replace eval()/exec() with safer alternatives like ast.literal_eval()")
        
        if any('password' in issue or 'secret' in issue for issue in issue_types):
            recommendations.append("Move secrets to environment variables or secure vaults")
        
        if any('system' in issue for issue in issue_types):
            recommendations.append("Replace os.system() with subprocess module")
        
        recommendations.extend([
            "Implement automated security scanning in CI/CD",
            "Use static analysis tools (bandit, safety)",
            "Regular security audits and penetration testing"
        ])
        
        return recommendations


class DocumentationGate(LightweightQualityGate):
    """Documentation quality analysis."""
    
    def __init__(self):
        super().__init__("Documentation Analysis", min_score=65.0)
    
    def _execute(self) -> Tuple[bool, float, Dict[str, Any], List[str]]:
        """Analyze documentation quality."""
        
        # Check README
        readme_score = self._check_readme()
        
        # Check docstrings
        docstring_score, docstring_stats = self._check_docstrings()
        
        # Check code comments
        comment_score = self._check_comments()
        
        # Combined score
        score = (readme_score * 0.3 + docstring_score * 0.5 + comment_score * 0.2)
        passed = score >= self.min_score
        
        recommendations = self._generate_doc_recommendations(readme_score, docstring_stats, score)
        
        details = {
            'readme_score': readme_score,
            'docstring_score': docstring_score,
            'comment_score': comment_score,
            'docstring_stats': docstring_stats
        }
        
        return passed, score, details, recommendations
    
    def _check_readme(self):
        """Check README quality."""
        readme_files = ['README.md', 'README.rst', 'README.txt']
        
        for readme_name in readme_files:
            readme_path = Path('/root/repo') / readme_name
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding='utf-8')
                    
                    score = 50  # Base score for existing README
                    
                    # Check for important sections
                    if 'install' in content.lower(): score += 10
                    if 'usage' in content.lower(): score += 10
                    if 'example' in content.lower(): score += 10
                    if 'api' in content.lower(): score += 5
                    if 'license' in content.lower(): score += 5
                    if 'contributing' in content.lower(): score += 5
                    if len(content) > 1000: score += 5  # Substantial content
                    
                    return min(100, score)
                    
                except Exception:
                    return 20
        
        return 0  # No README found
    
    def _check_docstrings(self):
        """Check docstring coverage and quality."""
        python_files = list(Path('/root/repo/src').rglob('*.py')) if Path('/root/repo/src').exists() else []
        
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if ast.get_docstring(node):
                                documented_classes += 1
                
                except SyntaxError:
                    continue
                    
            except Exception:
                continue
        
        # Calculate coverage
        func_coverage = documented_functions / max(total_functions, 1) * 100
        class_coverage = documented_classes / max(total_classes, 1) * 100
        overall_coverage = (func_coverage + class_coverage) / 2
        
        stats = {
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'total_classes': total_classes,
            'documented_classes': documented_classes,
            'function_coverage': func_coverage,
            'class_coverage': class_coverage
        }
        
        return overall_coverage, stats
    
    def _check_comments(self):
        """Check comment density and quality."""
        python_files = list(Path('/root/repo').rglob('*.py'))
        
        total_lines = 0
        comment_lines = 0
        
        for file_path in python_files[:20]:  # Limit for performance
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        total_lines += 1
                        if stripped.startswith('#'):
                            comment_lines += 1
                            
            except Exception:
                continue
        
        if total_lines == 0:
            return 0
        
        comment_ratio = comment_lines / total_lines * 100
        
        # Score based on reasonable comment density (5-15%)
        if comment_ratio < 2:
            return 20
        elif comment_ratio < 5:
            return 50
        elif comment_ratio <= 15:
            return 90
        else:
            return 70  # Too many comments might indicate unclear code
    
    def _generate_doc_recommendations(self, readme_score, docstring_stats, score):
        """Generate documentation recommendations."""
        recommendations = []
        
        if score < 50:
            recommendations.append("CRITICAL: Documentation is severely lacking")
        elif score < 70:
            recommendations.append("Documentation needs significant improvement")
        
        if readme_score < 60:
            recommendations.append("Improve README with installation, usage, and examples")
        
        if docstring_stats['function_coverage'] < 70:
            recommendations.append(f"Add docstrings to functions (current: {docstring_stats['function_coverage']:.1f}%)")
        
        if docstring_stats['class_coverage'] < 80:
            recommendations.append(f"Add docstrings to classes (current: {docstring_stats['class_coverage']:.1f}%)")
        
        recommendations.extend([
            "Use consistent docstring format (Google/Sphinx style)",
            "Add type hints to improve code documentation",
            "Generate API documentation with Sphinx",
            "Include code examples in docstrings"
        ])
        
        return recommendations


class LightweightQualityRunner:
    """Lightweight quality gate runner."""
    
    def __init__(self):
        self.gates = [
            CodeQualityGate(),
            SecurityGate(),
            DocumentationGate()
        ]
    
    def run_all(self) -> QualityReport:
        """Run all quality gates."""
        logger.info("Starting Lightweight Quality Gate Analysis...")
        
        start_time = time.time()
        results = []
        
        for gate in self.gates:
            result = gate.run()
            results.append(result)
        
        # Calculate overall metrics
        total_score = sum(r.score for r in results) / len(results)
        passed_gates = sum(1 for r in results if r.passed)
        overall_passed = passed_gates == len(results)
        
        # Generate global recommendations
        global_recommendations = self._generate_global_recommendations(results)
        
        summary = {
            'execution_time': time.time() - start_time,
            'gates_passed': passed_gates,
            'total_gates': len(results),
            'pass_rate': (passed_gates / len(results)) * 100
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
    
    def _generate_global_recommendations(self, results):
        """Generate global recommendations."""
        recommendations = []
        
        failed_gates = [r for r in results if not r.passed]
        if failed_gates:
            recommendations.append(f"Fix {len(failed_gates)} failing quality gates")
        
        # Priority order
        security_failed = any('Security' in r.name for r in failed_gates)
        if security_failed:
            recommendations.insert(0, "URGENT: Fix security issues before deployment")
        
        low_scores = [r for r in results if r.score < 60]
        if low_scores:
            recommendations.append("Address critical quality issues in low-scoring areas")
        
        recommendations.extend([
            "Integrate quality gates into CI/CD pipeline",
            "Set up automated quality monitoring",
            "Establish quality gates as pre-deployment checklist"
        ])
        
        return recommendations
    
    def save_report(self, report, output_file=None):
        """Save quality report."""
        if output_file is None:
            output_file = f"/root/repo/lightweight_quality_report_{int(time.time())}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Report saved: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


def main():
    """Main execution."""
    runner = LightweightQualityRunner()
    report = runner.run_all()
    
    # Save report
    runner.save_report(report)
    
    # Print summary
    print("\n" + "="*80)
    print("LIGHTWEIGHT QUALITY GATES REPORT")  
    print("="*80)
    print(f"Overall Status: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}")
    print(f"Overall Score: {report.overall_score:.1f}/100")
    print(f"Gates Passed: {report.summary['gates_passed']}/{report.summary['total_gates']}")
    print(f"Pass Rate: {report.summary['pass_rate']:.1f}%")
    print(f"Execution Time: {report.summary['execution_time']:.1f}s")
    
    print(f"\n📊 Individual Gate Results:")
    for result in report.results:
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}: {result.score:.1f}/100")
    
    if report.global_recommendations:
        print(f"\n🎯 Priority Recommendations:")
        for i, rec in enumerate(report.global_recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "="*80)
    
    return 0 if report.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())