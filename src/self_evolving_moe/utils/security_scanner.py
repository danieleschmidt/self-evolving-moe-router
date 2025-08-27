"""
Advanced security scanner for the self-evolving MoE system.

This module provides comprehensive security analysis with intelligent
pattern detection and context-aware evaluation to minimize false positives.
"""

import ast
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class SecurityLevel(Enum):
    """Security vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Security issue data structure."""
    level: SecurityLevel
    category: str
    message: str
    file_path: str
    line_number: int
    context: str
    suggestion: str


class SecurityScanner:
    """Intelligent security scanner with context awareness."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.whitelist_patterns = {
            'eval_whitelist': [
                r'model\.eval\(\)',  # PyTorch model eval
                r'self\.eval\(\)',   # PyTorch model eval
                r'\.eval\(\)',       # Method calls
                r'def eval\(',       # Method definitions
                r'eval\(.*\).*#.*safe',  # Commented as safe
            ],
            'exec_whitelist': [
                r'def exec\(',       # Method definitions
                r'exec\(.*\).*#.*safe',  # Commented as safe
            ]
        }
        
    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """
        Scan a single file for security issues.
        
        Args:
            file_path: Path to file to scan
            
        Returns:
            List of security issues found
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            issues = []
            
            # AST-based analysis for more accurate detection
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_ast(tree, file_path, content))
            except SyntaxError:
                # Fallback to regex analysis for files with syntax errors
                issues.extend(self._analyze_regex(content, file_path))
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
            return []
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, content: str) -> List[SecurityIssue]:
        """Analyze AST for security issues."""
        issues = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    if func_name == 'eval':
                        line_no = getattr(node, 'lineno', 0)
                        context = lines[line_no - 1] if line_no <= len(lines) else ""
                        
                        # Check if this is whitelisted
                        if not self._is_whitelisted(context, 'eval_whitelist'):
                            issues.append(SecurityIssue(
                                level=SecurityLevel.HIGH,
                                category="Dangerous Function Call",
                                message="Use of eval() function detected",
                                file_path=str(file_path),
                                line_number=line_no,
                                context=context.strip(),
                                suggestion="Replace eval() with safer alternatives like ast.literal_eval() or specific parsing logic"
                            ))
                    
                    elif func_name == 'exec':
                        line_no = getattr(node, 'lineno', 0)
                        context = lines[line_no - 1] if line_no <= len(lines) else ""
                        
                        if not self._is_whitelisted(context, 'exec_whitelist'):
                            issues.append(SecurityIssue(
                                level=SecurityLevel.HIGH,
                                category="Dangerous Function Call",
                                message="Use of exec() function detected",
                                file_path=str(file_path),
                                line_number=line_no,
                                context=context.strip(),
                                suggestion="Replace exec() with safer alternatives or specific function calls"
                            ))
            
            # Check for hardcoded secrets
            if isinstance(node, ast.Str) or isinstance(node, ast.Constant):
                value = node.s if isinstance(node, ast.Str) else getattr(node, 'value', '')
                if isinstance(value, str) and len(value) > 8:
                    if self._looks_like_secret(value):
                        line_no = getattr(node, 'lineno', 0)
                        context = lines[line_no - 1] if line_no <= len(lines) else ""
                        
                        issues.append(SecurityIssue(
                            level=SecurityLevel.MEDIUM,
                            category="Potential Secret",
                            message="Potential hardcoded secret detected",
                            file_path=str(file_path),
                            line_number=line_no,
                            context=context.strip(),
                            suggestion="Move secrets to environment variables or secure configuration"
                        ))
        
        return issues
    
    def _analyze_regex(self, content: str, file_path: Path) -> List[SecurityIssue]:
        """Fallback regex analysis for files that can't be parsed as AST."""
        issues = []
        lines = content.split('\n')
        
        patterns = [
            (r'\beval\s*\(', SecurityLevel.HIGH, "Use of eval() function"),
            (r'\bexec\s*\(', SecurityLevel.HIGH, "Use of exec() function"),
            (r'os\.system\s*\(', SecurityLevel.MEDIUM, "Use of os.system()"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, level, message in patterns:
                if re.search(pattern, line):
                    # Check whitelisting
                    whitelist_key = f"{message.split()[2]}_whitelist"  # Extract function name
                    if not self._is_whitelisted(line, whitelist_key):
                        issues.append(SecurityIssue(
                            level=level,
                            category="Pattern Match",
                            message=message,
                            file_path=str(file_path),
                            line_number=i,
                            context=line.strip(),
                            suggestion="Review usage and replace with safer alternatives"
                        ))
        
        return issues
    
    def _is_whitelisted(self, context: str, whitelist_key: str) -> bool:
        """Check if a line matches whitelist patterns."""
        if whitelist_key not in self.whitelist_patterns:
            return False
        
        patterns = self.whitelist_patterns[whitelist_key]
        return any(re.search(pattern, context, re.IGNORECASE) for pattern in patterns)
    
    def _looks_like_secret(self, value: str) -> bool:
        """Heuristic to identify potential secrets."""
        # Skip obvious non-secrets
        common_strings = {'test', 'example', 'placeholder', 'default', 'sample'}
        if value.lower() in common_strings:
            return False
        
        # Check for secret-like patterns
        secret_indicators = [
            r'(?i)(api[_-]?key|secret|token|password|pass)',  # Common secret keywords
            r'^[A-Za-z0-9+/]{20,}={0,2}$',  # Base64-like
            r'^[a-fA-F0-9]{16,}$',  # Hexadecimal
        ]
        
        return any(re.search(pattern, value) for pattern in secret_indicators)
    
    def scan_directory(self, directory: Path, extensions: Set[str] = {'.py'}) -> Dict[str, List[SecurityIssue]]:
        """
        Scan directory for security issues.
        
        Args:
            directory: Directory to scan
            extensions: File extensions to include
            
        Returns:
            Dictionary mapping file paths to security issues
        """
        results = {}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                issues = self.scan_file(file_path)
                if issues:
                    results[str(file_path)] = issues
        
        return results
    
    def generate_report(self, scan_results: Dict[str, List[SecurityIssue]]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        total_issues = sum(len(issues) for issues in scan_results.values())
        
        level_counts = {level.value: 0 for level in SecurityLevel}
        category_counts = {}
        
        for issues in scan_results.values():
            for issue in issues:
                level_counts[issue.level.value] += 1
                category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        security_score = max(0, 100 - (
            level_counts['critical'] * 25 +
            level_counts['high'] * 15 +
            level_counts['medium'] * 8 +
            level_counts['low'] * 3 +
            level_counts['info'] * 1
        ))
        
        return {
            'security_score': security_score,
            'total_issues': total_issues,
            'files_scanned': len(scan_results),
            'level_breakdown': level_counts,
            'category_breakdown': category_counts,
            'detailed_issues': scan_results,
            'recommendations': self._generate_recommendations(scan_results)
        }
    
    def _generate_recommendations(self, scan_results: Dict[str, List[SecurityIssue]]) -> List[str]:
        """Generate actionable security recommendations."""
        recommendations = []
        
        # Count issue types
        issue_types = {}
        for issues in scan_results.values():
            for issue in issues:
                key = f"{issue.category}_{issue.level.value}"
                issue_types[key] = issue_types.get(key, 0) + 1
        
        # Generate recommendations based on findings
        if any('eval' in key.lower() for key in issue_types):
            recommendations.append("Replace eval() and exec() calls with safer alternatives")
        
        if any('secret' in key.lower() for key in issue_types):
            recommendations.append("Move hardcoded secrets to environment variables")
        
        if any('system' in key.lower() for key in issue_types):
            recommendations.append("Replace os.system() calls with subprocess module")
        
        recommendations.append("Implement regular security audits")
        recommendations.append("Use static analysis tools in CI/CD pipeline")
        
        return recommendations


# Convenience function for quick scanning
def scan_project(project_root: Path) -> Dict[str, Any]:
    """Quick scan of entire project."""
    scanner = SecurityScanner()
    results = scanner.scan_directory(project_root)
    return scanner.generate_report(results)