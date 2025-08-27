#!/usr/bin/env python3
"""
Automated Code Improvement System - Generation 1
Intelligently fixes code quality issues identified by quality gates
"""

import ast
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import sys


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CodeImprover:
    """Automated code improvement with intelligent fixes."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path('/root/repo')
        self.changes_made = []
        
    def improve_code_quality(self) -> Dict[str, Any]:
        """Apply automated code quality improvements."""
        logger.info("Starting automated code quality improvements...")
        
        improvements = {
            'line_length_fixes': 0,
            'docstring_additions': 0, 
            'complexity_reductions': 0,
            'style_fixes': 0,
            'files_processed': 0,
            'changes_made': []
        }
        
        # Get Python files, prioritizing source files
        python_files = self._get_python_files_priority()
        
        for file_path in python_files[:15]:  # Limit to prevent overwhelming changes
            try:
                logger.info(f"Improving {file_path.relative_to(self.project_root)}")
                file_improvements = self._improve_file(file_path)
                
                # Aggregate improvements
                for key in improvements:
                    if key in file_improvements and isinstance(improvements[key], int):
                        improvements[key] += file_improvements[key]
                
                if file_improvements['changes_made']:
                    improvements['changes_made'].extend(file_improvements['changes_made'])
                
                improvements['files_processed'] += 1
                
            except Exception as e:
                logger.error(f"Failed to improve {file_path}: {e}")
                continue
        
        logger.info(f"Code improvement complete. Processed {improvements['files_processed']} files.")
        return improvements
    
    def _get_python_files_priority(self) -> List[Path]:
        """Get Python files prioritized by importance."""
        # Priority 1: Core source files
        src_files = list((self.project_root / 'src').rglob('*.py')) if (self.project_root / 'src').exists() else []
        
        # Priority 2: Main directory Python files (excluding tests and quality gates)
        main_files = [
            f for f in self.project_root.glob('*.py') 
            if not any(exclude in f.name.lower() for exclude in [
                'test_', '_test', 'quality_gates', 'simple_demo', 'demo_'
            ])
        ]
        
        # Priority 3: Important subdirectories
        other_files = []
        for subdir in ['examples', 'scripts']:
            subdir_path = self.project_root / subdir
            if subdir_path.exists():
                other_files.extend(subdir_path.rglob('*.py'))
        
        # Combine with priority order
        return src_files + main_files + other_files
    
    def _improve_file(self, file_path: Path) -> Dict[str, Any]:
        """Improve a single Python file."""
        improvements = {
            'line_length_fixes': 0,
            'docstring_additions': 0,
            'complexity_reductions': 0,
            'style_fixes': 0,
            'changes_made': []
        }
        
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Skip empty files
            if not original_content.strip():
                return improvements
            
            modified_content = original_content
            
            # Apply improvements
            modified_content, line_fixes = self._fix_long_lines(modified_content)
            improvements['line_length_fixes'] = line_fixes
            
            modified_content, docstring_adds = self._add_missing_docstrings(modified_content, file_path)
            improvements['docstring_additions'] = docstring_adds
            
            modified_content, style_fixes = self._fix_style_issues(modified_content)
            improvements['style_fixes'] = style_fixes
            
            # Record changes
            if modified_content != original_content:
                changes = []
                if line_fixes > 0:
                    changes.append(f"Fixed {line_fixes} long lines")
                if docstring_adds > 0:
                    changes.append(f"Added {docstring_adds} docstrings") 
                if style_fixes > 0:
                    changes.append(f"Fixed {style_fixes} style issues")
                
                improvements['changes_made'] = changes
                
                # Write improved content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                logger.info(f"  Applied: {', '.join(changes)}")
        
        except Exception as e:
            logger.warning(f"Could not improve {file_path}: {e}")
        
        return improvements
    
    def _fix_long_lines(self, content: str) -> Tuple[str, int]:
        """Fix lines that are too long."""
        lines = content.split('\n')
        modified_lines = []
        fixes_count = 0
        
        for i, line in enumerate(lines):
            if len(line) > 120 and not line.strip().startswith('#'):
                # Try to fix common long line patterns
                fixed_line = self._fix_long_line(line)
                if fixed_line != line:
                    modified_lines.append(fixed_line)
                    fixes_count += 1
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)
        
        return '\n'.join(modified_lines), fixes_count
    
    def _fix_long_line(self, line: str) -> str:
        """Fix a single long line."""
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        
        # Fix long string concatenations
        if '+' in stripped and ('"' in stripped or "'" in stripped):
            # Simple string concatenation fix
            parts = re.split(r'(\s*\+\s*)', stripped)
            if len(parts) > 2:
                fixed_parts = []
                current_line = indent + parts[0]
                
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        addition = parts[i] + parts[i + 1]
                        if len(current_line + addition) > 120:
                            fixed_parts.append(current_line)
                            current_line = indent + "    " + parts[i + 1]
                        else:
                            current_line += addition
                
                if current_line.strip():
                    fixed_parts.append(current_line)
                
                if len(fixed_parts) > 1:
                    return '\n'.join(fixed_parts)
        
        # Fix long function calls with many parameters
        if '(' in stripped and stripped.count(',') >= 3:
            # Find function call pattern
            match = re.match(r'^(.+?\()(.*)\)(.*)$', stripped)
            if match:
                func_part, params_part, end_part = match.groups()
                
                # Split parameters
                params = [p.strip() for p in params_part.split(',') if p.strip()]
                if len(params) >= 3:
                    fixed_lines = [indent + func_part]
                    
                    for param in params:
                        fixed_lines.append(indent + "    " + param + ",")
                    
                    # Remove trailing comma from last parameter and add closing
                    if fixed_lines:
                        fixed_lines[-1] = fixed_lines[-1].rstrip(',')
                        fixed_lines.append(indent + ")" + end_part)
                    
                    return '\n'.join(fixed_lines)
        
        return line
    
    def _add_missing_docstrings(self, content: str, file_path: Path) -> Tuple[str, int]:
        """Add missing docstrings to functions and classes."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content, 0
        
        lines = content.split('\n')
        additions = 0
        
        # Find functions and classes without docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    # Generate appropriate docstring
                    docstring = self._generate_docstring(node, file_path)
                    
                    # Find insertion point (after function/class definition line)
                    func_line = node.lineno - 1  # Convert to 0-based indexing
                    
                    if func_line < len(lines):
                        # Find the end of the function/class signature
                        insert_line = func_line
                        while insert_line < len(lines) and not lines[insert_line].rstrip().endswith(':'):
                            insert_line += 1
                        
                        if insert_line < len(lines):
                            # Determine indentation
                            next_line_idx = insert_line + 1
                            if next_line_idx < len(lines):
                                next_line = lines[next_line_idx]
                                if next_line.strip():
                                    indent = len(next_line) - len(next_line.lstrip())
                                else:
                                    # Estimate indentation
                                    base_indent = len(lines[func_line]) - len(lines[func_line].lstrip())
                                    indent = base_indent + 4
                            else:
                                base_indent = len(lines[func_line]) - len(lines[func_line].lstrip())
                                indent = base_indent + 4
                            
                            # Insert docstring
                            docstring_lines = [
                                ' ' * indent + '"""',
                                ' ' * indent + docstring,
                                ' ' * indent + '"""'
                            ]
                            
                            # Insert after the function/class definition
                            lines[insert_line + 1:insert_line + 1] = docstring_lines
                            additions += 1
        
        return '\n'.join(lines), additions
    
    def _generate_docstring(self, node: ast.AST, file_path: Path) -> str:
        """Generate appropriate docstring for AST node."""
        if isinstance(node, ast.ClassDef):
            return f"{node.name} class."
        
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Analyze function to generate better docstring
            func_name = node.name
            
            # Check for common patterns
            if func_name.startswith('test_'):
                return f"Test {func_name[5:].replace('_', ' ')}."
            elif func_name.startswith('_'):
                return f"Internal helper function."
            elif 'init' in func_name.lower():
                return f"Initialize {func_name.replace('_', ' ')}."
            elif 'process' in func_name.lower():
                return f"Process data for {func_name.replace('_', ' ')}."
            elif 'get' in func_name.lower():
                return f"Get {func_name[3:].replace('_', ' ')}."
            elif 'set' in func_name.lower():
                return f"Set {func_name[3:].replace('_', ' ')}."
            elif 'create' in func_name.lower():
                return f"Create {func_name.replace('create_', '').replace('_', ' ')}."
            else:
                return f"{func_name.replace('_', ' ').title()} function."
        
        return "Undocumented component."
    
    def _fix_style_issues(self, content: str) -> Tuple[str, int]:
        """Fix basic style issues."""
        lines = content.split('\n')
        modified_lines = []
        fixes_count = 0
        
        for line in lines:
            original_line = line
            
            # Fix trailing whitespace
            line = line.rstrip()
            
            # Fix multiple consecutive blank lines (keep max 2)
            # This will be handled at the content level later
            
            if line != original_line:
                fixes_count += 1
            
            modified_lines.append(line)
        
        # Fix multiple consecutive blank lines
        final_lines = []
        blank_count = 0
        
        for line in modified_lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    final_lines.append(line)
            else:
                blank_count = 0
                final_lines.append(line)
        
        return '\n'.join(final_lines), fixes_count
    
    def run_formatters(self) -> Dict[str, Any]:
        """Run automated code formatters if available."""
        formatter_results = {
            'black_applied': False,
            'isort_applied': False,
            'autopep8_applied': False,
            'errors': []
        }
        
        try:
            # Try to run black (if available)
            try:
                result = subprocess.run(
                    ['python3', '-m', 'black', '--line-length', '120', str(self.project_root)],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    formatter_results['black_applied'] = True
                    logger.info("Black formatter applied successfully")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("Black formatter not available")
        
        except Exception as e:
            formatter_results['errors'].append(f"Formatter error: {e}")
        
        return formatter_results


class ProjectImprover:
    """Comprehensive project improvement orchestrator."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path('/root/repo')
        self.code_improver = CodeImprover(project_root)
        
    def run_comprehensive_improvement(self) -> Dict[str, Any]:
        """Run comprehensive project improvement."""
        logger.info("Starting comprehensive project improvement...")
        
        results = {
            'timestamp': str(time.strftime('%Y-%m-%d %H:%M:%S')),
            'code_quality_improvements': {},
            'formatter_results': {},
            'total_files_processed': 0,
            'total_changes_made': 0,
            'success': False
        }
        
        try:
            # 1. Apply automated code improvements
            code_results = self.code_improver.improve_code_quality()
            results['code_quality_improvements'] = code_results
            results['total_files_processed'] = code_results['files_processed']
            
            # Count total changes
            total_changes = (
                code_results['line_length_fixes'] + 
                code_results['docstring_additions'] +
                code_results['style_fixes']
            )
            results['total_changes_made'] = total_changes
            
            # 2. Run formatters
            formatter_results = self.code_improver.run_formatters()
            results['formatter_results'] = formatter_results
            
            results['success'] = True
            
            # Summary
            logger.info(f"Project improvement complete:")
            logger.info(f"  Files processed: {results['total_files_processed']}")
            logger.info(f"  Total changes: {results['total_changes_made']}")
            logger.info(f"  Line length fixes: {code_results['line_length_fixes']}")
            logger.info(f"  Docstrings added: {code_results['docstring_additions']}")
            logger.info(f"  Style fixes: {code_results['style_fixes']}")
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            logger.error(f"Project improvement failed: {e}")
        
        return results
    
    def save_improvement_report(self, results: Dict[str, Any]):
        """Save improvement report."""
        import time
        report_file = self.project_root / f"improvement_report_{int(time.time())}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Improvement report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


def main():
    """Main execution function."""
    import time
    
    print("🔧 AUTOMATED CODE IMPROVEMENT SYSTEM")
    print("=" * 50)
    
    improver = ProjectImprover()
    results = improver.run_comprehensive_improvement()
    
    # Save report
    improver.save_improvement_report(results)
    
    # Print summary
    if results['success']:
        print(f"\n✅ Improvement Complete!")
        print(f"📊 Summary:")
        print(f"  • Files processed: {results['total_files_processed']}")
        print(f"  • Total changes: {results['total_changes_made']}")
        print(f"  • Line fixes: {results['code_quality_improvements']['line_length_fixes']}")
        print(f"  • Docstrings added: {results['code_quality_improvements']['docstring_additions']}")
        print(f"  • Style fixes: {results['code_quality_improvements']['style_fixes']}")
        
        if results['code_quality_improvements']['changes_made']:
            print(f"\n📝 Changes Applied:")
            for i, change in enumerate(results['code_quality_improvements']['changes_made'][:10], 1):
                print(f"  {i}. {change}")
    else:
        print(f"\n❌ Improvement failed: {results.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)
    
    return 0 if results['success'] else 1


if __name__ == "__main__":
    import time
    sys.exit(main())