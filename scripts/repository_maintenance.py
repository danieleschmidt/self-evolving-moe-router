#!/usr/bin/env python3
"""
Repository maintenance automation script for Self-Evolving MoE-Router
Handles routine maintenance tasks like cleanup, optimization, and health checks
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class RepositoryMaintenance:
    """Automated repository maintenance and cleanup"""
    
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "terragon-labs/self-evolving-moe-router")
        self.repo_path = Path.cwd()
        
    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str, str]:
        """Run a shell command and return success status, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=cwd or self.repo_path,
                check=False
            )
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def _make_github_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
        """Make a request to GitHub API"""
        if not self.github_token:
            print("Warning: GITHUB_TOKEN not set, GitHub operations will be skipped")
            return None
        
        url = f"https://api.github.com/repos/{self.repo_name}/{endpoint}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.RequestException as e:
            print(f"GitHub API request failed: {e}")
            return None
    
    def clean_cache_files(self) -> int:
        """Clean Python cache files and directories"""
        print("🧹 Cleaning cache files...")
        
        cleaned_count = 0
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo", 
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/htmlcov",
            "**/.coverage",
            "**/coverage.xml",
            "**/.tox",
            "**/build",
            "**/dist",
            "**/*.egg-info"
        ]
        
        for pattern in cache_patterns:
            for path in self.repo_path.glob(pattern):
                if path.exists():
                    try:
                        if path.is_file():
                            path.unlink()
                            cleaned_count += 1
                        elif path.is_dir():
                            shutil.rmtree(path)
                            cleaned_count += 1
                        print(f"  Removed: {path}")
                    except Exception as e:
                        print(f"  Failed to remove {path}: {e}")
        
        print(f"✅ Cleaned {cleaned_count} cache files/directories")
        return cleaned_count
    
    def clean_old_logs(self, days: int = 30) -> int:
        """Clean log files older than specified days"""
        print(f"🗂️  Cleaning log files older than {days} days...")
        
        logs_dir = self.repo_path / "logs"
        if not logs_dir.exists():
            print("  No logs directory found")
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for log_file in logs_dir.rglob("*.log"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    cleaned_count += 1
                    print(f"  Removed old log: {log_file}")
            except Exception as e:
                print(f"  Failed to process {log_file}: {e}")
        
        print(f"✅ Cleaned {cleaned_count} old log files")
        return cleaned_count
    
    def clean_old_checkpoints(self, keep_count: int = 10) -> int:
        """Clean old model checkpoints, keeping only the most recent ones"""
        print(f"💾 Cleaning old checkpoints, keeping {keep_count} most recent...")
        
        checkpoints_dir = self.repo_path / "checkpoints"
        if not checkpoints_dir.exists():
            print("  No checkpoints directory found")
            return 0
        
        # Find all checkpoint files
        checkpoint_files = []
        for pattern in ["*.pt", "*.pth", "*.ckpt", "*.safetensors"]:
            checkpoint_files.extend(checkpoints_dir.glob(pattern))
        
        if len(checkpoint_files) <= keep_count:
            print(f"  Only {len(checkpoint_files)} checkpoints found, no cleanup needed")
            return 0
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        cleaned_count = 0
        for checkpoint in checkpoint_files[keep_count:]:
            try:
                file_size = checkpoint.stat().st_size / (1024 * 1024)  # MB
                checkpoint.unlink()
                cleaned_count += 1
                print(f"  Removed old checkpoint: {checkpoint.name} ({file_size:.1f} MB)")
            except Exception as e:
                print(f"  Failed to remove {checkpoint}: {e}")
        
        print(f"✅ Cleaned {cleaned_count} old checkpoints")
        return cleaned_count
    
    def optimize_git_repository(self) -> bool:
        """Optimize Git repository by running maintenance commands"""
        print("🔧 Optimizing Git repository...")
        
        git_commands = [
            (["git", "gc", "--aggressive"], "Garbage collection"),
            (["git", "prune"], "Pruning unreachable objects"),
            (["git", "repack", "-a", "-d"], "Repacking objects"),
        ]
        
        success_count = 0
        for cmd, description in git_commands:
            print(f"  Running: {description}")
            success, stdout, stderr = self._run_command(cmd)
            if success:
                success_count += 1
                print(f"    ✅ {description} completed")
            else:
                print(f"    ❌ {description} failed: {stderr}")
        
        print(f"✅ Git optimization completed ({success_count}/{len(git_commands)} commands successful)")
        return success_count == len(git_commands)
    
    def update_gitignore(self) -> bool:
        """Update .gitignore with common patterns"""
        print("📝 Updating .gitignore...")
        
        gitignore_path = self.repo_path / ".gitignore"
        
        # Common patterns to ensure are in .gitignore
        essential_patterns = [
            # Python
            "__pycache__/",
            "*.py[cod]",
            "*$py.class",
            "*.so",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "MANIFEST",
            
            # Virtual environments
            ".env",
            ".venv",
            "env/",
            "venv/",
            "ENV/",
            "env.bak/",
            "venv.bak/",
            
            # IDE
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "*~",
            
            # OS
            ".DS_Store",
            ".DS_Store?",
            "._*",
            ".Spotlight-V100",
            ".Trashes",
            "ehthumbs.db",
            "Thumbs.db",
            
            # Testing and coverage
            ".coverage",
            ".pytest_cache/",
            ".tox/",
            ".nox/",
            "htmlcov/",
            ".cache",
            "nosetests.xml",
            "coverage.xml",
            "*.cover",
            "*.py,cover",
            ".hypothesis/",
            
            # Jupyter
            ".ipynb_checkpoints",
            
            # ML/AI specific
            "*.pt",
            "*.pth", 
            "*.ckpt",
            "*.safetensors",
            "wandb/",
            "tensorboard_logs/",
            "mlruns/",
            "checkpoints/",
            "data/",
            "datasets/",
            "logs/",
            "results/",
            
            # Temporary files
            "tmp/",
            "temp/",
            ".tmp/",
            "*.log",
        ]
        
        # Read current .gitignore
        existing_patterns = set()
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        existing_patterns.add(line)
        
        # Find patterns to add
        new_patterns = [pattern for pattern in essential_patterns if pattern not in existing_patterns]
        
        if new_patterns:
            with open(gitignore_path, 'a') as f:
                f.write(f"\n# Added by maintenance script - {datetime.now().strftime('%Y-%m-%d')}\n")
                for pattern in new_patterns:
                    f.write(f"{pattern}\n")
            
            print(f"  Added {len(new_patterns)} new patterns to .gitignore")
            return True
        else:
            print("  .gitignore is up to date")
            return False
    
    def check_large_files(self, size_mb: int = 100) -> List[Path]:
        """Find large files that might need to be excluded"""
        print(f"🔍 Checking for files larger than {size_mb} MB...")
        
        large_files = []
        size_bytes = size_mb * 1024 * 1024
        
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                try:
                    if file_path.stat().st_size > size_bytes:
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        large_files.append(file_path)
                        print(f"  Large file found: {file_path} ({file_size_mb:.1f} MB)")
                except OSError:
                    # Skip files we can't access
                    continue
        
        if large_files:
            print(f"⚠️  Found {len(large_files)} large files")
            print("  Consider adding them to .gitignore or using Git LFS")
        else:
            print("✅ No large files found")
        
        return large_files
    
    def clean_merged_branches(self, dry_run: bool = False) -> int:
        """Clean up merged remote branches"""
        print("🌿 Cleaning merged branches...")
        
        if not self.github_token:
            print("  Skipping branch cleanup: GITHUB_TOKEN not set")
            return 0
        
        # Get all branches
        branches_data = self._make_github_request("branches?per_page=100")
        if not branches_data:
            print("  Failed to get branches from GitHub")
            return 0
        
        # Get merged pull requests
        merged_prs = self._make_github_request("pulls?state=closed&per_page=100")
        if not merged_prs:
            print("  Failed to get merged PRs from GitHub")
            return 0
        
        # Find branches from merged PRs that can be deleted
        merged_branches = set()
        for pr in merged_prs:
            if pr.get("merged_at") and pr.get("head", {}).get("ref"):
                branch_name = pr["head"]["ref"]
                # Don't delete main/master branches or development branches
                if branch_name not in ["main", "master", "develop", "dev"]:
                    merged_branches.add(branch_name)
        
        deleted_count = 0
        for branch in branches_data:
            branch_name = branch["name"]
            
            if branch_name in merged_branches:
                if dry_run:
                    print(f"  Would delete merged branch: {branch_name}")
                    deleted_count += 1
                else:
                    # Delete the branch
                    delete_result = self._make_github_request(f"git/refs/heads/{branch_name}", method="DELETE")
                    if delete_result is not None:  # Success returns empty response
                        print(f"  Deleted merged branch: {branch_name}")
                        deleted_count += 1
                    else:
                        print(f"  Failed to delete branch: {branch_name}")
        
        action = "would delete" if dry_run else "deleted"
        print(f"✅ {action.title()} {deleted_count} merged branches")
        return deleted_count
    
    def update_issue_labels(self) -> bool:
        """Ensure standard issue labels exist"""
        print("🏷️  Updating issue labels...")
        
        if not self.github_token:
            print("  Skipping label update: GITHUB_TOKEN not set")
            return False
        
        # Standard labels to ensure exist
        standard_labels = [
            {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
            {"name": "enhancement", "color": "a2eeef", "description": "New feature or request"},
            {"name": "documentation", "color": "0075ca", "description": "Improvements or additions to documentation"},
            {"name": "good first issue", "color": "7057ff", "description": "Good for newcomers"},
            {"name": "help wanted", "color": "008672", "description": "Extra attention is needed"},
            {"name": "invalid", "color": "e4e669", "description": "This doesn't seem right"},
            {"name": "question", "color": "d876e3", "description": "Further information is requested"},
            {"name": "wontfix", "color": "ffffff", "description": "This will not be worked on"},
            {"name": "priority:high", "color": "b60205", "description": "High priority issue"},
            {"name": "priority:medium", "color": "fbca04", "description": "Medium priority issue"},
            {"name": "priority:low", "color": "0e8a16", "description": "Low priority issue"},
            {"name": "type:security", "color": "ee0701", "description": "Security related issue"},
            {"name": "type:performance", "color": "1d76db", "description": "Performance related issue"},
            {"name": "dependencies", "color": "0366d6", "description": "Pull requests that update a dependency file"},
        ]
        
        # Get existing labels
        existing_labels = self._make_github_request("labels")
        if existing_labels is None:
            print("  Failed to get existing labels")
            return False
        
        existing_label_names = {label["name"] for label in existing_labels}
        
        created_count = 0
        for label in standard_labels:
            if label["name"] not in existing_label_names:
                result = self._make_github_request("labels", method="POST", data=label)
                if result:
                    print(f"  Created label: {label['name']}")
                    created_count += 1
                else:
                    print(f"  Failed to create label: {label['name']}")
        
        print(f"✅ Created {created_count} new labels")
        return True
    
    def generate_maintenance_report(self) -> str:
        """Generate a maintenance report"""
        report_lines = [
            "# Repository Maintenance Report",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Repository: {self.repo_name}",
            "",
            "## Maintenance Tasks Completed",
            ""
        ]
        
        # This would be expanded to include results from all maintenance tasks
        # For now, just a placeholder structure
        
        return "\n".join(report_lines)
    
    def run_maintenance(self, cleanup_logs_days: int = 30, keep_checkpoints: int = 10, 
                       cleanup_branches: bool = False, dry_run: bool = False) -> bool:
        """Run complete maintenance routine"""
        print("🔧 Starting repository maintenance...")
        
        success_count = 0
        total_tasks = 0
        
        # Cache cleanup
        total_tasks += 1
        try:
            self.clean_cache_files()
            success_count += 1
        except Exception as e:
            print(f"❌ Cache cleanup failed: {e}")
        
        # Log cleanup
        total_tasks += 1
        try:
            self.clean_old_logs(cleanup_logs_days)
            success_count += 1
        except Exception as e:
            print(f"❌ Log cleanup failed: {e}")
        
        # Checkpoint cleanup
        total_tasks += 1
        try:
            self.clean_old_checkpoints(keep_checkpoints)
            success_count += 1
        except Exception as e:
            print(f"❌ Checkpoint cleanup failed: {e}")
        
        # Git optimization
        total_tasks += 1
        try:
            if self.optimize_git_repository():
                success_count += 1
        except Exception as e:
            print(f"❌ Git optimization failed: {e}")
        
        # Update .gitignore
        total_tasks += 1
        try:
            self.update_gitignore()
            success_count += 1
        except Exception as e:
            print(f"❌ .gitignore update failed: {e}")
        
        # Check for large files
        try:
            self.check_large_files()
        except Exception as e:
            print(f"❌ Large file check failed: {e}")
        
        # Clean merged branches (if requested)
        if cleanup_branches:
            total_tasks += 1
            try:
                self.clean_merged_branches(dry_run=dry_run)
                success_count += 1
            except Exception as e:
                print(f"❌ Branch cleanup failed: {e}")
        
        # Update issue labels
        total_tasks += 1
        try:
            if self.update_issue_labels():
                success_count += 1
        except Exception as e:
            print(f"❌ Label update failed: {e}")
        
        print(f"\n✅ Maintenance completed: {success_count}/{total_tasks} tasks successful")
        
        return success_count == total_tasks


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository maintenance automation")
    parser.add_argument("--cleanup-logs-days", type=int, default=30,
                       help="Clean log files older than N days (default: 30)")
    parser.add_argument("--keep-checkpoints", type=int, default=10,
                       help="Keep N most recent checkpoints (default: 10)")
    parser.add_argument("--cleanup-branches", action="store_true",
                       help="Clean up merged remote branches")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    try:
        maintenance = RepositoryMaintenance()
        success = maintenance.run_maintenance(
            cleanup_logs_days=args.cleanup_logs_days,
            keep_checkpoints=args.keep_checkpoints,
            cleanup_branches=args.cleanup_branches,
            dry_run=args.dry_run
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Repository maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()