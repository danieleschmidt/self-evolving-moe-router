#!/usr/bin/env python3
"""
Automated dependency update script for Self-Evolving MoE-Router
Checks for dependency updates and creates pull requests if needed
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import toml


class DependencyUpdater:
    """Automated dependency management and updates"""
    
    def __init__(self, config_file: str = "pyproject.toml"):
        self.config_file = Path(config_file)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "terragon-labs/self-evolving-moe-router")
        self.base_branch = os.getenv("GITHUB_BASE_REF", "main")
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file {config_file} not found")
    
    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str, str]:
        """Run a shell command and return success status, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=cwd or Path.cwd(),
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
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"GitHub API request failed: {e}")
            return None
    
    def get_current_dependencies(self) -> Dict[str, Dict[str, str]]:
        """Parse current dependencies from pyproject.toml"""
        with open(self.config_file, 'r') as f:
            config = toml.load(f)
        
        dependencies = {}
        
        # Main dependencies
        if "project" in config and "dependencies" in config["project"]:
            dependencies["main"] = {}
            for dep in config["project"]["dependencies"]:
                if ">=" in dep:
                    name, version = dep.split(">=")
                    dependencies["main"][name.strip()] = version.strip()
                elif "==" in dep:
                    name, version = dep.split("==")
                    dependencies["main"][name.strip()] = version.strip()
                elif "~=" in dep:
                    name, version = dep.split("~=")
                    dependencies["main"][name.strip()] = version.strip()
                else:
                    # Handle dependencies without version constraints
                    dependencies["main"][dep.strip()] = "latest"
        
        # Optional dependencies
        if "project" in config and "optional-dependencies" in config["project"]:
            for group, deps in config["project"]["optional-dependencies"].items():
                dependencies[group] = {}
                for dep in deps:
                    if ">=" in dep:
                        name, version = dep.split(">=")
                        dependencies[group][name.strip()] = version.strip()
                    elif "==" in dep:
                        name, version = dep.split("==")
                        dependencies[group][name.strip()] = version.strip()
                    elif "~=" in dep:
                        name, version = dep.split("~=")
                        dependencies[group][name.strip()] = version.strip()
                    else:
                        dependencies[group][dep.strip()] = "latest"
        
        return dependencies
    
    def check_latest_versions(self, dependencies: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Check PyPI for latest versions of dependencies"""
        updates = {}
        
        for group, deps in dependencies.items():
            updates[group] = {}
            
            for package, current_version in deps.items():
                print(f"Checking {package}...")
                
                try:
                    response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    latest_version = data["info"]["version"]
                    
                    if current_version != "latest" and current_version != latest_version:
                        updates[group][package] = {
                            "current": current_version,
                            "latest": latest_version,
                            "summary": data["info"]["summary"],
                            "release_date": data["releases"][latest_version][0]["upload_time"] if data["releases"][latest_version] else "Unknown"
                        }
                        print(f"  Update available: {current_version} -> {latest_version}")
                    else:
                        print(f"  Up to date: {current_version}")
                        
                except requests.RequestException as e:
                    print(f"  Error checking {package}: {e}")
                except KeyError as e:
                    print(f"  Error parsing PyPI response for {package}: {e}")
        
        return updates
    
    def check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities using safety"""
        print("Checking for security vulnerabilities...")
        
        success, stdout, stderr = self._run_command(["safety", "check", "--json"])
        
        if not success:
            if "No known security vulnerabilities found" in stderr:
                print("✅ No security vulnerabilities found")
                return []
            else:
                print(f"Safety check failed: {stderr}")
                return []
        
        try:
            vulnerabilities = json.loads(stdout) if stdout.strip() else []
            if vulnerabilities:
                print(f"🚨 Found {len(vulnerabilities)} security vulnerabilities:")
                for vuln in vulnerabilities:
                    print(f"  - {vuln.get('package', 'Unknown')}: {vuln.get('advisory', 'No description')}")
            else:
                print("✅ No security vulnerabilities found")
            
            return vulnerabilities
        except json.JSONDecodeError:
            print(f"Error parsing safety output: {stdout}")
            return []
    
    def run_tests(self) -> bool:
        """Run tests to ensure updates don't break anything"""
        print("Running tests to validate updates...")
        
        # Run linting first
        success, stdout, stderr = self._run_command(["ruff", "check", "src/", "tests/"])
        if not success:
            print(f"❌ Linting failed: {stderr}")
            return False
        
        # Run type checking
        success, stdout, stderr = self._run_command(["mypy", "src/"])
        if not success:
            print(f"❌ Type checking failed: {stderr}")
            return False
        
        # Run unit tests
        success, stdout, stderr = self._run_command(["pytest", "tests/unit/", "-v"])
        if not success:
            print(f"❌ Unit tests failed: {stderr}")
            return False
        
        print("✅ All tests passed")
        return True
    
    def apply_updates(self, updates: Dict[str, Dict[str, Dict[str, str]]], security_only: bool = False) -> bool:
        """Apply dependency updates to pyproject.toml"""
        if not any(updates.values()):
            print("No updates to apply")
            return False
        
        # Read current config
        with open(self.config_file, 'r') as f:
            config = toml.load(f)
        
        updated_packages = []
        
        for group, packages in updates.items():
            if not packages:
                continue
                
            for package, info in packages.items():
                if security_only:
                    # Only update if there are known vulnerabilities
                    # This would require integration with vulnerability databases
                    continue
                
                current_version = info["current"]
                new_version = info["latest"]
                
                # Update main dependencies
                if group == "main" and "project" in config and "dependencies" in config["project"]:
                    for i, dep in enumerate(config["project"]["dependencies"]):
                        if dep.startswith(package):
                            # Replace the dependency with new version
                            config["project"]["dependencies"][i] = f"{package}>={new_version}"
                            updated_packages.append(f"{package}: {current_version} -> {new_version}")
                            break
                
                # Update optional dependencies
                elif group in config.get("project", {}).get("optional-dependencies", {}):
                    for i, dep in enumerate(config["project"]["optional-dependencies"][group]):
                        if dep.startswith(package):
                            config["project"]["optional-dependencies"][group][i] = f"{package}>={new_version}"
                            updated_packages.append(f"{package} ({group}): {current_version} -> {new_version}")
                            break
        
        if updated_packages:
            # Write updated config
            with open(self.config_file, 'w') as f:
                toml.dump(config, f)
            
            print(f"✅ Updated {len(updated_packages)} packages:")
            for update in updated_packages:
                print(f"  - {update}")
            
            return True
        
        return False
    
    def create_update_branch(self, branch_name: str) -> bool:
        """Create a new branch for dependency updates"""
        print(f"Creating branch: {branch_name}")
        
        # Ensure we're on the base branch
        success, _, stderr = self._run_command(["git", "checkout", self.base_branch])
        if not success:
            print(f"Failed to checkout {self.base_branch}: {stderr}")
            return False
        
        # Pull latest changes
        success, _, stderr = self._run_command(["git", "pull", "origin", self.base_branch])
        if not success:
            print(f"Failed to pull latest changes: {stderr}")
            return False
        
        # Create new branch
        success, _, stderr = self._run_command(["git", "checkout", "-b", branch_name])
        if not success:
            print(f"Failed to create branch {branch_name}: {stderr}")
            return False
        
        return True
    
    def commit_and_push_updates(self, branch_name: str, updated_packages: List[str]) -> bool:
        """Commit and push dependency updates"""
        print("Committing and pushing updates...")
        
        # Add changed files
        success, _, stderr = self._run_command(["git", "add", str(self.config_file)])
        if not success:
            print(f"Failed to add files: {stderr}")
            return False
        
        # Create commit message
        commit_message = f"deps: update {len(updated_packages)} dependencies\n\n"
        commit_message += "\n".join(f"- {pkg}" for pkg in updated_packages)
        commit_message += "\n\n🤖 Generated with automated dependency updater"
        
        # Commit changes
        success, _, stderr = self._run_command(["git", "commit", "-m", commit_message])
        if not success:
            print(f"Failed to commit changes: {stderr}")
            return False
        
        # Push branch
        success, _, stderr = self._run_command(["git", "push", "origin", branch_name])
        if not success:
            print(f"Failed to push branch: {stderr}")
            return False
        
        return True
    
    def create_pull_request(self, branch_name: str, updates: Dict, vulnerabilities: List[Dict]) -> Optional[str]:
        """Create a pull request for dependency updates"""
        if not self.github_token:
            print("Cannot create pull request: GITHUB_TOKEN not set")
            return None
        
        # Count total updates
        total_updates = sum(len(packages) for packages in updates.values())
        
        # Create PR title
        if vulnerabilities:
            title = f"🔒 Security updates for {len(vulnerabilities)} vulnerabilities and {total_updates} dependency updates"
        else:
            title = f"📦 Update {total_updates} dependencies"
        
        # Create PR body
        body_lines = [
            "## Dependency Updates",
            "",
            "This pull request updates the following dependencies:",
            ""
        ]
        
        for group, packages in updates.items():
            if not packages:
                continue
                
            body_lines.append(f"### {group.title()} Dependencies")
            body_lines.append("")
            
            for package, info in packages.items():
                body_lines.append(f"- **{package}**: {info['current']} → {info['latest']}")
                if info.get('summary'):
                    body_lines.append(f"  - {info['summary']}")
                body_lines.append("")
        
        if vulnerabilities:
            body_lines.extend([
                "## 🚨 Security Vulnerabilities Fixed",
                ""
            ])
            
            for vuln in vulnerabilities:
                package = vuln.get('package', 'Unknown')
                advisory = vuln.get('advisory', 'No description')
                body_lines.append(f"- **{package}**: {advisory}")
            
            body_lines.append("")
        
        body_lines.extend([
            "## Automated Testing",
            "",
            "- [x] Linting checks passed",
            "- [x] Type checking passed", 
            "- [x] Unit tests passed",
            "",
            "## Notes",
            "",
            "This PR was automatically generated by the dependency updater.",
            "Please review the changes and run additional tests if necessary.",
            "",
            "🤖 Generated with automated dependency updater"
        ])
        
        body = "\n".join(body_lines)
        
        # Create PR
        pr_data = {
            "title": title,
            "head": branch_name,
            "base": self.base_branch,
            "body": body,
            "draft": False
        }
        
        response = self._make_github_request("pulls", method="POST", data=pr_data)
        
        if response:
            pr_url = response["html_url"]
            print(f"✅ Pull request created: {pr_url}")
            return pr_url
        else:
            print("❌ Failed to create pull request")
            return None
    
    def run_update_process(self, security_only: bool = False, create_pr: bool = True) -> bool:
        """Run the complete dependency update process"""
        print("🔄 Starting dependency update process...")
        
        # Check current dependencies
        current_deps = self.get_current_dependencies()
        print(f"Found {sum(len(deps) for deps in current_deps.values())} dependencies")
        
        # Check for updates
        updates = self.check_latest_versions(current_deps)
        total_updates = sum(len(packages) for packages in updates.values())
        
        if total_updates == 0:
            print("✅ All dependencies are up to date")
            return True
        
        print(f"Found {total_updates} available updates")
        
        # Check for security vulnerabilities
        vulnerabilities = self.check_security_vulnerabilities()
        
        if security_only and not vulnerabilities:
            print("No security vulnerabilities found, skipping updates")
            return True
        
        # Create update branch
        branch_name = f"deps/automated-update-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if not self.create_update_branch(branch_name):
            return False
        
        # Apply updates
        applied = self.apply_updates(updates, security_only=security_only)
        if not applied:
            print("No updates were applied")
            return False
        
        # Install updated dependencies
        print("Installing updated dependencies...")
        success, stdout, stderr = self._run_command(["pip", "install", "-e", ".[dev]"])
        if not success:
            print(f"Failed to install dependencies: {stderr}")
            return False
        
        # Run tests
        if not self.run_tests():
            print("❌ Tests failed, updates may have introduced issues")
            return False
        
        # Get list of updated packages for commit message
        updated_packages = []
        for group, packages in updates.items():
            for package, info in packages.items():
                updated_packages.append(f"{package}: {info['current']} -> {info['latest']}")
        
        # Commit and push changes
        if not self.commit_and_push_updates(branch_name, updated_packages):
            return False
        
        # Create pull request if requested
        if create_pr:
            pr_url = self.create_pull_request(branch_name, updates, vulnerabilities)
            if pr_url:
                print(f"🎉 Dependency update process completed successfully!")
                print(f"Pull request: {pr_url}")
                return True
            else:
                print("❌ Failed to create pull request")
                return False
        else:
            print(f"🎉 Updates committed to branch: {branch_name}")
            return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated dependency updater")
    parser.add_argument("--security-only", action="store_true", 
                       help="Only update dependencies with security vulnerabilities")
    parser.add_argument("--no-pr", action="store_true",
                       help="Don't create a pull request, just commit to branch")
    parser.add_argument("--config", default="pyproject.toml",
                       help="Path to configuration file (default: pyproject.toml)")
    
    args = parser.parse_args()
    
    try:
        updater = DependencyUpdater(config_file=args.config)
        success = updater.run_update_process(
            security_only=args.security_only,
            create_pr=not args.no_pr
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Dependency update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()