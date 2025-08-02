#!/usr/bin/env python3
"""
Automated metrics collection script for Self-Evolving MoE-Router
Collects and updates project metrics from various sources
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class MetricsCollector:
    """Collects and aggregates project metrics from various sources"""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.metrics_data = self._load_metrics_config()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "terragon-labs/self-evolving-moe-router")
        
    def _load_metrics_config(self) -> Dict[str, Any]:
        """Load the metrics configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Metrics config file not found at {self.config_path}")
            return {}
    
    def _save_metrics_config(self) -> None:
        """Save the updated metrics configuration"""
        self.metrics_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        with open(self.config_path, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
        
        print(f"Metrics updated and saved to {self.config_path}")
    
    def _run_command(self, cmd: List[str]) -> Optional[str]:
        """Run a shell command and return output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            return None
    
    def _make_github_request(self, endpoint: str) -> Optional[Dict]:
        """Make a request to GitHub API"""
        if not self.github_token:
            print("Warning: GITHUB_TOKEN not set, skipping GitHub metrics")
            return None
        
        url = f"https://api.github.com/repos/{self.repo_name}/{endpoint}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"GitHub API request failed: {e}")
            return None
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics"""
        metrics = {}
        
        # Test coverage from coverage.xml or pytest
        coverage_file = Path("coverage.xml")
        if coverage_file.exists():
            # Parse coverage from XML file
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    line_rate = float(coverage_elem.get("line-rate", 0))
                    metrics["test_coverage"] = round(line_rate * 100, 2)
            except Exception as e:
                print(f"Error parsing coverage: {e}")
        
        # Code complexity using radon
        complexity_output = self._run_command(["radon", "cc", "src/", "-a"])
        if complexity_output:
            # Parse average complexity from radon output
            lines = complexity_output.split('\n')
            for line in lines:
                if "Average complexity:" in line:
                    try:
                        complexity = float(line.split(':')[1].strip().split()[0])
                        metrics["cyclomatic_complexity"] = complexity
                    except (IndexError, ValueError):
                        pass
        
        # Technical debt using SonarQube API or other tools
        # This would require SonarQube integration
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security vulnerability metrics"""
        metrics = {}
        
        # Run safety check for dependency vulnerabilities
        safety_output = self._run_command(["safety", "check", "--json"])
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                if isinstance(safety_data, list):
                    metrics["dependency_vulnerabilities"] = len(safety_data)
            except json.JSONDecodeError:
                pass
        
        # Run bandit for security issues
        bandit_output = self._run_command(["bandit", "-r", "src/", "-f", "json"])
        if bandit_output:
            try:
                bandit_data = json.loads(bandit_output)
                results = bandit_data.get("results", [])
                
                critical = sum(1 for r in results if r.get("issue_severity") == "HIGH")
                high = sum(1 for r in results if r.get("issue_severity") == "MEDIUM")
                medium = sum(1 for r in results if r.get("issue_severity") == "LOW")
                
                metrics["critical_vulnerabilities"] = critical
                metrics["high_vulnerabilities"] = high
                metrics["medium_vulnerabilities"] = medium
            except json.JSONDecodeError:
                pass
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from benchmarks"""
        metrics = {}
        
        # Check for benchmark results file
        benchmark_file = Path("benchmark-results.json")
        if benchmark_file.exists():
            try:
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                # Extract key performance metrics
                if "inference" in benchmark_data:
                    inference = benchmark_data["inference"]
                    metrics["inference_latency_p95"] = inference.get("latency_p95_ms", 0)
                    metrics["throughput"] = inference.get("throughput_samples_per_sec", 0)
                    metrics["memory_usage"] = inference.get("memory_usage_gb", 0)
                    metrics["gpu_utilization"] = inference.get("gpu_utilization_percent", 0)
                
                if "evolution" in benchmark_data:
                    evolution = benchmark_data["evolution"]
                    metrics["evolution_convergence_rate"] = evolution.get("convergence_rate", 0)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing benchmark results: {e}")
        
        return metrics
    
    def collect_repository_metrics(self) -> Dict[str, Any]:
        """Collect repository health metrics from GitHub"""
        metrics = {}
        
        # Get repository information
        repo_data = self._make_github_request("")
        if repo_data:
            metrics["open_issues"] = repo_data.get("open_issues_count", 0)
        
        # Get pull requests
        pr_data = self._make_github_request("pulls?state=open")
        if pr_data:
            metrics["open_pull_requests"] = len(pr_data)
        
        # Get contributors
        contributors_data = self._make_github_request("contributors")
        if contributors_data:
            metrics["contributor_count"] = len(contributors_data)
        
        # Get commit activity
        commits_data = self._make_github_request("commits?since=" + 
                                                (datetime.now(timezone.utc).replace(days=-7)).isoformat())
        if commits_data:
            metrics["commit_frequency"] = len(commits_data) / 7  # commits per day
        
        return metrics
    
    def collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development process metrics"""
        metrics = {}
        
        # GitHub Actions workflow runs
        workflows_data = self._make_github_request("actions/runs?per_page=100")
        if workflows_data and "workflow_runs" in workflows_data:
            runs = workflows_data["workflow_runs"]
            successful_runs = [r for r in runs if r["conclusion"] == "success"]
            total_runs = len(runs)
            
            if total_runs > 0:
                metrics["build_success_rate"] = round((len(successful_runs) / total_runs) * 100, 2)
        
        # Calculate lead time from PR creation to merge
        merged_prs = self._make_github_request("pulls?state=closed&per_page=50")
        if merged_prs:
            lead_times = []
            for pr in merged_prs:
                if pr.get("merged_at"):
                    created = datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00'))
                    merged = datetime.fromisoformat(pr["merged_at"].replace('Z', '+00:00'))
                    lead_time = (merged - created).total_seconds() / 3600  # hours
                    lead_times.append(lead_time)
            
            if lead_times:
                metrics["lead_time"] = round(sum(lead_times) / len(lead_times), 2)
        
        return metrics
    
    def collect_evolution_metrics(self) -> Dict[str, Any]:
        """Collect evolution algorithm specific metrics"""
        metrics = {}
        
        # Check for evolution logs or results
        evolution_log = Path("logs/evolution.log")
        if evolution_log.exists():
            try:
                with open(evolution_log, 'r') as f:
                    lines = f.readlines()
                
                # Parse the last few lines for recent metrics
                for line in reversed(lines[-100:]):  # Check last 100 lines
                    if "best_fitness" in line.lower():
                        try:
                            # Extract fitness value (assuming JSON log format)
                            import re
                            fitness_match = re.search(r'"best_fitness":\s*([\d.]+)', line)
                            if fitness_match:
                                metrics["best_fitness_achieved"] = float(fitness_match.group(1))
                                break
                        except ValueError:
                            continue
                            
            except FileNotFoundError:
                pass
        
        # Check for evolution state files
        state_file = Path("checkpoints/evolution_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                metrics["population_diversity"] = state_data.get("diversity_score", 0)
                metrics["generations_to_convergence"] = state_data.get("generation", 0)
                metrics["expert_utilization_balance"] = state_data.get("load_balance_score", 0)
                metrics["routing_efficiency"] = state_data.get("routing_efficiency", 0)
                
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        
        return metrics
    
    def update_metrics(self) -> None:
        """Update all metrics in the configuration"""
        if "metrics" not in self.metrics_data:
            self.metrics_data["metrics"] = {}
        
        # Collect metrics from all sources
        collectors = {
            "code_quality": self.collect_code_quality_metrics,
            "security": self.collect_security_metrics,
            "performance": self.collect_performance_metrics,
            "repository": self.collect_repository_metrics,
            "development": self.collect_development_metrics,
            "evolution": self.collect_evolution_metrics,
        }
        
        for category, collector in collectors.items():
            print(f"Collecting {category} metrics...")
            try:
                new_metrics = collector()
                
                if category not in self.metrics_data["metrics"]:
                    self.metrics_data["metrics"][category] = {"metrics": {}}
                
                # Update metrics with new values
                for metric_name, value in new_metrics.items():
                    if metric_name in self.metrics_data["metrics"][category]["metrics"]:
                        old_value = self.metrics_data["metrics"][category]["metrics"][metric_name]["current"]
                        
                        # Determine trend
                        if value > old_value:
                            trend = "improving"
                        elif value < old_value:
                            trend = "declining"
                        else:
                            trend = "stable"
                        
                        self.metrics_data["metrics"][category]["metrics"][metric_name].update({
                            "current": value,
                            "trend": trend,
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        })
                        
                        print(f"  Updated {metric_name}: {old_value} -> {value} ({trend})")
                    else:
                        print(f"  New metric {metric_name}: {value}")
                        
            except Exception as e:
                print(f"Error collecting {category} metrics: {e}")
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check metrics against alert thresholds"""
        alerts = []
        
        if "alerts" not in self.metrics_data or "thresholds" not in self.metrics_data["alerts"]:
            return alerts
        
        thresholds = self.metrics_data["alerts"]["thresholds"]
        
        for severity in ["critical", "warning"]:
            if severity not in thresholds:
                continue
                
            for metric_name, threshold in thresholds[severity].items():
                # Find the metric in the data structure
                for category_data in self.metrics_data["metrics"].values():
                    if "metrics" in category_data and metric_name in category_data["metrics"]:
                        current_value = category_data["metrics"][metric_name]["current"]
                        
                        # Determine if alert should be triggered based on metric type
                        alert_triggered = False
                        if metric_name in ["test_coverage", "build_success_rate"]:
                            # Higher is better
                            alert_triggered = current_value < threshold
                        else:
                            # Lower is better
                            alert_triggered = current_value > threshold
                        
                        if alert_triggered:
                            alerts.append({
                                "severity": severity,
                                "metric": metric_name,
                                "current": current_value,
                                "threshold": threshold,
                                "message": f"{metric_name} is {current_value}, threshold is {threshold}"
                            })
        
        return alerts
    
    def generate_report(self) -> str:
        """Generate a summary report of current metrics"""
        report_lines = [
            "# Project Metrics Report",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Repository: {self.repo_name}",
            "",
            "## Key Metrics Summary",
            ""
        ]
        
        for category, data in self.metrics_data.get("metrics", {}).items():
            if "metrics" not in data:
                continue
                
            report_lines.append(f"### {category.title().replace('_', ' ')}")
            report_lines.append("")
            
            for metric_name, metric_data in data["metrics"].items():
                current = metric_data.get("current", "N/A")
                target = metric_data.get("target", "N/A")
                trend = metric_data.get("trend", "unknown")
                unit = metric_data.get("unit", "")
                
                trend_emoji = {"improving": "↗️", "declining": "↘️", "stable": "➡️"}.get(trend, "❓")
                
                report_lines.append(f"- **{metric_name.replace('_', ' ').title()}**: {current} {unit} "
                                  f"(target: {target} {unit}) {trend_emoji}")
            
            report_lines.append("")
        
        # Add alerts section
        alerts = self.check_alerts()
        if alerts:
            report_lines.extend([
                "## Alerts",
                ""
            ])
            
            for alert in alerts:
                severity_emoji = {"critical": "🚨", "warning": "⚠️"}.get(alert["severity"], "ℹ️")
                report_lines.append(f"- {severity_emoji} **{alert['severity'].title()}**: {alert['message']}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main entry point"""
    collector = MetricsCollector()
    
    # Update metrics
    collector.update_metrics()
    
    # Save updated metrics
    collector._save_metrics_config()
    
    # Generate and display report
    report = collector.generate_report()
    print("\n" + "="*50)
    print(report)
    
    # Check for alerts
    alerts = collector.check_alerts()
    if alerts:
        print(f"\n🚨 Found {len(alerts)} alerts!")
        for alert in alerts:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
        sys.exit(1)
    else:
        print("\n✅ All metrics within acceptable ranges")


if __name__ == "__main__":
    main()