#!/usr/bin/env python3
"""
TERRAGON Production Health Check
Comprehensive health monitoring for Self-Evolving MoE-Router
"""

import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional


class TERRAGONHealthCheck:
    """Production health check system for TERRAGON."""
    
    def __init__(self, config_path: str = "/app/config/production.json"):
        self.config = self.load_config(config_path)
        self.app_root = Path("/app")
        self.health_status = {"overall": "healthy", "checks": {}}
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load production configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "server": {"port": 8080},
                "monitoring": {"health_check_interval": 30}
            }
    
    def check_api_endpoint(self) -> bool:
        """Check if API endpoint is responsive."""
        try:
            port = self.config.get("server", {}).get("port", 8080)
            timeout = 5
            
            # Try health endpoint first
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
                if response.status_code == 200:
                    self.health_status["checks"]["api_endpoint"] = {"status": "healthy", "response_time": response.elapsed.total_seconds()}
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # Try root endpoint as fallback
            response = requests.get(f"http://localhost:{port}/", timeout=timeout)
            healthy = response.status_code in [200, 404]  # 404 is acceptable for root
            
            self.health_status["checks"]["api_endpoint"] = {
                "status": "healthy" if healthy else "unhealthy",
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code
            }
            return healthy
            
        except Exception as e:
            self.health_status["checks"]["api_endpoint"] = {"status": "unhealthy", "error": str(e)}
            return False
    
    def check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            result = subprocess.run(['df', '/app'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                available = int(parts[3])  # Available space in KB
                used_percent = int(parts[4].rstrip('%'))
                
                healthy = used_percent < 90  # Fail if > 90% used
                
                self.health_status["checks"]["disk_space"] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "available_kb": available,
                    "used_percent": used_percent
                }
                return healthy
            
        except Exception as e:
            self.health_status["checks"]["disk_space"] = {"status": "error", "error": str(e)}
        
        return True  # Don't fail health check on disk check errors
    
    def check_memory_usage(self) -> bool:
        """Check memory usage."""
        try:
            result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                total = int(parts[1])
                available = int(parts[6]) if len(parts) > 6 else int(parts[3])
                
                used_percent = ((total - available) / total) * 100
                healthy = used_percent < 95  # Fail if > 95% used
                
                self.health_status["checks"]["memory"] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "total_mb": total,
                    "available_mb": available,
                    "used_percent": round(used_percent, 1)
                }
                return healthy
                
        except Exception as e:
            self.health_status["checks"]["memory"] = {"status": "error", "error": str(e)}
        
        return True  # Don't fail health check on memory check errors
    
    def check_critical_files(self) -> bool:
        """Check that critical application files exist."""
        critical_files = [
            "high_performance_evolution.py",
            "quality_gates_improved.py", 
            "research_standalone.py"
        ]
        
        all_exist = True
        file_status = {}
        
        for file_name in critical_files:
            file_path = self.app_root / file_name
            exists = file_path.exists()
            all_exist = all_exist and exists
            
            file_status[file_name] = {
                "exists": exists,
                "size": file_path.stat().st_size if exists else 0
            }
        
        self.health_status["checks"]["critical_files"] = {
            "status": "healthy" if all_exist else "unhealthy",
            "files": file_status
        }
        
        return all_exist
    
    def check_python_environment(self) -> bool:
        """Check Python environment and imports."""
        try:
            # Test critical imports
            import_tests = [
                ("json", "import json"),
                ("time", "import time"),
                ("pathlib", "from pathlib import Path"),
                ("typing", "from typing import Dict, List, Any"),
                ("dataclasses", "from dataclasses import dataclass")
            ]
            
            results = {}
            all_passed = True
            
            for name, import_stmt in import_tests:
                try:
                    exec(import_stmt)
                    results[name] = {"status": "ok"}
                except Exception as e:
                    results[name] = {"status": "failed", "error": str(e)}
                    all_passed = False
            
            self.health_status["checks"]["python_environment"] = {
                "status": "healthy" if all_passed else "unhealthy",
                "imports": results,
                "python_version": sys.version
            }
            
            return all_passed
            
        except Exception as e:
            self.health_status["checks"]["python_environment"] = {"status": "error", "error": str(e)}
            return False
    
    def run_comprehensive_health_check(self) -> bool:
        """Run all health checks and return overall status."""
        
        checks = [
            ("API Endpoint", self.check_api_endpoint),
            ("Disk Space", self.check_disk_space),
            ("Memory Usage", self.check_memory_usage),
            ("Critical Files", self.check_critical_files),
            ("Python Environment", self.check_python_environment)
        ]
        
        results = []
        for name, check_func in checks:
            try:
                result = check_func()
                results.append(result)
                print(f"✅ {name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results.append(False)
                print(f"❌ {name}: ERROR - {e}")
        
        # Overall health - all critical checks must pass
        overall_healthy = all(results)
        self.health_status["overall"] = "healthy" if overall_healthy else "unhealthy"
        self.health_status["timestamp"] = time.time()
        self.health_status["summary"] = {
            "total_checks": len(results),
            "passed": sum(results),
            "failed": len(results) - sum(results)
        }
        
        # Save health status
        try:
            with open("/app/logs/health_status.json", "w") as f:
                json.dump(self.health_status, f, indent=2)
        except Exception:
            pass  # Don't fail if we can't save status
        
        return overall_healthy


def main():
    """Main health check execution."""
    print("🔍 TERRAGON Health Check - Production System Validation")
    
    health_checker = TERRAGONHealthCheck()
    
    try:
        is_healthy = health_checker.run_comprehensive_health_check()
        
        print("\n" + "="*60)
        if is_healthy:
            print("✅ TERRAGON SYSTEM: HEALTHY")
            print("All health checks passed successfully")
            exit_code = 0
        else:
            print("❌ TERRAGON SYSTEM: UNHEALTHY") 
            print("One or more health checks failed")
            exit_code = 1
            
        print("="*60)
        
        # Print summary
        summary = health_checker.health_status.get("summary", {})
        print(f"Health Checks: {summary.get('passed', 0)}/{summary.get('total_checks', 0)} passed")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ TERRAGON HEALTH CHECK FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()