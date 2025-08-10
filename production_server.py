#!/usr/bin/env python3
"""
TERRAGON Production Server
Self-Evolving MoE-Router Production API Server
"""

import sys
import json
import time
import asyncio
import logging
import argparse
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

# Production dependencies (would be installed in container)
try:
    from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    # Fallback for lightweight testing
    FASTAPI_AVAILABLE = False


# Import TERRAGON systems
try:
    from high_performance_evolution import HighPerformanceEvolutionSystem, EvolutionConfig
    from quality_gates_improved import QualityGateExecutor
    from research_standalone import TERRAGONResearchExecution
    TERRAGON_SYSTEMS_AVAILABLE = True
except ImportError:
    TERRAGON_SYSTEMS_AVAILABLE = False


@dataclass
class ServerConfig:
    """Production server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    environment: str = "production"
    log_level: str = "INFO"
    max_connections: int = 1000
    keepalive_timeout: int = 65
    graceful_timeout: int = 30
    enable_cors: bool = True
    enable_gzip: bool = True
    enable_metrics: bool = True
    config_file: Optional[str] = None


class ProductionMetrics:
    """Production metrics collection."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.evolution_runs = 0
        self.routing_latencies = []
        self.start_time = time.time()
        
    def record_request(self, latency: float, status_code: int):
        """Record API request metrics."""
        self.request_count += 1
        if status_code >= 400:
            self.error_count += 1
        self.routing_latencies.append(latency)
        
        # Keep only last 1000 latencies
        if len(self.routing_latencies) > 1000:
            self.routing_latencies = self.routing_latencies[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        uptime = time.time() - self.start_time
        avg_latency = sum(self.routing_latencies) / len(self.routing_latencies) if self.routing_latencies else 0
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_latency_ms": avg_latency * 1000,
            "evolution_runs": self.evolution_runs,
            "requests_per_second": self.request_count / max(uptime, 1)
        }


class TERRAGONProductionServer:
    """TERRAGON Production Server."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.metrics = ProductionMetrics()
        self.evolution_system = None
        self.research_system = None
        self.quality_executor = None
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - TERRAGON - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("terragon.production")
        
        # Initialize systems if available
        if TERRAGON_SYSTEMS_AVAILABLE:
            self._initialize_terragon_systems()
    
    def _initialize_terragon_systems(self):
        """Initialize TERRAGON core systems."""
        try:
            self.logger.info("Initializing TERRAGON core systems")
            
            # High-performance evolution system
            evolution_config = EvolutionConfig(
                population_size=50,
                generations=100,
                num_experts=8,
                input_dim=512,
                hidden_dim=256,
                output_dim=128
            )
            self.evolution_system = HighPerformanceEvolutionSystem(evolution_config)
            
            # Quality gates executor
            self.quality_executor = QualityGateExecutor(Path.cwd())
            
            # Research execution system
            self.research_system = TERRAGONResearchExecution()
            
            self.logger.info("✅ TERRAGON systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TERRAGON systems: {e}")
    
    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_path}: {e}")
            return {}
    
    def create_fastapi_app(self) -> "FastAPI":
        """Create FastAPI application."""
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available - install production dependencies")
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            self.logger.info("🚀 Starting TERRAGON Production Server")
            self.running = True
            yield
            # Shutdown
            self.logger.info("🛑 Shutting down TERRAGON Production Server")
            self.running = False
        
        app = FastAPI(
            title="TERRAGON Self-Evolving MoE-Router",
            description="Production API for TERRAGON Self-Evolving Mixture of Experts Router",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Middleware
        if self.config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        if self.config.enable_gzip:
            app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request timing middleware
        @app.middleware("http")
        async def timing_middleware(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_request(process_time, response.status_code)
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        # Routes
        self._setup_routes(app)
        
        return app
    
    def _setup_routes(self, app: "FastAPI"):
        """Setup API routes."""
        
        @app.get("/")
        async def root():
            return {
                "service": "TERRAGON Self-Evolving MoE-Router",
                "version": "1.0.0",
                "status": "running",
                "environment": self.config.environment
            }
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            health_status = {
                "status": "healthy" if self.running else "unhealthy",
                "timestamp": time.time(),
                "uptime": time.time() - self.metrics.start_time,
                "systems": {}
            }
            
            # Check system availability
            if TERRAGON_SYSTEMS_AVAILABLE:
                health_status["systems"]["evolution"] = "available"
                health_status["systems"]["research"] = "available"
                health_status["systems"]["quality_gates"] = "available"
            else:
                health_status["systems"]["warning"] = "TERRAGON systems not fully available"
            
            return health_status
        
        @app.get("/ready")
        async def readiness_check():
            """Readiness check endpoint."""
            ready = self.running and TERRAGON_SYSTEMS_AVAILABLE
            if not ready:
                raise HTTPException(status_code=503, detail="Service not ready")
            
            return {"status": "ready", "timestamp": time.time()}
        
        @app.get("/metrics")
        async def get_metrics():
            """Prometheus-style metrics endpoint."""
            if not self.config.enable_metrics:
                raise HTTPException(status_code=404, detail="Metrics disabled")
            
            metrics = self.metrics.get_metrics()
            
            # Simple Prometheus format
            prometheus_metrics = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    prometheus_metrics.append(f"terragon_{key} {value}")
            
            return {"metrics": metrics, "prometheus": "\n".join(prometheus_metrics)}
        
        @app.post("/evolution/start")
        async def start_evolution(background_tasks: BackgroundTasks):
            """Start evolution process."""
            if not self.evolution_system:
                raise HTTPException(status_code=503, detail="Evolution system not available")
            
            # Start evolution in background
            background_tasks.add_task(self._run_evolution)
            
            return {"message": "Evolution started", "timestamp": time.time()}
        
        @app.get("/evolution/status")
        async def evolution_status():
            """Get evolution status."""
            if not self.evolution_system:
                raise HTTPException(status_code=503, detail="Evolution system not available")
            
            return {
                "status": "Evolution system available",
                "runs_completed": self.metrics.evolution_runs
            }
        
        @app.post("/research/execute")
        async def execute_research(background_tasks: BackgroundTasks):
            """Execute research mode."""
            if not self.research_system:
                raise HTTPException(status_code=503, detail="Research system not available")
            
            background_tasks.add_task(self._run_research)
            
            return {"message": "Research execution started", "timestamp": time.time()}
        
        @app.post("/quality-gates/run")
        async def run_quality_gates():
            """Run quality gates validation."""
            if not self.quality_executor:
                raise HTTPException(status_code=503, detail="Quality gates system not available")
            
            try:
                report = self.quality_executor.execute_all_gates()
                return {
                    "overall_passed": report.overall_passed,
                    "total_score": report.total_score,
                    "summary": report.execution_summary
                }
            except Exception as e:
                self.logger.error(f"Quality gates execution failed: {e}")
                raise HTTPException(status_code=500, detail=f"Quality gates failed: {str(e)}")
        
        # Route simulation endpoint
        @app.post("/route")
        async def route_request(request_data: dict):
            """Simulate MoE routing."""
            start_time = time.time()
            
            # Simulate routing logic
            await asyncio.sleep(0.01)  # Simulate processing time
            
            processing_time = time.time() - start_time
            
            return {
                "routed_to": "expert_1",
                "confidence": 0.95,
                "processing_time_ms": processing_time * 1000,
                "request_id": f"req_{int(time.time()*1000)}"
            }
    
    async def _run_evolution(self):
        """Background evolution execution."""
        try:
            self.logger.info("Starting evolution run")
            
            if self.evolution_system:
                # Simulate evolution run
                await asyncio.sleep(10)  # Simulate evolution time
                self.metrics.evolution_runs += 1
                
            self.logger.info("Evolution run completed")
            
        except Exception as e:
            self.logger.error(f"Evolution run failed: {e}")
    
    async def _run_research(self):
        """Background research execution."""
        try:
            self.logger.info("Starting research execution")
            
            if self.research_system:
                # Simulate research execution
                await asyncio.sleep(30)  # Simulate research time
                
            self.logger.info("Research execution completed")
            
        except Exception as e:
            self.logger.error(f"Research execution failed: {e}")
    
    def run_production_server(self):
        """Run production server with FastAPI."""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI not available - cannot start production server")
            self.run_simple_server()
            return
        
        app = self.create_fastapi_app()
        
        # Configure uvicorn
        uvicorn_config = {
            "host": self.config.host,
            "port": self.config.port,
            "workers": 1 if self.config.environment == "development" else self.config.workers,
            "log_level": self.config.log_level.lower(),
            "access_log": True,
            "loop": "auto",
            "lifespan": "on"
        }
        
        self.logger.info(f"Starting TERRAGON production server on {self.config.host}:{self.config.port}")
        uvicorn.run(app, **uvicorn_config)
    
    def run_simple_server(self):
        """Run simple HTTP server for testing."""
        import http.server
        import socketserver
        
        class TERRAGONHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = json.dumps({"status": "healthy", "timestamp": time.time()})
                    self.wfile.write(response.encode())
                else:
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    response = json.dumps({
                        "service": "TERRAGON Self-Evolving MoE-Router",
                        "status": "running (simple mode)",
                        "message": "Production dependencies not available"
                    })
                    self.wfile.write(response.encode())
        
        with socketserver.TCPServer(("", self.config.port), TERRAGONHandler) as httpd:
            self.logger.info(f"Starting simple TERRAGON server on port {self.config.port}")
            httpd.serve_forever()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main production server entry point."""
    parser = argparse.ArgumentParser(description="TERRAGON Production Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--env", default="production", help="Environment (production/development)")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create server configuration
    config = ServerConfig(
        host=args.host,
        port=args.port,
        workers=args.workers,
        environment=args.env,
        log_level=args.log_level,
        config_file=args.config
    )
    
    # Load additional config from file if provided
    if args.config and Path(args.config).exists():
        file_config = {}
        try:
            with open(args.config, 'r') as f:
                file_config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
        
        # Update config with file values
        for key, value in file_config.get("server", {}).items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    print("🎯 TERRAGON Self-Evolving MoE-Router - Production Server")
    print(f"Environment: {config.environment}")
    print(f"Host: {config.host}:{config.port}")
    print(f"Workers: {config.workers}")
    print(f"FastAPI Available: {FASTAPI_AVAILABLE}")
    print(f"TERRAGON Systems: {TERRAGON_SYSTEMS_AVAILABLE}")
    print("="*60)
    
    # Create and run server
    server = TERRAGONProductionServer(config)
    
    try:
        server.run_production_server()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()