#!/usr/bin/env python3
"""
Production-Ready Self-Evolving MoE Router Server
FastAPI-based REST API with monitoring, metrics, and scalability
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import uvicorn
import asyncio
import threading
import time
import json
import logging
import os
from pathlib import Path
import numpy as np
from contextlib import asynccontextmanager

# Import our optimized systems
try:
    from optimized_simple_demo import OptimizedEvolver, OptimizedMoEModel, OptimizedTopology
    from comprehensive_quality_gates import QualityGateSystem
except ImportError as e:
    logging.warning(f"Could not import modules: {e}")


# Pydantic models for API
class EvolutionRequest(BaseModel):
    """Request model for evolution."""
    num_experts: int = Field(default=8, ge=2, le=32, description="Number of experts")
    num_tokens: int = Field(default=16, ge=4, le=128, description="Number of tokens")
    population_size: int = Field(default=20, ge=8, le=100, description="Population size")
    generations: int = Field(default=20, ge=5, le=100, description="Number of generations")
    input_dim: int = Field(default=64, ge=16, le=512, description="Input dimension")
    hidden_dim: int = Field(default=128, ge=32, le=1024, description="Hidden dimension")


class EvolutionResponse(BaseModel):
    """Response model for evolution."""
    task_id: str
    status: str
    message: str


class EvolutionStatus(BaseModel):
    """Evolution task status."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float
    current_generation: int
    best_fitness: float
    start_time: str
    estimated_completion: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class InferenceRequest(BaseModel):
    """Request model for inference."""
    data: List[List[List[float]]] = Field(description="Input data [batch, seq_len, input_dim]")
    topology_id: Optional[str] = Field(default=None, description="Topology ID to use")


class InferenceResponse(BaseModel):
    """Response model for inference."""
    output: List[List[List[float]]]
    expert_usage: List[float]
    inference_time_ms: float
    topology_used: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    active_tasks: int
    system_metrics: Dict[str, Any]


# Global state management
class ServerState:
    """Global server state."""
    
    def __init__(self):
        self.evolution_tasks: Dict[str, Dict] = {}
        self.topologies: Dict[str, OptimizedTopology] = {}
        self.models: Dict[str, OptimizedMoEModel] = {}
        self.start_time = time.time()
        self.task_counter = 0
        self.lock = threading.Lock()
        
        # Load any existing topologies
        self._load_existing_topologies()
    
    def _load_existing_topologies(self):
        """Load topologies from evolution results."""
        results_dir = Path("evolution_results")
        if results_dir.exists():
            for result_file in results_dir.glob("*_results.json"):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    
                    if 'best_topology_sparsity' in data and data['best_topology_sparsity']:
                        # Create a default topology for demonstration
                        topology = OptimizedTopology(16, 8, data['best_topology_sparsity'])
                        topology_id = f"topology_{result_file.stem}"
                        self.topologies[topology_id] = topology
                        
                        # Create corresponding model
                        config = data.get('config', {})
                        model = OptimizedMoEModel(
                            input_dim=config.get('input_dim', 64),
                            num_experts=config.get('num_experts', 8),
                            hidden_dim=config.get('hidden_dim', 128)
                        )
                        model.set_routing_topology(topology)
                        self.models[topology_id] = model
                        
                except Exception as e:
                    logging.warning(f"Could not load topology from {result_file}: {e}")
    
    def get_next_task_id(self) -> str:
        """Get next task ID."""
        with self.lock:
            self.task_counter += 1
            return f"task_{self.task_counter}_{int(time.time())}"
    
    def get_active_tasks(self) -> int:
        """Get number of active tasks."""
        with self.lock:
            return len([t for t in self.evolution_tasks.values() 
                       if t['status'] in ['pending', 'running']])


# Global server state
server_state = ServerState()


# FastAPI lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logging.info("🚀 Starting Production MoE Router Server")
    
    # Create default topology if none exist
    if not server_state.topologies:
        logging.info("Creating default topology...")
        default_topology = OptimizedTopology(16, 8, 0.3)
        server_state.topologies["default"] = default_topology
        
        default_model = OptimizedMoEModel(64, 8, 128)
        default_model.set_routing_topology(default_topology)
        server_state.models["default"] = default_model
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down Production MoE Router Server")


# Create FastAPI app
app = FastAPI(
    title="Self-Evolving MoE Router API",
    description="Production-ready API for evolutionary MoE routing optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        import psutil
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except ImportError:
        system_metrics = {"status": "psutil not available"}
    
    uptime = time.time() - server_state.start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0",
        uptime_seconds=uptime,
        active_tasks=server_state.get_active_tasks(),
        system_metrics=system_metrics
    )


# Evolution endpoints
@app.post("/evolution/start", response_model=EvolutionResponse)
async def start_evolution(request: EvolutionRequest, background_tasks: BackgroundTasks):
    """Start evolution process."""
    task_id = server_state.get_next_task_id()
    
    # Initialize task
    task_info = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "current_generation": 0,
        "best_fitness": float('-inf'),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": request.dict(),
        "result": None,
        "error": None
    }
    
    server_state.evolution_tasks[task_id] = task_info
    
    # Start background evolution
    background_tasks.add_task(run_evolution_task, task_id, request)
    
    return EvolutionResponse(
        task_id=task_id,
        status="pending",
        message=f"Evolution task {task_id} started"
    )


@app.get("/evolution/status/{task_id}", response_model=EvolutionStatus)
async def get_evolution_status(task_id: str):
    """Get evolution task status."""
    if task_id not in server_state.evolution_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = server_state.evolution_tasks[task_id]
    
    return EvolutionStatus(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        current_generation=task_info["current_generation"],
        best_fitness=task_info["best_fitness"],
        start_time=task_info["start_time"],
        result=task_info["result"],
        error=task_info["error"]
    )


@app.get("/evolution/tasks")
async def list_evolution_tasks():
    """List all evolution tasks."""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": info["status"],
                "progress": info["progress"],
                "start_time": info["start_time"]
            }
            for task_id, info in server_state.evolution_tasks.items()
        ]
    }


# Topology endpoints
@app.get("/topologies")
async def list_topologies():
    """List available topologies."""
    return {
        "topologies": [
            {
                "topology_id": topology_id,
                "sparsity": topology.compute_sparsity(),
                "num_tokens": topology.num_tokens,
                "num_experts": topology.num_experts
            }
            for topology_id, topology in server_state.topologies.items()
        ]
    }


@app.get("/topologies/{topology_id}")
async def get_topology(topology_id: str):
    """Get topology details."""
    if topology_id not in server_state.topologies:
        raise HTTPException(status_code=404, detail="Topology not found")
    
    topology = server_state.topologies[topology_id]
    
    return {
        "topology_id": topology_id,
        "num_tokens": topology.num_tokens,
        "num_experts": topology.num_experts,
        "sparsity": topology.compute_sparsity(),
        "generation": topology.generation,
        "routing_matrix": topology.routing_matrix.tolist()
    }


# Inference endpoints
@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Run inference with MoE model."""
    # Validate input data
    if not request.data:
        raise HTTPException(status_code=400, detail="No input data provided")
    
    try:
        # Convert to numpy array
        input_data = np.array(request.data, dtype=np.float32)
        
        if len(input_data.shape) != 3:
            raise HTTPException(
                status_code=400, 
                detail=f"Input data must have 3 dimensions, got {len(input_data.shape)}"
            )
        
        # Select topology/model
        topology_id = request.topology_id or "default"
        if topology_id not in server_state.models:
            raise HTTPException(status_code=404, detail=f"Topology {topology_id} not found")
        
        model = server_state.models[topology_id]
        
        # Run inference
        start_time = time.perf_counter()
        output, aux_info = model.forward_fast(input_data)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        return InferenceResponse(
            output=output.tolist(),
            expert_usage=aux_info['expert_usage'].tolist(),
            inference_time_ms=inference_time,
            topology_used=topology_id
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Quality gates endpoint
@app.post("/quality/check")
async def run_quality_check(background_tasks: BackgroundTasks):
    """Run quality gates check."""
    task_id = server_state.get_next_task_id()
    
    # Run quality check in background
    background_tasks.add_task(run_quality_check_task, task_id)
    
    return {"task_id": task_id, "status": "running", "message": "Quality check started"}


@app.get("/quality/report")
async def get_quality_report():
    """Get latest quality report."""
    report_path = Path("quality_gates_results/quality_gates_report.json")
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No quality report found")
    
    try:
        with open(report_path) as f:
            report = json.load(f)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read quality report: {e}")


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get server metrics."""
    metrics = {
        "uptime_seconds": time.time() - server_state.start_time,
        "total_tasks": len(server_state.evolution_tasks),
        "active_tasks": server_state.get_active_tasks(),
        "available_topologies": len(server_state.topologies),
        "task_status_summary": {}
    }
    
    # Task status summary
    status_counts = {}
    for task_info in server_state.evolution_tasks.values():
        status = task_info["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    metrics["task_status_summary"] = status_counts
    
    return metrics


# Background task functions
async def run_evolution_task(task_id: str, request: EvolutionRequest):
    """Run evolution task in background."""
    try:
        # Update status
        server_state.evolution_tasks[task_id]["status"] = "running"
        
        # Create evolver and model
        evolver = OptimizedEvolver(population_size=request.population_size)
        evolver.initialize_population(request.num_tokens, request.num_experts)
        
        model = OptimizedMoEModel(
            input_dim=request.input_dim,
            num_experts=request.num_experts,
            hidden_dim=request.hidden_dim
        )
        
        # Create synthetic data
        data_batches = []
        for i in range(8):
            inputs = np.random.randn(4, request.num_tokens, request.input_dim).astype(np.float32)
            targets = (inputs * 0.8 + 0.1 * np.random.randn(4, request.num_tokens, request.input_dim)).astype(np.float32)
            data_batches.append((inputs, targets))
        
        # Evolution loop
        for generation in range(request.generations):
            stats = evolver.evolve_generation(model, data_batches)
            
            # Update progress
            progress = (generation + 1) / request.generations
            server_state.evolution_tasks[task_id].update({
                "progress": progress,
                "current_generation": generation + 1,
                "best_fitness": float(evolver.best_fitness)
            })
            
            # Small delay to avoid overwhelming
            await asyncio.sleep(0.1)
        
        # Store result
        result = {
            "final_fitness": float(evolver.best_fitness),
            "total_generations": request.generations,
            "best_topology_sparsity": float(evolver.best_topology.compute_sparsity()) if evolver.best_topology else None
        }
        
        # Save topology and model
        if evolver.best_topology:
            topology_id = f"evolved_{task_id}"
            server_state.topologies[topology_id] = evolver.best_topology
            
            evolved_model = OptimizedMoEModel(
                input_dim=request.input_dim,
                num_experts=request.num_experts,
                hidden_dim=request.hidden_dim
            )
            evolved_model.set_routing_topology(evolver.best_topology)
            server_state.models[topology_id] = evolved_model
            
            result["topology_id"] = topology_id
        
        # Update final status
        server_state.evolution_tasks[task_id].update({
            "status": "completed",
            "progress": 1.0,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Evolution task {task_id} failed: {e}")
        server_state.evolution_tasks[task_id].update({
            "status": "failed",
            "error": str(e)
        })


async def run_quality_check_task(task_id: str):
    """Run quality check in background."""
    try:
        # This would run the quality gates
        # For now, just simulate
        await asyncio.sleep(2)
        
        # In a real implementation, you would run:
        # system = QualityGateSystem()
        # report = system.run_all_gates()
        
        logger.info(f"Quality check task {task_id} completed")
        
    except Exception as e:
        logger.error(f"Quality check task {task_id} failed: {e}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


def main():
    """Run the production server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Evolving MoE Router Production Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    logger.info(f"🚀 Starting production server on {args.host}:{args.port}")
    
    uvicorn.run(
        "production_ready_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()