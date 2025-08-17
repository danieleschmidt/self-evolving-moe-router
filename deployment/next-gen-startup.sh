#!/bin/bash

# TERRAGON Next-Gen Production Startup Script
# Comprehensive startup for all next-generation features

set -e

echo "🚀 Starting TERRAGON Next-Gen Evolution System v5.0"
echo "=============================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/terragon_env"
DEPLOYMENT_MODE="${1:-single}"  # single, distributed, cluster

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Python version: $PYTHON_VERSION"
    
    # Check Docker (for containerized deployment)
    if command -v docker &> /dev/null; then
        log_info "Docker found: $(docker --version)"
        DOCKER_AVAILABLE=true
    else
        log_warning "Docker not found - containerized deployment not available"
        DOCKER_AVAILABLE=false
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        log_info "Docker Compose found: $(docker-compose --version)"
        COMPOSE_AVAILABLE=true
    elif docker compose version &> /dev/null; then
        log_info "Docker Compose V2 found: $(docker compose version)"
        COMPOSE_AVAILABLE=true
        COMPOSE_CMD="docker compose"
    else
        log_warning "Docker Compose not found"
        COMPOSE_AVAILABLE=false
        COMPOSE_CMD="docker-compose"
    fi
}

# Function to setup virtual environment
setup_environment() {
    log_info "Setting up Python environment..."
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install -r requirements.txt
    
    log_success "Environment setup complete"
}

# Function to run quality gates
run_quality_gates() {
    log_info "Running comprehensive quality gates..."
    
    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Set Python path
    export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}"
    
    # Run quality gates
    if python comprehensive_quality_gates.py; then
        log_success "Quality gates passed"
    else
        log_warning "Quality gates failed - continuing with startup"
    fi
}

# Function to start single node deployment
start_single_node() {
    log_info "Starting single node deployment..."
    
    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Set environment variables
    export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}"
    export MONITORING_ENABLED=true
    export QUANTUM_ENABLED=true
    export DISTRIBUTED_ENABLED=false
    export ADAPTIVE_MUTATIONS_ENABLED=true
    export AUTO_TUNING_ENABLED=true
    
    # Start the next-gen server
    log_info "Starting TERRAGON Next-Gen Production Server..."
    python next_gen_production_server.py \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info &
    
    SERVER_PID=$!
    echo $SERVER_PID > next_gen_server.pid
    
    log_success "Next-Gen server started with PID: $SERVER_PID"
    
    # Wait a moment for server to start
    sleep 5
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Health check passed - server is running"
    else
        log_error "Health check failed - server may not be responding"
    fi
}

# Function to start distributed deployment
start_distributed() {
    log_info "Starting distributed deployment with consensus..."
    
    if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
        cd "$PROJECT_DIR/deployment"
        
        log_info "Starting distributed nodes with Docker Compose..."
        $COMPOSE_CMD -f next-gen-docker-compose.yml --profile distributed up -d
        
        log_success "Distributed deployment started"
        
        # Wait for services to be ready
        log_info "Waiting for services to be ready..."
        sleep 30
        
        # Health check all nodes
        for port in 8000 8002 8003; do
            if curl -f http://localhost:$port/health > /dev/null 2>&1; then
                log_success "Node on port $port is healthy"
            else
                log_warning "Node on port $port is not responding"
            fi
        done
        
    else
        log_error "Docker and Docker Compose are required for distributed deployment"
        exit 1
    fi
}

# Function to start cluster deployment
start_cluster() {
    log_info "Starting full cluster deployment with monitoring..."
    
    if [ "$DOCKER_AVAILABLE" = true ] && [ "$COMPOSE_AVAILABLE" = true ]; then
        cd "$PROJECT_DIR/deployment"
        
        log_info "Starting full cluster with monitoring stack..."
        $COMPOSE_CMD -f next-gen-docker-compose.yml up -d
        
        log_success "Cluster deployment started"
        
        # Wait for all services
        log_info "Waiting for all services to be ready..."
        sleep 60
        
        # Health checks
        services=(
            "http://localhost:8000/health|Main Server"
            "http://localhost:9090/-/healthy|Prometheus"
            "http://localhost:3000/api/health|Grafana"
        )
        
        for service in "${services[@]}"; do
            IFS='|' read -r url name <<< "$service"
            if curl -f "$url" > /dev/null 2>&1; then
                log_success "$name is healthy"
            else
                log_warning "$name is not responding"
            fi
        done
        
    else
        log_error "Docker and Docker Compose are required for cluster deployment"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    log_info "Running next-gen feature tests..."
    
    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"
    export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH}"
    
    # Run the comprehensive test suite
    if python -m pytest tests/test_next_gen_features.py -v; then
        log_success "Next-gen tests passed"
    else
        log_warning "Some tests failed - check test output"
    fi
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [DEPLOYMENT_MODE] [OPTIONS]"
    echo ""
    echo "Deployment Modes:"
    echo "  single      - Single node deployment (default)"
    echo "  distributed - Multi-node deployment with consensus"
    echo "  cluster     - Full cluster with monitoring stack"
    echo "  test        - Run tests only"
    echo ""
    echo "Options:"
    echo "  --skip-quality-gates  - Skip quality gate checks"
    echo "  --skip-tests         - Skip test execution"
    echo "  --force-rebuild      - Force rebuild of Docker images"
    echo ""
    echo "Examples:"
    echo "  $0 single"
    echo "  $0 distributed --skip-quality-gates"
    echo "  $0 cluster --force-rebuild"
    echo "  $0 test"
}

# Parse command line arguments
SKIP_QUALITY_GATES=false
SKIP_TESTS=false
FORCE_REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-quality-gates)
            SKIP_QUALITY_GATES=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            if [ -z "$DEPLOYMENT_MODE" ]; then
                DEPLOYMENT_MODE="$1"
            else
                log_error "Unknown option: $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate deployment mode
case $DEPLOYMENT_MODE in
    single|distributed|cluster|test)
        ;;
    *)
        log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
        show_usage
        exit 1
        ;;
esac

# Main execution
main() {
    log_info "TERRAGON Next-Gen Startup - Mode: $DEPLOYMENT_MODE"
    
    # Check prerequisites
    check_prerequisites
    
    # Handle test mode
    if [ "$DEPLOYMENT_MODE" = "test" ]; then
        setup_environment
        run_tests
        exit 0
    fi
    
    # Force rebuild if requested
    if [ "$FORCE_REBUILD" = true ] && [ "$DOCKER_AVAILABLE" = true ]; then
        log_info "Forcing Docker image rebuild..."
        cd "$PROJECT_DIR/deployment"
        $COMPOSE_CMD -f next-gen-docker-compose.yml build --no-cache
    fi
    
    # Setup environment for non-Docker deployments
    if [ "$DEPLOYMENT_MODE" = "single" ]; then
        setup_environment
    fi
    
    # Run quality gates
    if [ "$SKIP_QUALITY_GATES" = false ] && [ "$DEPLOYMENT_MODE" = "single" ]; then
        run_quality_gates
    fi
    
    # Start deployment
    case $DEPLOYMENT_MODE in
        single)
            start_single_node
            ;;
        distributed)
            start_distributed
            ;;
        cluster)
            start_cluster
            ;;
    esac
    
    # Run tests if not skipped
    if [ "$SKIP_TESTS" = false ] && [ "$DEPLOYMENT_MODE" = "single" ]; then
        run_tests
    fi
    
    # Display success information
    echo ""
    log_success "TERRAGON Next-Gen Evolution System is running!"
    echo "=============================================="
    
    case $DEPLOYMENT_MODE in
        single)
            echo "🌐 Main Server: http://localhost:8000"
            echo "📊 Health Check: http://localhost:8000/health"
            echo "🔧 Next-Gen Status: http://localhost:8000/status/next-gen"
            echo "📈 Performance Analysis: http://localhost:8000/analysis/next-gen-performance"
            ;;
        distributed)
            echo "🌐 Node 1: http://localhost:8000"
            echo "🌐 Node 2: http://localhost:8002"
            echo "🌐 Node 3: http://localhost:8003"
            echo "📊 Monitoring: http://localhost:9090"
            ;;
        cluster)
            echo "🌐 Main Server: http://localhost:8000"
            echo "📊 Prometheus: http://localhost:9090"
            echo "📈 Grafana: http://localhost:3000 (admin/terragon2024)"
            echo "⚖️ Load Balancer: http://localhost:80"
            ;;
    esac
    
    echo ""
    echo "📚 Next-Gen Features Available:"
    echo "  🔬 Quantum-Inspired Evolution"
    echo "  🌐 Distributed Consensus"  
    echo "  🧬 Adaptive Mutations"
    echo "  📊 Real-Time Monitoring"
    echo "  🎯 Auto-Tuning"
    echo ""
    
    if [ "$DEPLOYMENT_MODE" = "single" ]; then
        echo "🛑 To stop the server: kill \$(cat next_gen_server.pid)"
    else
        echo "🛑 To stop the cluster: cd deployment && $COMPOSE_CMD -f next-gen-docker-compose.yml down"
    fi
    
    echo ""
    log_info "Startup complete! 🎉"
}

# Run main function
main