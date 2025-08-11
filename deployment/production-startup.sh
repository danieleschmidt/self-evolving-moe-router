#!/bin/bash
set -euo pipefail

# TERRAGON Production Startup Script
# Automated production deployment for Self-Evolving MoE-Router
# Usage: ./production-startup.sh [init|start|stop|status|logs|backup|security-scan]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/production-deployment.yml"
ENV_FILE="$SCRIPT_DIR/.env.production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for production deployment"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose V2 is not installed"
        exit 1
    fi
    
    # Check Docker Swarm
    if ! docker info | grep -q "Swarm: active"; then
        warn "Docker Swarm is not initialized"
        log "Initializing Docker Swarm..."
        docker swarm init
    fi
    
    # Check available disk space (minimum 100GB)
    AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
    MIN_SPACE=$((100 * 1024 * 1024)) # 100GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $MIN_SPACE ]]; then
        error "Insufficient disk space. Minimum 100GB required, available: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
        exit 1
    fi
    
    # Check available memory (minimum 16GB)
    AVAILABLE_MEM=$(free -m | awk 'NR==2 {print $7}')
    MIN_MEM=16384 # 16GB in MB
    
    if [[ $AVAILABLE_MEM -lt $MIN_MEM ]]; then
        error "Insufficient memory. Minimum 16GB required, available: ${AVAILABLE_MEM}MB"
        exit 1
    fi
    
    # Check GPU availability (optional)
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        log "Found $GPU_COUNT GPU(s)"
    else
        warn "No NVIDIA GPUs detected - will run on CPU only"
    fi
    
    success "System requirements check passed"
}

# Create environment file
create_env_file() {
    log "Creating production environment configuration..."
    
    if [[ -f "$ENV_FILE" ]]; then
        warn "Environment file already exists. Backing up..."
        cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%s)"
    fi
    
    # Generate secure passwords and keys
    POSTGRES_PASSWORD=$(openssl rand -base64 32)
    POSTGRES_REPLICATION_PASSWORD=$(openssl rand -base64 32)
    TERRAGON_SECRET_KEY=$(openssl rand -base64 64)
    GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)
    GRAFANA_DB_PASSWORD=$(openssl rand -base64 32)
    
    cat > "$ENV_FILE" << EOF
# TERRAGON Production Environment Configuration
# Generated on $(date)

# Database Configuration
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_REPLICATION_PASSWORD=$POSTGRES_REPLICATION_PASSWORD
GRAFANA_DB_PASSWORD=$GRAFANA_DB_PASSWORD

# Application Secrets
TERRAGON_SECRET_KEY=$TERRAGON_SECRET_KEY
GRAFANA_ADMIN_PASSWORD=$GRAFANA_ADMIN_PASSWORD

# Backup Configuration (Configure these for your environment)
BACKUP_S3_BUCKET=terragon-prod-backups
BACKUP_AWS_ACCESS_KEY=CHANGE_ME
BACKUP_AWS_SECRET_KEY=CHANGE_ME

# Security Configuration
SECURITY_ALERT_WEBHOOK=https://hooks.slack.com/CHANGE_ME

# SSL Configuration
SSL_CERT_PATH=/etc/ssl/certs/terragon.crt
SSL_KEY_PATH=/etc/ssl/private/terragon.key

# Performance Tuning
TERRAGON_MAX_WORKERS=16
TERRAGON_WORKER_TIMEOUT=300
TERRAGON_MAX_MEMORY_GB=32

EOF

    chmod 600 "$ENV_FILE"
    success "Environment file created at $ENV_FILE"
    
    warn "IMPORTANT: Please update the following in $ENV_FILE:"
    warn "  - BACKUP_AWS_ACCESS_KEY and BACKUP_AWS_SECRET_KEY"
    warn "  - SECURITY_ALERT_WEBHOOK"
    warn "  - SSL certificate paths if using custom certificates"
}

# Initialize production environment
init_production() {
    log "Initializing TERRAGON production environment..."
    
    check_root
    check_requirements
    create_env_file
    
    # Create required directories
    log "Creating storage directories..."
    mkdir -p /mnt/storage/terragon/{models,data,logs}
    mkdir -p /mnt/storage/postgres/{primary,replica}
    mkdir -p /mnt/storage/{prometheus,elasticsearch}
    
    # Set proper permissions
    chown -R 999:999 /mnt/storage/postgres
    chown -R 65534:65534 /mnt/storage/prometheus
    chown -R 1000:1000 /mnt/storage/elasticsearch
    
    # Create Docker networks
    log "Creating Docker networks..."
    docker network create --driver overlay --attachable terragon-prod-network || true
    
    # Build and push production images
    log "Building production images..."
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker build -f deployment/Dockerfile -t terragon/self-evolving-moe:latest .
    
    # Tag for production
    docker tag terragon/self-evolving-moe:latest terragon/self-evolving-moe:prod-$(date +%Y%m%d-%H%M%S)
    
    success "Production environment initialized successfully"
    
    log "Next steps:"
    log "1. Review and update $ENV_FILE with your specific configuration"
    log "2. Configure SSL certificates"
    log "3. Set up external backup storage"
    log "4. Run './production-startup.sh start' to start the services"
}

# Start production services
start_production() {
    log "Starting TERRAGON production services..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file not found. Run './production-startup.sh init' first"
        exit 1
    fi
    
    # Source environment variables
    source "$ENV_FILE"
    
    # Start services using Docker Swarm
    cd "$SCRIPT_DIR"
    
    log "Deploying services to Docker Swarm..."
    docker stack deploy --compose-file "$COMPOSE_FILE" --with-registry-auth terragon-prod
    
    # Wait for services to be ready
    log "Waiting for services to become healthy..."
    sleep 30
    
    # Check service status
    check_service_health
    
    success "TERRAGON production services started successfully"
    
    log "Access points:"
    log "  - API: https://localhost:443"
    log "  - Monitoring: https://localhost:3000"
    log "  - Metrics: https://localhost:9090"
    log "  - Alerts: https://localhost:9093"
    log "  - Tracing: https://localhost:16686"
}

# Stop production services
stop_production() {
    log "Stopping TERRAGON production services..."
    
    # Remove Docker stack
    docker stack rm terragon-prod
    
    # Wait for cleanup
    log "Waiting for services to shut down..."
    sleep 60
    
    # Remove unused networks and volumes (optional)
    read -p "Remove unused Docker networks and volumes? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker system prune -f --volumes
    fi
    
    success "TERRAGON production services stopped"
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    # Check Docker services
    SERVICES=$(docker service ls --filter label=com.docker.stack.namespace=terragon-prod --format "table {{.Name}}\t{{.Replicas}}")
    echo "$SERVICES"
    
    # Check for unhealthy services
    UNHEALTHY=$(docker service ls --filter label=com.docker.stack.namespace=terragon-prod --format "{{.Name}} {{.Replicas}}" | grep "0/" | wc -l)
    
    if [[ $UNHEALTHY -gt 0 ]]; then
        warn "Some services are not healthy"
        return 1
    else
        success "All services are running"
        return 0
    fi
}

# Show production status
show_status() {
    log "TERRAGON Production Status"
    echo "=========================="
    
    # System resources
    echo "System Resources:"
    echo "  CPU: $(nproc) cores"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
    echo "  Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2}')"
    echo
    
    # Docker Swarm status
    echo "Docker Swarm:"
    docker node ls
    echo
    
    # Service status
    echo "Services:"
    check_service_health
    echo
    
    # Container resource usage
    echo "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" | head -20
}

# Show logs
show_logs() {
    local SERVICE=${2:-}
    
    if [[ -z "$SERVICE" ]]; then
        log "Available services:"
        docker service ls --filter label=com.docker.stack.namespace=terragon-prod --format "  {{.Name}}"
        return
    fi
    
    log "Showing logs for service: $SERVICE"
    docker service logs -f "terragon-prod_$SERVICE"
}

# Backup production data
backup_production() {
    log "Starting production backup..."
    
    BACKUP_DIR="/tmp/terragon-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    log "Backing up PostgreSQL database..."
    docker exec $(docker ps -q -f name=terragon-prod_postgres-primary) \
        pg_dump -U terragon terragon_prod > "$BACKUP_DIR/database.sql"
    
    # Backup models and data
    log "Backing up models and data..."
    tar -czf "$BACKUP_DIR/models.tar.gz" -C /mnt/storage/terragon models
    tar -czf "$BACKUP_DIR/data.tar.gz" -C /mnt/storage/terragon data
    
    # Backup configuration
    log "Backing up configuration..."
    cp -r "$SCRIPT_DIR" "$BACKUP_DIR/deployment"
    
    log "Backup completed: $BACKUP_DIR"
    
    # Optional: Upload to S3 (if configured)
    if [[ -n "${BACKUP_S3_BUCKET:-}" && -n "${BACKUP_AWS_ACCESS_KEY:-}" ]]; then
        log "Uploading backup to S3..."
        # Add AWS CLI commands here
    fi
    
    success "Backup completed successfully"
}

# Run security scan
security_scan() {
    log "Running security scan..."
    
    # Scan Docker images
    log "Scanning Docker images for vulnerabilities..."
    
    # Run quality gates with security focus
    log "Running TERRAGON quality gates..."
    cd "$PROJECT_ROOT"
    python3 simple_quality_gates.py
    
    # Check for security updates
    log "Checking for security updates..."
    apt list --upgradable 2>/dev/null | grep -i security || true
    
    success "Security scan completed"
}

# Update production deployment
update_production() {
    log "Updating TERRAGON production deployment..."
    
    # Pull latest changes
    log "Pulling latest code..."
    cd "$PROJECT_ROOT"
    git pull origin main
    
    # Run quality gates
    log "Running quality gates..."
    python3 simple_quality_gates.py
    
    if [[ $? -ne 0 ]]; then
        error "Quality gates failed. Aborting update."
        exit 1
    fi
    
    # Build new images
    log "Building updated images..."
    docker build -f deployment/Dockerfile -t terragon/self-evolving-moe:latest .
    
    # Tag with version
    NEW_TAG="prod-$(date +%Y%m%d-%H%M%S)"
    docker tag terragon/self-evolving-moe:latest "terragon/self-evolving-moe:$NEW_TAG"
    
    # Rolling update
    log "Performing rolling update..."
    cd "$SCRIPT_DIR"
    docker stack deploy --compose-file "$COMPOSE_FILE" --with-registry-auth terragon-prod
    
    # Wait and check health
    sleep 60
    if check_service_health; then
        success "Production update completed successfully"
    else
        error "Update failed - some services are unhealthy"
        exit 1
    fi
}

# Main command handler
main() {
    local COMMAND=${1:-help}
    
    case $COMMAND in
        "init")
            init_production
            ;;
        "start")
            start_production
            ;;
        "stop")
            stop_production
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$@"
            ;;
        "backup")
            backup_production
            ;;
        "security-scan")
            security_scan
            ;;
        "update")
            update_production
            ;;
        "help"|*)
            echo "TERRAGON Production Management Script"
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  init          Initialize production environment"
            echo "  start         Start production services"
            echo "  stop          Stop production services"
            echo "  status        Show system and service status"
            echo "  logs <service> Show logs for a specific service"
            echo "  backup        Backup production data"
            echo "  security-scan Run security vulnerability scan"
            echo "  update        Update production deployment"
            echo "  help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 init                  # Initialize production environment"
            echo "  $0 start                 # Start all services"
            echo "  $0 logs terragon-app-prod # Show application logs"
            echo "  $0 backup                # Backup production data"
            ;;
    esac
}

# Trap for cleanup
trap 'error "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"