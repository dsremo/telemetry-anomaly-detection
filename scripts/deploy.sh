#!/usr/bin/env bash
# Dsremo Telemetry Detection — Production Deploy Script
#
# Usage:
#   ./scripts/deploy.sh            # Full clean rebuild (new deps / something stuck)
#   ./scripts/deploy.sh --quick    # Rebuild with layer cache (code-only changes)
#   ./scripts/deploy.sh --api      # Rebuild + restart API only (~30s, Python changes)
#
# Run on the SpaceAi EC2 from the repo root.
# NEVER use docker-compose.yml (dev only).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$REPO_DIR/docker-compose.detect.prod.yml"
OVERRIDE_FILE="$REPO_DIR/docker-compose.detect.prod.override.yml"
COMPOSE_ARGS="-f $COMPOSE_FILE -f $OVERRIDE_FILE"
ENV_FILE="$REPO_DIR/.env.detect"
MODE="${1:-}"

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
step()  { echo -e "\n${BOLD}${BLUE}══ $* ══${NC}"; }
warn()  { echo -e "${YELLOW}[deploy]${NC} $*"; }
error() { echo -e "${RED}[deploy]${NC} $*" >&2; }
ok()    { echo -e "  ${GREEN}✓${NC} $*"; }
fail()  { echo -e "  ${RED}✗${NC} $*"; }

# ── Pre-flight ──────────────────────────────────────────────────────────────
step "Pre-flight checks"
[[ -f "$ENV_FILE" ]] || { error ".env.detect not found at $ENV_FILE. Aborting."; exit 1; }
command -v docker &>/dev/null || { error "docker not found"; exit 1; }
docker compose version &>/dev/null || { error "docker compose v2 not found"; exit 1; }

# Verify required secrets are set
source "$ENV_FILE"
[[ -n "${DETECT_DB_PASSWORD:-}" ]] || { error "DETECT_DB_PASSWORD not set in .env.detect"; exit 1; }
[[ -n "${DETECT_JWT_SECRET:-}" ]]  || { error "DETECT_JWT_SECRET not set in .env.detect"; exit 1; }

cd "$REPO_DIR"
ok "Environment file found and validated"
ok "Docker available"

# ── 1. Pull latest code ────────────────────────────────────────────────────
step "1 · Pull latest code"
git fetch origin
git pull --ff-only
COMMIT=$(git rev-parse --short HEAD)
info "Now at commit: $COMMIT"

# ── PARTIAL DEPLOY — --api mode ────────────────────────────────────────────
if [[ "$MODE" == "--api" ]]; then
    step "PARTIAL DEPLOY — API only"
    docker compose $COMPOSE_ARGS --env-file "$ENV_FILE" build detect-api 2>&1
    ok "API image rebuilt"
    docker compose $COMPOSE_ARGS --env-file "$ENV_FILE" up -d --no-deps detect-api 2>&1
    ok "API container restarted"
    info "Waiting 20s for API to stabilize..."
    sleep 20
    docker compose $COMPOSE_ARGS ps --format "table {{.Name}}\t{{.Status}}" | grep detect-api || true
    echo ""
    info "✓ API partial deploy done at commit $COMMIT"
    exit 0
fi

# ── 2. Stop all containers gracefully ─────────────────────────────────────
step "2 · Stop all containers"
docker compose $COMPOSE_ARGS stop 2>/dev/null && ok "All containers stopped" || ok "No containers were running"

# ── 3. Remove old containers (keeps volumes — data is safe) ────────────────
step "3 · Remove old containers"
docker compose $COMPOSE_ARGS rm -f 2>/dev/null && ok "Containers removed" || ok "Nothing to remove"

# ── 4. Prune build cache ───────────────────────────────────────────────────
step "4 · Prune build cache"
if [[ "$MODE" == "--quick" ]]; then
    warn "Skipping cache prune (--quick mode)"
else
    docker builder prune -f --filter until=24h 2>/dev/null || true
    ok "Build cache pruned"
fi

# ── 5. Build all images ────────────────────────────────────────────────────
step "5 · Build all images"
if [[ "$MODE" == "--quick" ]]; then
    docker compose $COMPOSE_ARGS --env-file "$ENV_FILE" build 2>&1
else
    docker compose $COMPOSE_ARGS --env-file "$ENV_FILE" build --no-cache 2>&1
fi
docker compose $COMPOSE_ARGS --env-file "$ENV_FILE" up -d 2>&1
ok "All images built and containers started"

# ── 6. Wait for DB ─────────────────────────────────────────────────────────
step "6 · Wait for database"
for i in $(seq 1 30); do
    if docker compose $COMPOSE_ARGS exec -T detect-db \
        pg_isready -U "${DETECT_DB_USER:-dsremo}" -d "${DETECT_DB_NAME:-dsremo}" &>/dev/null; then
        ok "Database is healthy"
        break
    fi
    [[ $i -eq 30 ]] && { error "Database failed to start after 60s. Aborting."; exit 1; }
    warn "  Waiting for DB ($i/30)..."
    sleep 2
done

# ── 7. Run DB migrations ───────────────────────────────────────────────────
step "7 · Run DB migrations"
info "Running migrations via API startup (dsremo auto-migrates on first run)..."
# The dsremo app runs migrations on startup via dsremo.db.migrations
# Verify by checking API becomes healthy
ok "Migrations delegated to API startup"

# ── 8. Wait for API healthy ────────────────────────────────────────────────
step "8 · Wait for API health"
for i in $(seq 1 60); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' detect-api 2>/dev/null || echo "missing")
    if [[ "$STATUS" == "healthy" ]]; then
        ok "API is healthy"
        break
    fi
    if [[ $i -eq 60 ]]; then
        error "API failed to become healthy after 300s."
        docker logs detect-api --tail=30
        exit 1
    fi
    warn "  API status: $STATUS ($i/60)..."
    sleep 5
done

# ── 9. Nginx reload ────────────────────────────────────────────────────────
# System nginx on host handles SSL for detect.dsremo.com (shared EC2 setup)
step "9 · Reload system nginx"
if sudo nginx -t 2>/dev/null && sudo systemctl reload nginx; then
    ok "System nginx reloaded"
else
    warn "Could not reload nginx — check config manually"
fi

# ── 10. Container status ───────────────────────────────────────────────────
step "10 · Container status"
docker compose $COMPOSE_ARGS ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# ── 11. Smoke tests ────────────────────────────────────────────────────────
step "11 · Smoke tests"
HOST="detect.dsremo.com"
PASS=0; FAIL=0; FAILS=()

smoke() {
    local label="$1" path="$2" expect="${3:-200}"
    local code
    # Hit API directly on port 8400 (system nginx proxies externally)
    code=$(curl -sf -o /dev/null -w "%{http_code}" "http://localhost:8400$path" 2>/dev/null || echo "000")
    if [[ "$code" == "$expect" ]]; then
        ok "$label ($code)"
        PASS=$((PASS+1))
    else
        fail "$label — expected $expect got $code"
        FAIL=$((FAIL+1))
        FAILS+=("$label: expected $expect got $code")
    fi
}

smoke "Health endpoint"    "/api/v1/health"
smoke "Dashboard UI"       "/"
smoke "API docs"           "/docs"
smoke "Tenants endpoint"   "/api/v1/tenants"    "401"
smoke "Keys endpoint"      "/api/v1/keys"       "401"
smoke "Channels endpoint"  "/api/v1/channels"   "401"

# Check DB is connected
API_HEALTH=$(curl -sf "http://localhost:8400/api/v1/health" 2>/dev/null || echo "{}")
DB_STATUS=$(echo "$API_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('db','?'))" 2>/dev/null || echo "?")
[[ "$DB_STATUS" == "ok" ]] && ok "DB connection: ok" || { fail "DB connection: $DB_STATUS"; FAIL=$((FAIL+1)); FAILS+=("DB not ok: $DB_STATUS"); }

echo ""
echo -e "${BOLD}══ Deploy Summary ══${NC}"
echo "  Commit:  $COMMIT"
echo "  Passed:  ${GREEN}$PASS${NC}"
echo "  Failed:  ${RED}$FAIL${NC}"

if [[ $FAIL -gt 0 ]]; then
    error "DEPLOY COMPLETED WITH FAILURES"
    for f in "${FAILS[@]}"; do echo -e "  ${RED}→${NC} $f"; done
    exit 1
else
    info "✓ Deploy successful — all $PASS checks passed. detect.dsremo.com is live."
fi
