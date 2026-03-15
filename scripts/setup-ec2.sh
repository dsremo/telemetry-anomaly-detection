#!/usr/bin/env bash
# Dsremo Telemetry Detection — One-time SpaceAi EC2 setup
# Tested on Ubuntu 24.04 LTS (t3.small, ap-south-1)
# Run as ubuntu user with sudo privileges:
#   chmod +x setup-ec2.sh && ./setup-ec2.sh

set -euo pipefail

REPO_URL="https://github.com/dsremo/telemetry-anomaly-detection.git"
DEPLOY_DIR="/home/ubuntu/detect"
DOMAIN="detect.dsremo.com"
CERTBOT_EMAIL="dsremo7@gmail.com"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
error() { echo -e "${RED}[setup]${NC} $*" >&2; }
step()  { echo -e "\n${BOLD}══ $* ══${NC}"; }

# ── 1. System update ──────────────────────────────────────────────────────────
step "1 · System update"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo apt-get install -y --no-install-recommends \
    curl wget git unzip ca-certificates gnupg lsb-release ufw
info "System updated"

# ── 2. Firewall ───────────────────────────────────────────────────────────────
step "2 · Firewall"
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
info "UFW enabled: SSH + 80 + 443"

# ── 3. Docker ─────────────────────────────────────────────────────────────────
step "3 · Docker"
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker ubuntu
    sudo systemctl enable docker
    info "Docker installed. Group membership active on next login."
else
    info "Docker already installed — skipping."
fi
docker compose version || { error "docker compose v2 not found — check Docker installation"; exit 1; }

# ── 4. Clone repository ───────────────────────────────────────────────────────
step "4 · Clone repository"
if [[ -d "$DEPLOY_DIR" ]]; then
    warn "Directory $DEPLOY_DIR already exists — pulling latest."
    git -C "$DEPLOY_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$DEPLOY_DIR"
fi
info "Repo at $DEPLOY_DIR"

# ── 5. Environment file ───────────────────────────────────────────────────────
step "5 · Environment file"
if [[ ! -f "$DEPLOY_DIR/.env.detect" ]]; then
    cp "$DEPLOY_DIR/.env.detect.example" "$DEPLOY_DIR/.env.detect"

    # Auto-generate secrets
    DB_PASS=$(openssl rand -hex 32)
    JWT_SECRET=$(openssl rand -hex 32)
    sed -i "s|GENERATE_WITH: openssl rand -hex 32|$DB_PASS|1" "$DEPLOY_DIR/.env.detect"
    sed -i "s|GENERATE_WITH: openssl rand -hex 32|$JWT_SECRET|1" "$DEPLOY_DIR/.env.detect"

    info ".env.detect created with auto-generated secrets"
    warn "Secrets saved to $DEPLOY_DIR/.env.detect — keep this file safe, never commit it."
else
    info ".env.detect already exists — skipping."
fi

# ── 6. Get SSL certificate (port 80 must be free — nginx not yet running) ─────
step "6 · SSL certificate"
warn "Make sure DNS A record for $DOMAIN points to this server's IP"
warn "and port 80 is reachable before continuing."
echo ""
read -rp "Press ENTER to request SSL certificate (Ctrl+C to skip and do manually)..."

sudo apt-get install -y certbot
sudo certbot certonly --standalone \
    -d "$DOMAIN" \
    --non-interactive \
    --agree-tos \
    --email "$CERTBOT_EMAIL" || {
        warn "Certbot failed — run manually after DNS propagates:"
        warn "  sudo certbot certonly --standalone -d $DOMAIN --agree-tos --email $CERTBOT_EMAIL"
    }

# Copy certs to Docker volume location so nginx container can read them
VOLPATH="/var/lib/docker/volumes/detect_certbot_certs/_data"
sudo mkdir -p "$VOLPATH/live/$DOMAIN"
if [[ -d "/etc/letsencrypt/live/$DOMAIN" ]]; then
    sudo cp "/etc/letsencrypt/live/$DOMAIN/"*.pem "$VOLPATH/live/$DOMAIN/"
    info "Certs copied to Docker volume at $VOLPATH"
else
    warn "Certs not found at /etc/letsencrypt/live/$DOMAIN — run certbot manually then re-run this step."
fi

# SSL auto-renewal cron
(crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && docker exec detect-nginx nginx -s reload") \
    | sort -u | crontab -
info "SSL auto-renewal cron installed"

# ── 7. Deploy ─────────────────────────────────────────────────────────────────
step "7 · Deploy"
# Docker group takes effect after re-login; use sg to apply immediately
chmod +x "$DEPLOY_DIR/scripts/deploy.sh"
sg docker -c "$DEPLOY_DIR/scripts/deploy.sh" || {
    warn "If deploy failed due to docker permissions, log out and back in then run:"
    warn "  cd $DEPLOY_DIR && bash scripts/deploy.sh"
    exit 1
}

# ── 8. Summary ────────────────────────────────────────────────────────────────
step "Setup Complete"
info "════════════════════════════════════════════════════════"
info "Dsremo Telemetry Detection is live!"
info ""
info "URL:       https://$DOMAIN"
info "Logs:      docker compose -f $DEPLOY_DIR/docker-compose.detect.prod.yml logs -f"
info "Redeploy:  cd $DEPLOY_DIR && bash scripts/deploy.sh"
info "API only:  bash scripts/deploy.sh --api"
info "════════════════════════════════════════════════════════"
