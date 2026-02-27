/**
 * Sentinel Dashboard — zero-dependency vanilla JS.
 *
 * Connects via WebSocket for live anomaly events.
 * Polls REST API for initial state and periodic updates.
 * All state managed in plain objects — no framework needed.
 */

const API_BASE = window.location.origin + '/api/v1';
const WS_SCHEME = window.location.protocol === 'https:' ? 'wss' : 'ws';
const WS_URL = `${WS_SCHEME}://${window.location.host}/api/v1/ws/live`;

// --- State ---
const state = {
    anomalies: [],
    satellites: [],
    selectedAnomaly: null,
    subsystemStatus: {
        eps: 'nominal',
        adcs: 'nominal',
        thermal: 'nominal',
        comms: 'nominal',
    },
    totalPoints: 0,
    pointsLastHour: 0,
    connected: false,
    // Channels tab
    channels: [],
    selectedChannel: null,
};

// --- WebSocket ---
let ws = null;
let reconnectTimer = null;
let pingInterval = null;

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        state.connected = true;
        updateConnectionStatus(true);
        // Single keep-alive interval — cleared on each reconnect to prevent leak
        if (pingInterval) clearInterval(pingInterval);
        pingInterval = setInterval(() => { if (ws.readyState === WebSocket.OPEN) ws.send('ping'); }, 30000);
    };

    ws.onmessage = (event) => {
        if (event.data === 'pong') return;
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'anomaly') handleAnomalyEvent(msg.data);
        } catch (e) {
            console.warn('WS parse error:', e);
        }
    };

    ws.onclose = () => {
        state.connected = false;
        updateConnectionStatus(false);
        reconnectTimer = setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => ws.close();
}

// --- Event Handlers ---
function handleAnomalyEvent(data) {
    state.anomalies.unshift(data);
    if (state.anomalies.length > 200) state.anomalies.length = 200;

    updateSubsystemStatus(data);
    renderTimeline();
    renderMetrics();
    addAlert(data);
}

function updateSubsystemStatus(anomaly) {
    const subsystem = anomaly.subsystem || guessSubsystem(anomaly.parameter);
    if (!subsystem) return;

    const severityRank = { nominal: 0, watch: 1, warning: 2, critical: 3 };
    const current = state.subsystemStatus[subsystem] || 'nominal';
    if (severityRank[anomaly.severity] > severityRank[current]) {
        state.subsystemStatus[subsystem] = anomaly.severity;
        renderSubsystems();
    }
}

function guessSubsystem(param) {
    if (!param) return '';
    if (param.includes('battery') || param.includes('solar') || param.includes('bus')) return 'eps';
    if (param.includes('wheel') || param.includes('pointing')) return 'adcs';
    if (param.includes('temp') || param.includes('panel')) return 'thermal';
    if (param.includes('signal') || param.includes('bit') || param.includes('link')) return 'comms';
    return '';
}

// --- Rendering ---
function renderTimeline() {
    const list = document.getElementById('timelineList');
    const sevFilter = document.getElementById('severityFilter').value;
    const subFilter = document.getElementById('subsystemFilter').value;

    let filtered = state.anomalies;
    if (sevFilter) filtered = filtered.filter(a => a.severity === sevFilter);
    if (subFilter) filtered = filtered.filter(a => (a.subsystem || guessSubsystem(a.parameter)) === subFilter);

    if (filtered.length === 0) {
        list.innerHTML = '<div class="empty-state">No anomalies match filters</div>';
        return;
    }

    list.innerHTML = filtered.slice(0, 50).map((a, i) => {
        const time = formatTime(a.timestamp);
        const selected = state.selectedAnomaly?.id === a.id ? 'selected' : '';
        const explanation = (a.explanation || '').split('|')[0].trim();

        return `
            <div class="timeline-item ${selected}" data-index="${i}" onclick="selectAnomalyByIndex(${state.anomalies.indexOf(a)})">
                <span class="timeline-time">${time}</span>
                <span class="timeline-dot ${a.severity}"></span>
                <div class="timeline-content">
                    <div class="timeline-param">${a.parameter || 'unknown'}</div>
                    <div class="timeline-explanation">${explanation}</div>
                </div>
                <span class="timeline-severity ${a.severity}">${(a.severity || '').toUpperCase()}</span>
            </div>
        `;
    }).join('');
}

function renderMetrics() {
    document.getElementById('anomalyCount').textContent = state.anomalies.length;

    // Highest severity
    const severityRank = { nominal: 0, watch: 1, warning: 2, critical: 3 };
    let maxSev = 'nominal';
    let maxParam = '--';
    for (const a of state.anomalies.slice(0, 20)) {
        if (severityRank[a.severity] > severityRank[maxSev]) {
            maxSev = a.severity;
            maxParam = a.parameter || '';
        }
    }
    const badge = document.getElementById('maxSeverity');
    badge.textContent = maxSev.toUpperCase();
    badge.className = `metric-value severity-badge ${maxSev}`;
    if (maxSev === 'critical') badge.classList.add('pulse');
    document.getElementById('severityParam').textContent = maxParam;

    document.getElementById('satCount').textContent = state.satellites.length || '0';
    document.getElementById('satList').textContent = state.satellites.slice(0, 3).join(', ') || '--';

    document.getElementById('totalPoints').textContent = formatLargeNumber(state.totalPoints);
    document.getElementById('pointsTrend').textContent = `${formatLargeNumber(state.pointsLastHour)}/hr`;
}

function renderSubsystems() {
    for (const [sub, status] of Object.entries(state.subsystemStatus)) {
        const card = document.querySelector(`.subsystem-card[data-subsystem="${sub}"]`);
        if (!card) continue;
        const statusEl = card.querySelector('.subsystem-status');
        statusEl.textContent = status.toUpperCase();
        statusEl.className = `subsystem-status ${status}`;
        card.classList.toggle('alert', status === 'critical');
    }
}

function renderExplanation(anomaly) {
    const body = document.getElementById('explanationBody');
    if (!anomaly) {
        body.innerHTML = '<div class="empty-state">Select an anomaly to see its explanation</div>';
        return;
    }

    const parts = (anomaly.explanation || '').split('|').map(s => s.trim());
    const detectors = anomaly.detectors_triggered || [];
    const contributing = anomaly.contributing_params || {};

    let contribHTML = '';
    if (Object.keys(contributing).length > 0) {
        const sorted = Object.entries(contributing).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
        contribHTML = sorted.slice(0, 5).map(([k, v]) =>
            `<div class="detail-row"><span class="detail-label">${k}</span><span class="detail-value">${v > 0 ? '+' : ''}${v.toFixed(4)}</span></div>`
        ).join('');
    }

    body.innerHTML = `
        <div class="detail-row">
            <span class="detail-label">Parameter</span>
            <span class="detail-value">${anomaly.parameter || '--'}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Value</span>
            <span class="detail-value">${anomaly.value != null ? anomaly.value.toFixed(4) : '--'}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Severity</span>
            <span class="detail-value severity-badge ${anomaly.severity}">${(anomaly.severity || '').toUpperCase()}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Confidence</span>
            <span class="detail-value">${anomaly.confidence != null ? (anomaly.confidence * 100).toFixed(1) + '%' : '--'}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Detectors</span>
            <span class="detail-value">${detectors.join(', ') || '--'}</span>
        </div>
        ${contribHTML ? '<div style="margin-top:12px;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Contributing Parameters</div>' + contribHTML : ''}
        ${parts.length > 1 ? `<div class="causal-chain">${parts.slice(1).join(' | ')}</div>` : ''}
        <div class="counterfactual">What-if: ${anomaly.counterfactual || 'If this parameter returns to nominal, the alert will auto-resolve.'}</div>
    `;
}

// --- Interaction ---
window.selectAnomaly = function(id) {
    const anomaly = state.anomalies.find(a => a.id === id);
    state.selectedAnomaly = anomaly;
    renderTimeline();
    renderExplanation(anomaly);
};

window.selectAnomalyByIndex = function(index) {
    const anomaly = state.anomalies[index];
    if (!anomaly) return;
    state.selectedAnomaly = anomaly;
    renderTimeline();
    renderExplanation(anomaly);
};

function addAlert(data) {
    if (data.severity !== 'warning' && data.severity !== 'critical') return;
    const list = document.getElementById('alertList');
    if (list.querySelector('.empty-state')) list.innerHTML = '';

    const item = document.createElement('div');
    item.className = 'alert-item';
    item.innerHTML = `
        <span class="alert-time">${formatTime(data.timestamp)}</span>
        <span class="timeline-severity ${data.severity}">${data.severity.toUpperCase()}</span>
        <span class="alert-title">${data.satellite_id || '--'} / ${data.parameter || '--'}</span>
    `;
    list.prepend(item);
    if (list.children.length > 50) list.lastChild.remove();
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connStatus');
    const dot = el.querySelector('.status-dot');
    const text = el.querySelector('.status-text');
    dot.className = `status-dot ${connected ? 'online' : 'offline'}`;
    text.textContent = connected ? 'Live' : 'Reconnecting...';
}

// --- API Polling ---
async function fetchHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        document.getElementById('uptime').textContent = formatDuration(data.uptime_seconds);
    } catch (e) { /* ignore */ }
}

async function fetchAnomalies() {
    try {
        const resp = await fetch(`${API_BASE}/anomalies?limit=50`);
        const data = await resp.json();
        if (Array.isArray(data) && data.length > 0) {
            // Merge with WS data, dedup by id
            const existing = new Set(state.anomalies.map(a => a.id));
            let newAlerts = false;
            for (const a of data) {
                if (!existing.has(a.id)) {
                    state.anomalies.push(a);
                    updateSubsystemStatus(a);
                    addAlert(a);
                    newAlerts = true;
                }
            }
            state.anomalies.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            renderTimeline();
            renderMetrics();
        }
    } catch (e) { /* ignore */ }
}

async function fetchSatellites() {
    try {
        const resp = await fetch(`${API_BASE}/satellites`);
        state.satellites = await resp.json();
        renderMetrics();
    } catch (e) { /* ignore */ }
}

async function fetchStats() {
    try {
        const resp = await fetch(`${API_BASE}/stats`);
        const data = await resp.json();
        state.totalPoints = data.total_telemetry_points ?? 0;
        state.pointsLastHour = data.points_last_hour ?? 0;
        renderMetrics();
    } catch (e) { /* ignore */ }
}

// --- Utilities ---
function formatTime(ts) {
    if (!ts) return '--:--';
    const d = new Date(ts);
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDuration(seconds) {
    if (!seconds) return '--';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function formatLargeNumber(n) {
    if (!n) return '0';
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
}

// --- Filters ---
document.getElementById('severityFilter').addEventListener('change', renderTimeline);
document.getElementById('subsystemFilter').addEventListener('change', renderTimeline);
document.getElementById('clearAlerts').addEventListener('click', () => {
    document.getElementById('alertList').innerHTML = '<div class="empty-state">No alerts dispatched</div>';
});

// ---------------------------------------------------------------------------
// Channels Tab
// ---------------------------------------------------------------------------

async function loadChannels() {
    const satelliteId = document.getElementById('chanSatFilter').value;
    const url = satelliteId
        ? `${API_BASE}/channels?satellite_id=${encodeURIComponent(satelliteId)}`
        : `${API_BASE}/channels`;
    try {
        const resp = await fetch(url);
        if (!resp.ok) {
            document.getElementById('channelTableWrap').innerHTML =
                `<div class="empty-state">Error ${resp.status}: ${resp.statusText}</div>`;
            return;
        }
        state.channels = await resp.json();
        renderChannelTable();
    } catch (e) {
        document.getElementById('channelTableWrap').innerHTML =
            '<div class="empty-state">Could not load channels — is the server running?</div>';
    }
}

function renderChannelTable() {
    const wrap = document.getElementById('channelTableWrap');
    if (state.channels.length === 0) {
        wrap.innerHTML = '<div class="empty-state">No channels found. Ingest telemetry data first.</div>';
        return;
    }

    // Populate satellite filter dropdown
    const satFilter = document.getElementById('chanSatFilter');
    const knownSats = new Set(state.channels.map(c => c.satellite_id));
    knownSats.forEach(sat => {
        if (!satFilter.querySelector(`option[value="${sat}"]`)) {
            const opt = document.createElement('option');
            opt.value = sat;
            opt.textContent = sat;
            satFilter.appendChild(opt);
        }
    });

    wrap.innerHTML = `
        <table class="channel-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Subsystem</th>
                    <th>Points</th>
                    <th>Last Seen</th>
                    <th>Cal State</th>
                    <th>z-thresh</th>
                    <th>Cooldown</th>
                    <th>Override</th>
                </tr>
            </thead>
            <tbody id="channelTableBody">
                ${state.channels.map((ch, i) => renderChannelRow(ch, i)).join('')}
            </tbody>
        </table>
    `;
}

function renderChannelRow(ch, i) {
    const selected = state.selectedChannel &&
        state.selectedChannel.satellite_id === ch.satellite_id &&
        state.selectedChannel.parameter === ch.parameter ? 'selected' : '';
    const calClass = ch.calibration_state || 'unknown';
    const lastSeen = ch.last_seen ? new Date(ch.last_seen).toLocaleDateString() : '--';
    const overrideBadge = ch.has_overrides
        ? '<span class="override-badge">CUSTOM</span>'
        : '<span style="color:var(--text-muted);font-size:11px">global</span>';

    return `
        <tr class="${selected}" onclick="selectChannel(${i})">
            <td class="chan-param">${ch.parameter}</td>
            <td>${(ch.subsystem || '').toUpperCase()}</td>
            <td>${ch.total_points?.toLocaleString() || '0'}</td>
            <td>${lastSeen}</td>
            <td><span class="cal-state ${ch.calibration_state || ''}">${ch.calibration_state || '--'}</span></td>
            <td>${ch.effective_z_threshold?.toFixed(1) || '--'}</td>
            <td>${ch.effective_alert_cooldown_s != null ? ch.effective_alert_cooldown_s + 's' : '--'}</td>
            <td>${overrideBadge}</td>
        </tr>
    `;
}

window.selectChannel = function(index) {
    state.selectedChannel = state.channels[index];
    renderChannelTable();
    renderChannelEditor();
};

function renderChannelEditor() {
    const ch = state.selectedChannel;
    const editor = document.getElementById('thresholdEditor');
    if (!ch) { editor.style.display = 'none'; return; }
    editor.style.display = '';

    document.getElementById('editorTitle').textContent =
        `${ch.satellite_id} / ${ch.parameter}`;

    const effRows = [
        ['z-threshold', ch.effective_z_threshold?.toFixed(2)],
        ['min confidence', (ch.effective_min_confidence * 100)?.toFixed(0) + '%'],
        ['cooldown', ch.effective_alert_cooldown_s + 's'],
    ].map(([k, v]) => `<div class="eff-row"><span>${k}</span><span class="eff-val">${v}</span></div>`).join('');

    document.getElementById('editorBody').innerHTML = `
        <div class="editor-field">
            <div class="editor-label">
                <span>z-threshold</span>
                <span class="editor-value">global: ${ch.effective_z_threshold?.toFixed(2)}</span>
            </div>
            <input class="editor-input" id="ei-z_threshold" type="number" step="0.1" min="0.1"
                placeholder="e.g. 3.5"
                value="${ch.has_overrides ? (ch.effective_z_threshold?.toFixed(2) || '') : ''}">
        </div>
        <div class="editor-field">
            <div class="editor-label">
                <span>min confidence (0–1)</span>
                <span class="editor-value">${(ch.effective_min_confidence || 0).toFixed(2)}</span>
            </div>
            <input class="editor-input" id="ei-min_confidence" type="number" step="0.05" min="0" max="1"
                placeholder="e.g. 0.5"
                value="${ch.has_overrides ? (ch.effective_min_confidence?.toFixed(2) || '') : ''}">
        </div>
        <div class="editor-field">
            <div class="editor-label">
                <span>alert cooldown (seconds)</span>
                <span class="editor-value">${ch.effective_alert_cooldown_s}s</span>
            </div>
            <input class="editor-input" id="ei-alert_cooldown_s" type="number" step="60" min="0"
                placeholder="e.g. 3600"
                value="${ch.has_overrides ? (ch.effective_alert_cooldown_s || '') : ''}">
        </div>
        <div class="editor-actions">
            <button class="btn-primary" onclick="saveChannelConfig()">Save Overrides</button>
            <button class="btn-danger" onclick="resetChannelConfig()">Reset to Global</button>
        </div>
        <div class="editor-effective">
            <div style="margin-bottom:6px;font-size:10px;text-transform:uppercase;letter-spacing:1px">
                Effective (what detection uses)
            </div>
            ${effRows}
        </div>
    `;
}

window.saveChannelConfig = async function() {
    const ch = state.selectedChannel;
    if (!ch) return;

    const body = {};
    const z = parseFloat(document.getElementById('ei-z_threshold').value);
    const mc = parseFloat(document.getElementById('ei-min_confidence').value);
    const cd = parseInt(document.getElementById('ei-alert_cooldown_s').value, 10);
    if (!isNaN(z) && z > 0)   body.z_threshold = z;
    if (!isNaN(mc))            body.min_confidence = mc;
    if (!isNaN(cd) && cd >= 0) body.alert_cooldown_s = cd;

    if (Object.keys(body).length === 0) {
        alert('Enter at least one override value.');
        return;
    }

    try {
        const resp = await fetch(
            `${API_BASE}/channels/${encodeURIComponent(ch.satellite_id)}/${encodeURIComponent(ch.parameter)}/config`,
            { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
        );
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            alert(`Error ${resp.status}: ${err.detail || resp.statusText}`);
            return;
        }
        await loadChannels();
        // Re-select the same channel (it may be at a different index after reload)
        const idx = state.channels.findIndex(
            c => c.satellite_id === ch.satellite_id && c.parameter === ch.parameter
        );
        if (idx >= 0) { state.selectedChannel = state.channels[idx]; renderChannelEditor(); }
    } catch (e) {
        alert('Failed to save: ' + e.message);
    }
};

window.resetChannelConfig = async function() {
    const ch = state.selectedChannel;
    if (!ch) return;
    if (!ch.has_overrides) { alert('No overrides to reset.'); return; }

    try {
        const resp = await fetch(
            `${API_BASE}/channels/${encodeURIComponent(ch.satellite_id)}/${encodeURIComponent(ch.parameter)}/config`,
            { method: 'DELETE' }
        );
        if (!resp.ok) {
            alert(`Error ${resp.status}: ${resp.statusText}`);
            return;
        }
        await loadChannels();
        const idx = state.channels.findIndex(
            c => c.satellite_id === ch.satellite_id && c.parameter === ch.parameter
        );
        if (idx >= 0) { state.selectedChannel = state.channels[idx]; renderChannelEditor(); }
    } catch (e) {
        alert('Failed to reset: ' + e.message);
    }
};

// --- Tab Switching ---
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(el => {
            el.style.display = el.id === `tab-${tab}` ? '' : 'none';
        });
        if (tab === 'channels' && state.channels.length === 0) loadChannels();
    });
});

document.getElementById('refreshChannelsBtn').addEventListener('click', loadChannels);
document.getElementById('chanSatFilter').addEventListener('change', loadChannels);
document.getElementById('editorClose').addEventListener('click', () => {
    state.selectedChannel = null;
    document.getElementById('thresholdEditor').style.display = 'none';
    renderChannelTable();
});

// --- Init ---
connectWebSocket();
fetchHealth();
fetchAnomalies();
fetchSatellites();
fetchStats();

// Periodic refresh
setInterval(fetchHealth, 15000);
setInterval(fetchAnomalies, 10000);
setInterval(fetchSatellites, 30000);
setInterval(fetchStats, 30000);
