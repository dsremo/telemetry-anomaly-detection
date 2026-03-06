/**
 * Sentinel Dashboard — zero-dependency vanilla JS + Chart.js.
 *
 * Connects via WebSocket for live anomaly events.
 * Polls REST API for initial state and periodic refresh.
 * 5 tabs: Monitor, Analysis, Channels, Alerts, Import.
 */

const API_BASE = window.location.origin + '/api/v1';
const WS_SCHEME = window.location.protocol === 'https:' ? 'wss' : 'ws';
const WS_URL = `${WS_SCHEME}://${window.location.host}/api/v1/ws/live`;

// ============================================================
// State
// ============================================================
const state = {
    incidents: [],
    anomalies: [],
    satellites: [],
    selectedAnomaly: null,
    subsystemStatus: {},
    totalPoints: 0,
    pointsLastHour: 0,
    totalAnomalies: 0,
    severityCounts: {},
    connected: false,
    satFilter: '',
    // Channels tab
    channels: [],
    selectedChannel: null,
    // Analysis charts (initialized lazily)
    chartsInitialized: false,
    charts: { severity: null, rate: null },
    // Alerts tab
    alertHistory: [],
    // Import tab
    xtceFile: null,
    csvFile: null,
    // Auth
    auth: {
        accessToken:   null,
        refreshToken:  null,
        user:          null,
        mode:          null,            // null = not authenticated | 'jwt' = signed in
        tenantContext: null,           // sentinel users only: which tenant to view
    },
    // Cancellable poll intervals
    pollIntervals: { health: null, anomaly: null, satellites: null, stats: null },
};

const SUBSYSTEM_META = {
    eps:     { label: 'EPS',     desc: 'Electrical Power' },
    adcs:    { label: 'ADCS',    desc: 'Attitude Control' },
    thermal: { label: 'THERMAL', desc: 'Thermal Control' },
    comms:   { label: 'COMMS',   desc: 'Communications' },
    payload: { label: 'PAYLOAD', desc: 'Payload Systems' },
    obc:     { label: 'OBC',     desc: 'On-Board Computer' },
    gnc:     { label: 'GNC',     desc: 'Guidance & Navigation' },
};

const SEVERITY_RANK = { nominal: 0, watch: 1, warning: 2, critical: 3 };

// ============================================================
// Toast Notifications
// ============================================================
function toast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => {
        el.classList.remove('show');
        setTimeout(() => el.remove(), 300);
    }, 3500);
}

// ============================================================
// WebSocket
// ============================================================
let ws = null;
let reconnectTimer = null;
let pingInterval = null;

function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        state.connected = true;
        updateConnectionStatus(true);
        if (pingInterval) clearInterval(pingInterval);
        pingInterval = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        }, 30000);
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
        clearInterval(pingInterval);
        reconnectTimer = setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => ws.close();
}

// ============================================================
// Anomaly Event Handler
// ============================================================
function handleAnomalyEvent(data) {
    const existing = state.anomalies.findIndex(a => a.id === data.id);
    if (existing >= 0) {
        state.anomalies[existing] = data;
    } else {
        state.anomalies.unshift(data);
    }
    if (state.anomalies.length > 500) state.anomalies.length = 500;

    updateSubsystemStatus(data);
    renderTimeline();
    renderMetrics();
    renderSeverityBar();
    addAlert(data);
    maybeUpdateAnalysis();
}

function updateSubsystemStatus(anomaly) {
    const sub = anomaly.subsystem || guessSubsystem(anomaly.parameter);
    if (!sub) return;
    const current = state.subsystemStatus[sub] || 'nominal';
    if (SEVERITY_RANK[anomaly.severity] > SEVERITY_RANK[current]) {
        state.subsystemStatus[sub] = anomaly.severity;
        renderSubsystems();
    }
}

function guessSubsystem(param) {
    if (!param) return '';
    const p = param.toLowerCase();
    if (p.includes('battery') || p.includes('solar') || p.includes('bus_v') || p.includes('current')) return 'eps';
    if (p.includes('wheel') || p.includes('pointing') || p.includes('attitude') || p.includes('torque')) return 'adcs';
    if (p.includes('temp') || p.includes('panel') || p.includes('thermal') || p.includes('heater')) return 'thermal';
    if (p.includes('signal') || p.includes('bit') || p.includes('link') || p.includes('rx') || p.includes('tx')) return 'comms';
    if (p.includes('cpu') || p.includes('memory') || p.includes('ram') || p.includes('reboot')) return 'obc';
    return '';
}

// ============================================================
// Monitor Tab — Timeline
// ============================================================
function renderTimeline() {
    const list = document.getElementById('timelineList');
    const sevFilter = document.getElementById('severityFilter').value;
    const subFilter = document.getElementById('subsystemFilter').value;
    const searchTerm = (document.getElementById('anomalySearch')?.value || '').toLowerCase();

    let filtered = state.satFilter
        ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
        : state.anomalies;

    if (sevFilter) filtered = filtered.filter(a => a.severity === sevFilter);
    if (subFilter) filtered = filtered.filter(a =>
        (a.subsystem || guessSubsystem(a.parameter)) === subFilter
    );
    if (searchTerm) filtered = filtered.filter(a =>
        (a.parameter || '').toLowerCase().includes(searchTerm) ||
        (a.satellite_id || '').toLowerCase().includes(searchTerm) ||
        (a.explanation || '').toLowerCase().includes(searchTerm)
    );

    if (filtered.length === 0) {
        list.innerHTML = '<div class="empty-state">No anomalies match the current filters.</div>';
        return;
    }

    list.innerHTML = filtered.map(a => {
        const idx = state.anomalies.indexOf(a);
        const selected = state.selectedAnomaly?.id === a.id ? 'selected' : '';
        const explanation = (a.explanation || '').split('|')[0].trim();
        return `
            <div class="timeline-item ${selected}" onclick="selectAnomalyByIndex(${idx})">
                <span class="timeline-time">${formatDateTime(a.timestamp)}</span>
                <span class="timeline-dot ${a.severity}"></span>
                <div class="timeline-content">
                    <div class="timeline-param">${a.satellite_id ? `<span style="color:var(--text-muted);font-size:11px">${a.satellite_id} /</span> ` : ''}${a.parameter || 'unknown'}</div>
                    <div class="timeline-explanation">${explanation}</div>
                </div>
                <span class="timeline-severity ${a.severity}">${(a.severity || '').toUpperCase()}</span>
            </div>`;
    }).join('');
}

// ============================================================
// Monitor Tab — Metrics
// ============================================================
function renderMetrics() {
    const relevant = state.satFilter
        ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
        : state.anomalies;

    // Use server-side total if available, otherwise fall back to loaded count
    const totalFromServer = state.totalAnomalies || 0;
    const displayTotal = totalFromServer || relevant.length;
    document.getElementById('anomalyCount').textContent = displayTotal;
    document.getElementById('anomalyTrend').textContent =
        totalFromServer > relevant.length
            ? `${relevant.length} of ${totalFromServer} loaded`
            : `${relevant.length} loaded`;

    // Highest severity — from server-side severity counts (covers all anomalies, not just loaded)
    const sc = state.severityCounts || {};
    let maxSev = 'nominal';
    if ((sc['watch']    || 0) > 0) maxSev = 'watch';
    if ((sc['warning']  || 0) > 0) maxSev = 'warning';
    if ((sc['critical'] || 0) > 0) maxSev = 'critical';
    // Parameter: find from loaded anomalies (best effort)
    let maxParam = '--';
    for (const a of relevant) {
        if (a.severity === maxSev) { maxParam = a.parameter || ''; break; }
    }
    const badge = document.getElementById('maxSeverity');
    badge.textContent = maxSev.toUpperCase();
    badge.className = `metric-value severity-badge ${maxSev}`;
    if (maxSev === 'critical') badge.classList.add('pulse');
    document.getElementById('severityParam').textContent = maxParam;

    // Satellite count
    const sats = state.satFilter ? [state.satFilter] : state.satellites;
    document.getElementById('satCount').textContent = sats.length || '0';
    document.getElementById('satList').textContent = sats.slice(0, 3).join(', ') || '--';

    // Telemetry points
    document.getElementById('totalPoints').textContent = formatLargeNumber(state.totalPoints);
    document.getElementById('pointsTrend').textContent = `${formatLargeNumber(state.pointsLastHour)}/hr`;
}

// ============================================================
// Monitor Tab — Severity Bar
// ============================================================
function renderSeverityBar() {
    // Use server-side severity counts (covers ALL anomalies, not just the loaded slice)
    // Fall back to computing from loaded anomalies if stats not yet available
    let counts, total;
    const sc = state.severityCounts;
    if (sc && (sc.critical || sc.warning || sc.watch)) {
        counts = {
            critical: sc.critical || 0,
            warning:  sc.warning  || 0,
            watch:    sc.watch    || 0,
        };
        total = counts.critical + counts.warning + counts.watch;
    } else {
        const relevant = state.satFilter
            ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
            : state.anomalies;
        total = relevant.length;
        counts = { critical: 0, warning: 0, watch: 0 };
        for (const a of relevant) {
            if (counts[a.severity] !== undefined) counts[a.severity]++;
        }
    }

    document.getElementById('severityTotal').textContent = `${total.toLocaleString()} anomalies`;
    // Legend below bar
    if (document.getElementById('sevLegCritical')) {
        document.getElementById('sevLegCritical').textContent = counts.critical.toLocaleString();
        document.getElementById('sevLegWarning').textContent  = counts.warning.toLocaleString();
        document.getElementById('sevLegWatch').textContent    = counts.watch.toLocaleString();
    }

    if (total === 0) {
        document.getElementById('sevCritical').style.width = '0%';
        document.getElementById('sevWarning').style.width  = '0%';
        document.getElementById('sevWatch').style.width    = '0%';
        document.getElementById('sevNominal').style.width  = '100%';
        return;
    }

    const pct = n => `${(n / total * 100).toFixed(1)}%`;
    const critEl = document.getElementById('sevCritical');
    const warnEl = document.getElementById('sevWarning');
    const watchEl = document.getElementById('sevWatch');
    critEl.style.width  = pct(counts.critical);
    critEl.title = `Critical: ${counts.critical.toLocaleString()} (${pct(counts.critical)})`;
    warnEl.style.width  = pct(counts.warning);
    warnEl.title = `Warning: ${counts.warning.toLocaleString()} (${pct(counts.warning)})`;
    watchEl.style.width = pct(counts.watch);
    watchEl.title = `Watch: ${counts.watch.toLocaleString()} (${pct(counts.watch)})`;
    const nominalPct = ((total - counts.critical - counts.warning - counts.watch) / total * 100);
    document.getElementById('sevNominal').style.width  = `${Math.max(0, nominalPct).toFixed(1)}%`;
}

// ============================================================
// Monitor Tab — Subsystems (dynamic)
// ============================================================
function renderSubsystems() {
    const defaults = ['eps', 'adcs', 'thermal', 'comms'];
    const allSubs = new Set(defaults);
    for (const a of state.anomalies) {
        const sub = a.subsystem || guessSubsystem(a.parameter);
        if (sub) allSubs.add(sub);
    }

    const grid = document.getElementById('subsystemGrid');
    grid.innerHTML = [...allSubs].map(sub => {
        const status = state.subsystemStatus[sub] || 'nominal';
        const meta = SUBSYSTEM_META[sub] || { label: sub.toUpperCase(), desc: sub };
        return `
            <div class="subsystem-card ${status === 'critical' ? 'alert' : ''}" data-subsystem="${sub}">
                <div class="subsystem-name">${meta.label}</div>
                <div class="subsystem-status ${status}">${status.toUpperCase()}</div>
                <div class="subsystem-desc">${meta.desc}</div>
            </div>`;
    }).join('');
}

// ============================================================
// Monitor Tab — Explanation
// ============================================================
function renderExplanation(anomaly) {
    const body = document.getElementById('explanationBody');
    const exportBtn = document.getElementById('exportAnomalyBtn');
    if (!anomaly) {
        body.innerHTML = '<div class="empty-state">Select an anomaly to see its explanation</div>';
        exportBtn.style.display = 'none';
        return;
    }
    exportBtn.style.display = '';

    const parts = (anomaly.explanation || '').split('|').map(s => s.trim());
    const detectors = anomaly.detectors_triggered || [];
    const contributing = anomaly.contributing_params || {};

    let contribHTML = '';
    if (Object.keys(contributing).length > 0) {
        const sorted = Object.entries(contributing).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
        contribHTML = `<div style="margin-top:12px;font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Contributing Parameters</div>` +
            sorted.slice(0, 5).map(([k, v]) =>
                `<div class="detail-row"><span class="detail-label">${k}</span><span class="detail-value">${v > 0 ? '+' : ''}${v.toFixed(4)}</span></div>`
            ).join('');
    }

    body.innerHTML = `
        <div class="detail-row"><span class="detail-label">Satellite</span><span class="detail-value">${anomaly.satellite_id || '--'}</span></div>
        <div class="detail-row"><span class="detail-label">Parameter</span><span class="detail-value">${anomaly.parameter || '--'}</span></div>
        <div class="detail-row"><span class="detail-label">Value</span><span class="detail-value">${anomaly.value != null ? anomaly.value.toFixed(4) : '--'}</span></div>
        <div class="detail-row"><span class="detail-label">Severity</span><span class="detail-value severity-badge ${anomaly.severity}">${(anomaly.severity || '').toUpperCase()}</span></div>
        <div class="detail-row"><span class="detail-label">Confidence</span><span class="detail-value">${anomaly.confidence != null ? (anomaly.confidence * 100).toFixed(1) + '%' : '--'}</span></div>
        <div class="detail-row"><span class="detail-label">Detectors</span><span class="detail-value">${detectors.join(', ') || '--'}</span></div>
        <div class="detail-row"><span class="detail-label">Time</span><span class="detail-value">${formatDateTime(anomaly.timestamp)}</span></div>
        ${contribHTML}
        ${parts.length > 1 ? `<div class="causal-chain">${parts.slice(1).join(' → ')}</div>` : ''}
        <div class="counterfactual">What-if: ${anomaly.counterfactual || 'If this parameter returns to nominal, the alert will auto-resolve.'}</div>
    `;
}

// ============================================================
// Anomaly Selection
// ============================================================
window.selectAnomalyByIndex = function(index) {
    const anomaly = state.anomalies[index];
    if (!anomaly) return;
    state.selectedAnomaly = anomaly;
    renderTimeline();
    renderExplanation(anomaly);
};

// ============================================================
// Alert Log (Monitor tab bottom strip)
// ============================================================
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
    el.querySelector('.status-dot').className = `status-dot ${connected ? 'online' : 'offline'}`;
    el.querySelector('.status-text').textContent = connected ? 'Live' : 'Reconnecting...';
}

// ============================================================
// Analysis Tab — Chart.js Integration
// ============================================================
function initCharts() {
    if (state.chartsInitialized) return;
    state.chartsInitialized = true;

    // Global defaults for dark theme
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#2a3a4e';
    Chart.defaults.font.family = "'SF Mono', 'Fira Code', 'Consolas', monospace";
    Chart.defaults.font.size = 11;

    // Severity Donut
    const donutCtx = document.getElementById('severityDonut').getContext('2d');
    state.charts.severity = new Chart(donutCtx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'Warning', 'Watch'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#ef4444', '#f97316', '#eab308'],
                borderWidth: 0,
                hoverOffset: 6,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '68%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => ` ${ctx.label}: ${ctx.parsed} anomalies`
                    }
                }
            }
        }
    });

    // Anomaly Rate Bar Chart
    const rateCtx = document.getElementById('rateBarChart').getContext('2d');
    state.charts.rate = new Chart(rateCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                { label: 'Critical', data: [], backgroundColor: '#ef4444', stack: 'stack' },
                { label: 'Warning',  data: [], backgroundColor: '#f97316', stack: 'stack' },
                { label: 'Watch',    data: [], backgroundColor: '#eab308', stack: 'stack' },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                    grid: { color: '#2a3a4e' },
                    ticks: { color: '#64748b', maxRotation: 45 },
                },
                y: {
                    stacked: true,
                    grid: { color: '#2a3a4e' },
                    ticks: { color: '#64748b', precision: 0 },
                    beginAtZero: true,
                },
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', boxWidth: 12, padding: 16 }
                }
            }
        }
    });

    updateAnalysis();
}

function updateAnalysis() {
    updateDonutChart();
    updateRateChart();
    updateTopLists();
}

function maybeUpdateAnalysis() {
    if (state.chartsInitialized) {
        const tab = document.getElementById('tab-analysis');
        if (tab && tab.style.display !== 'none') updateAnalysis();
    }
}

function updateDonutChart() {
    const chart = state.charts.severity;
    if (!chart) return;

    const relevant = state.satFilter
        ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
        : state.anomalies;

    const counts = { critical: 0, warning: 0, watch: 0 };
    for (const a of relevant) {
        if (counts[a.severity] !== undefined) counts[a.severity]++;
    }

    chart.data.datasets[0].data = [counts.critical, counts.warning, counts.watch];
    chart.update('none');

    const total = counts.critical + counts.warning + counts.watch;
    document.getElementById('donutTotal').textContent = `${total} total`;

    const legend = document.getElementById('donutLegend');
    legend.innerHTML = [
        { label: 'Critical', count: counts.critical, color: '#ef4444' },
        { label: 'Warning',  count: counts.warning,  color: '#f97316' },
        { label: 'Watch',    count: counts.watch,    color: '#eab308' },
    ].map(({ label, count, color }) => {
        const pct = total > 0 ? `${(count / total * 100).toFixed(0)}%` : '0%';
        return `
            <div class="legend-item">
                <div class="legend-dot" style="background:${color}"></div>
                <span class="legend-name">${label}</span>
                <span class="legend-val">${count}</span>
                <span class="legend-pct">${pct}</span>
            </div>`;
    }).join('');
}

function updateRateChart() {
    const chart = state.charts.rate;
    if (!chart) return;

    const range = document.getElementById('rateChartRange').value;
    const relevant = state.satFilter
        ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
        : state.anomalies;

    if (!relevant.length) {
        chart.data.labels = [];
        chart.data.datasets.forEach(d => { d.data = []; });
        chart.update('none');
        return;
    }

    // Anchor to the most recent anomaly's timestamp so historical data renders correctly.
    // (Data may predate "now" by months or years — using new Date() would produce empty buckets.)
    const times = relevant.map(a => new Date(a.timestamp).getTime()).filter(t => !isNaN(t));
    const dataMax = new Date(Math.max(...times));

    let labels, buckets;

    if (range === '24h') {
        labels = [];
        buckets = { critical: [], warning: [], watch: [] };
        for (let i = 23; i >= 0; i--) {
            const start = dataMax.getTime() - (i + 1) * 3600000;
            const end   = dataMax.getTime() - i * 3600000;
            const h = new Date(start).getHours().toString().padStart(2, '0');
            labels.push(h + ':00');
            for (const sev of ['critical', 'warning', 'watch']) {
                buckets[sev].push(relevant.filter(a => {
                    const t = new Date(a.timestamp).getTime();
                    return a.severity === sev && t >= start && t < end;
                }).length);
            }
        }
    } else {
        labels = [];
        buckets = { critical: [], warning: [], watch: [] };
        for (let i = 6; i >= 0; i--) {
            const ref = new Date(dataMax);
            ref.setDate(ref.getDate() - i);
            const start = new Date(ref); start.setHours(0, 0, 0, 0);
            const end   = new Date(ref); end.setHours(23, 59, 59, 999);
            labels.push(start.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' }));
            for (const sev of ['critical', 'warning', 'watch']) {
                buckets[sev].push(relevant.filter(a => {
                    const t = new Date(a.timestamp).getTime();
                    return a.severity === sev && t >= start.getTime() && t <= end.getTime();
                }).length);
            }
        }
    }

    chart.data.labels = labels;
    chart.data.datasets[0].data = buckets.critical;
    chart.data.datasets[1].data = buckets.warning;
    chart.data.datasets[2].data = buckets.watch;
    chart.update('none');
}

function updateTopLists() {
    const relevant = state.satFilter
        ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
        : state.anomalies;

    // Top channels
    const byChan = {};
    for (const a of relevant) {
        const key = a.parameter || 'unknown';
        byChan[key] = (byChan[key] || 0) + 1;
    }
    renderTopList('topChannels', byChan, 8);

    // Top satellites
    const bySat = {};
    for (const a of relevant) {
        const key = a.satellite_id || 'unknown';
        bySat[key] = (bySat[key] || 0) + 1;
    }
    renderTopList('topSatellites', bySat, 8);

    // Top detectors
    const byDet = {};
    for (const a of relevant) {
        for (const d of (a.detectors_triggered || [])) {
            byDet[d] = (byDet[d] || 0) + 1;
        }
    }
    renderTopList('topDetectors', byDet, 8);
}

function renderTopList(elId, counts, limit) {
    const el = document.getElementById(elId);
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, limit);
    if (sorted.length === 0) {
        el.innerHTML = '<div class="empty-state">No anomaly data yet</div>';
        return;
    }
    const max = sorted[0][1];
    el.innerHTML = sorted.map(([name, count], i) => `
        <div class="top-item">
            <span class="top-rank">${i + 1}</span>
            <span class="top-name" title="${name}">${name}</span>
            <div class="top-bar-bg"><div class="top-bar-fill" style="width:${(count / max * 100).toFixed(0)}%"></div></div>
            <span class="top-count">${count}</span>
        </div>`
    ).join('');
}

// ============================================================
// Channels Tab
// ============================================================
async function loadChannels() {
    const satelliteId = document.getElementById('chanSatFilter').value;
    const url = satelliteId
        ? `${API_BASE}/channels?satellite_id=${encodeURIComponent(satelliteId)}`
        : `${API_BASE}/channels`;
    try {
        const resp = await fetch(url, { headers: authHeaders() });
        if (!resp.ok) {
            document.getElementById('channelTableWrap').innerHTML =
                `<div class="empty-state">Error ${resp.status}: ${resp.statusText}</div>`;
            return;
        }
        state.channels = await resp.json();
        document.getElementById('channelCount').textContent = state.channels.length;
        document.getElementById('channelTrend').textContent = 'parameters';
        renderChannelTable();
    } catch (e) {
        document.getElementById('channelTableWrap').innerHTML =
            '<div class="empty-state">Could not load channels — is the server running?</div>';
    }
}

function renderChannelTable() {
    const wrap = document.getElementById('channelTableWrap');
    if (state.channels.length === 0) {
        wrap.innerHTML = '<div class="empty-state">No channels found. Import telemetry or XTCE first.</div>';
        return;
    }

    const satFilter = document.getElementById('chanSatFilter');
    const knownSats = new Set(state.channels.map(c => c.satellite_id));
    knownSats.forEach(sat => {
        if (!satFilter.querySelector(`option[value="${sat}"]`)) {
            const opt = document.createElement('option');
            opt.value = sat; opt.textContent = sat;
            satFilter.appendChild(opt);
        }
    });

    wrap.innerHTML = `
        <table class="channel-table">
            <thead>
                <tr>
                    <th>Parameter</th><th>Subsystem</th><th>Points</th>
                    <th>Last Seen</th><th>Cal State</th>
                    <th>z-thresh</th><th>Cooldown</th><th>Override</th><th></th>
                </tr>
            </thead>
            <tbody>${state.channels.map((ch, i) => renderChannelRow(ch, i)).join('')}</tbody>
        </table>`;
}

function renderChannelRow(ch, i) {
    const selected = state.selectedChannel &&
        state.selectedChannel.satellite_id === ch.satellite_id &&
        state.selectedChannel.parameter === ch.parameter ? 'selected' : '';
    const lastSeen = ch.last_seen ? new Date(ch.last_seen).toLocaleDateString() : '--';
    const overrideBadge = ch.has_overrides
        ? '<span class="override-badge">CUSTOM</span>'
        : '<span style="color:var(--text-muted);font-size:11px">global</span>';
    return `
        <tr class="${selected}" onclick="selectChannel(${i})" style="cursor:pointer">
            <td class="chan-param">${ch.parameter}</td>
            <td>${(ch.subsystem || '').toUpperCase()}</td>
            <td>${ch.total_points?.toLocaleString() || '0'}</td>
            <td>${lastSeen}</td>
            <td><span class="cal-state ${ch.calibration_state || ''}">${ch.calibration_state || '--'}</span></td>
            <td>${ch.effective_z_threshold?.toFixed(1) || '--'}</td>
            <td>${ch.effective_alert_cooldown_s != null ? ch.effective_alert_cooldown_s + 's' : '--'}</td>
            <td>${overrideBadge}</td>
            <td><button class="btn-secondary" style="padding:2px 8px;font-size:11px"
                onclick="event.stopPropagation();selectChannel(${i})">Edit</button></td>
        </tr>`;
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

    document.getElementById('editorTitle').textContent = `${ch.satellite_id} / ${ch.parameter}`;

    const effRows = [
        ['z-threshold',   ch.effective_z_threshold?.toFixed(2)],
        ['min confidence', (ch.effective_min_confidence * 100)?.toFixed(0) + '%'],
        ['cooldown',       ch.effective_alert_cooldown_s + 's'],
    ].map(([k, v]) => `<div class="eff-row"><span>${k}</span><span class="eff-val">${v}</span></div>`).join('');

    document.getElementById('editorBody').innerHTML = `
        <div class="editor-field">
            <div class="editor-label">
                <span>z-threshold</span>
                <span class="editor-value">current: ${ch.effective_z_threshold?.toFixed(2)}</span>
            </div>
            <input class="editor-input" id="ei-z_threshold" type="number" step="0.1" min="0.1"
                placeholder="e.g. 3.5" value="${ch.has_overrides ? (ch.effective_z_threshold?.toFixed(2) || '') : ''}">
        </div>
        <div class="editor-field">
            <div class="editor-label">
                <span>min confidence (0–1)</span>
                <span class="editor-value">${(ch.effective_min_confidence || 0).toFixed(2)}</span>
            </div>
            <input class="editor-input" id="ei-min_confidence" type="number" step="0.05" min="0" max="1"
                placeholder="e.g. 0.5" value="${ch.has_overrides ? (ch.effective_min_confidence?.toFixed(2) || '') : ''}">
        </div>
        <div class="editor-field">
            <div class="editor-label">
                <span>alert cooldown (seconds)</span>
                <span class="editor-value">${ch.effective_alert_cooldown_s}s</span>
            </div>
            <input class="editor-input" id="ei-alert_cooldown_s" type="number" step="60" min="0"
                placeholder="e.g. 3600" value="${ch.has_overrides ? (ch.effective_alert_cooldown_s || '') : ''}">
        </div>
        <div class="editor-actions">
            <button class="btn-primary" onclick="saveChannelConfig()">Save Overrides</button>
            <button class="btn-danger" onclick="resetChannelConfig()">Reset to Global</button>
        </div>
        <div class="editor-effective">
            <div style="margin-bottom:6px;font-size:10px;text-transform:uppercase;letter-spacing:1px">
                Effective values (used in detection)
            </div>
            ${effRows}
        </div>`;
}

window.saveChannelConfig = async function() {
    const ch = state.selectedChannel;
    if (!ch) return;

    const body = {};
    const z  = parseFloat(document.getElementById('ei-z_threshold').value);
    const mc = parseFloat(document.getElementById('ei-min_confidence').value);
    const cd = parseInt(document.getElementById('ei-alert_cooldown_s').value, 10);
    if (!isNaN(z) && z > 0)    body.z_threshold = z;
    if (!isNaN(mc))             body.min_confidence = mc;
    if (!isNaN(cd) && cd >= 0) body.alert_cooldown_s = cd;

    if (Object.keys(body).length === 0) { toast('Enter at least one override value.', 'warning'); return; }

    try {
        const resp = await fetch(
            `${API_BASE}/channels/${encodeURIComponent(ch.satellite_id)}/${encodeURIComponent(ch.parameter)}/config`,
            { method: 'PUT', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify(body) }
        );
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            toast(`Error ${resp.status}: ${err.detail || resp.statusText}`, 'error');
            return;
        }
        toast('Threshold config saved.', 'success');
        await loadChannels();
        const idx = state.channels.findIndex(
            c => c.satellite_id === ch.satellite_id && c.parameter === ch.parameter
        );
        if (idx >= 0) { state.selectedChannel = state.channels[idx]; renderChannelEditor(); }
    } catch (e) {
        toast('Failed to save: ' + e.message, 'error');
    }
};

window.resetChannelConfig = async function() {
    const ch = state.selectedChannel;
    if (!ch) return;
    if (!ch.has_overrides) { toast('No overrides to reset.', 'info'); return; }
    try {
        const resp = await fetch(
            `${API_BASE}/channels/${encodeURIComponent(ch.satellite_id)}/${encodeURIComponent(ch.parameter)}/config`,
            { method: 'DELETE', headers: authHeaders() }
        );
        if (!resp.ok) { toast(`Error ${resp.status}: ${resp.statusText}`, 'error'); return; }
        toast('Reset to global thresholds.', 'success');
        await loadChannels();
        const idx = state.channels.findIndex(
            c => c.satellite_id === ch.satellite_id && c.parameter === ch.parameter
        );
        if (idx >= 0) { state.selectedChannel = state.channels[idx]; renderChannelEditor(); }
    } catch (e) {
        toast('Failed to reset: ' + e.message, 'error');
    }
};

// ============================================================
// Anomaly Explorer (Sprint 12: ML Visibility + Operator Feedback)
// ============================================================

// Detector label + CSS class map
const DET_META = {
    cusum:            ['CUSUM', 'cusum'],
    ewma:             ['EWMA',  'ewma'],
    statistical:      ['STAT',  'stat'],
    changepoint:      ['CPLT',  'cp'],
    isolation_forest: ['ISO',   'iso'],
    variance:         ['VAR',   'var'],
    lstm:             ['GRU',   'ml'],
    tcn:              ['TCN',   'ml'],
};

function _detBadges(detectors) {
    if (!detectors || detectors.length === 0)
        return '<span style="color:var(--text-muted);font-size:11px">none</span>';
    return detectors.map(d => {
        const [label, cls] = DET_META[d] || [d.toUpperCase().slice(0, 5), ''];
        return `<span class="det-badge ${cls}" title="${d}">${label}</span>`;
    }).join('');
}

async function loadAnomalies() {
    const satId  = document.getElementById('anomSatFilter').value;
    const sev    = document.getElementById('anomSevFilter').value;
    const mlOnly = document.getElementById('anomMLFilter').value;
    let url = `${API_BASE}/anomalies?limit=100`;
    if (satId)        url += `&satellite_id=${encodeURIComponent(satId)}`;
    if (sev)          url += `&severity=${encodeURIComponent(sev)}`;
    if (mlOnly !== '') url += `&ml_only=${mlOnly}`;
    try {
        const resp = await fetch(url, { headers: authHeaders() });
        if (!resp.ok) {
            document.getElementById('anomalyExplorerWrap').innerHTML =
                `<div class="empty-state">Could not load anomalies (${resp.status}).</div>`;
            return;
        }
        state.anomalies = await resp.json();
        renderAnomalies();
    } catch (e) {
        document.getElementById('anomalyExplorerWrap').innerHTML =
            '<div class="empty-state">Could not load anomalies.</div>';
    }
}

function renderAnomalies() {
    const wrap     = document.getElementById('anomalyExplorerWrap');
    const chip     = document.getElementById('anomalyMLChip');
    const anomalies = state.anomalies || [];
    const mlCount   = anomalies.filter(a => a.ml_only).length;

    // Update ML count chip
    if (chip) {
        if (mlCount > 0) {
            chip.textContent = `${mlCount} ML Pattern${mlCount !== 1 ? 's' : ''}`;
            chip.style.display = '';
        } else {
            chip.style.display = 'none';
        }
    }

    if (anomalies.length === 0) {
        wrap.innerHTML = '<div class="empty-state">No anomalies found.</div>';
        return;
    }

    wrap.innerHTML = `
        <table class="channel-table">
            <thead>
                <tr>
                    <th>Time</th><th>Satellite</th><th>Parameter</th>
                    <th>Severity</th><th>Conf</th><th>Detectors</th><th>Feedback</th>
                </tr>
            </thead>
            <tbody>
                ${anomalies.map(a => {
                    const ts  = formatDateTime(a.timestamp);
                    const sev = a.severity || 'nominal';
                    const fp  = a.false_positive;
                    const rev = a.reviewed;
                    const mlBadge = a.ml_only
                        ? '<span class="det-badge ml-only" title="Subtle temporal pattern — statistics did not flag this">ML PATTERN</span>'
                        : '';
                    const tpCls = rev && !fp ? 'tp active-tp' : 'tp';
                    const fpCls = rev && fp  ? 'fp active-fp' : 'fp';
                    const explainTitle = (a.explanation || '').replace(/"/g, '&quot;');
                    return `<tr title="${explainTitle}">
                        <td>${ts}</td>
                        <td>${a.satellite_id}</td>
                        <td class="chan-param">${a.parameter} ${mlBadge}</td>
                        <td><span class="timeline-severity ${sev}">${sev.toUpperCase()}</span></td>
                        <td style="color:var(--accent)">${(a.confidence * 100).toFixed(0)}%</td>
                        <td>${_detBadges(a.detectors_triggered)}</td>
                        <td>
                            <button class="fb-btn ${tpCls}" title="Mark True Positive (confirmed anomaly)"
                                onclick="submitFeedback('${a.id}', true)">&#10003;</button>
                            <button class="fb-btn ${fpCls}" title="Mark False Positive (not a real anomaly)"
                                onclick="submitFeedback('${a.id}', false)">&#10007;</button>
                        </td>
                    </tr>`;
                }).join('')}
            </tbody>
        </table>`;
}

window.submitFeedback = async function(anomalyId, isTP) {
    const verdict = isTP ? 'true_positive' : 'false_positive';
    try {
        const resp = await fetch(`${API_BASE}/anomalies/${anomalyId}/feedback`, {
            method: 'PATCH',
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
            body: JSON.stringify({ verdict }),
        });
        if (!resp.ok) { toast(`Feedback failed (${resp.status})`, 'error'); return; }
        toast(isTP ? '✓ Marked as true positive' : '✗ Marked as false positive', 'success');
        await loadAnomalies();   // re-render with updated reviewed state
    } catch (e) {
        toast('Feedback error: ' + e.message, 'error');
    }
};

// ============================================================
// Incident Explorer (Sprint 17 — Hierarchical Alert Routing)
// ============================================================
async function loadIncidents() {
    const satId  = document.getElementById('incSatFilter').value;
    const status = document.getElementById('incStatusFilter').value;
    let url = `${API_BASE}/incidents?limit=100`;
    if (satId)  url += `&satellite_id=${encodeURIComponent(satId)}`;
    if (status) url += `&status=${encodeURIComponent(status)}`;
    try {
        const resp = await fetch(url, { headers: authHeaders() });
        if (!resp.ok) {
            document.getElementById('incidentExplorerWrap').innerHTML =
                `<div class="empty-state">Could not load incidents (${resp.status}).</div>`;
            return;
        }
        state.incidents = await resp.json();
        renderIncidents();
    } catch (e) {
        document.getElementById('incidentExplorerWrap').innerHTML =
            '<div class="empty-state">Could not load incidents.</div>';
    }
}

function renderIncidents() {
    const wrap      = document.getElementById('incidentExplorerWrap');
    const incidents = state.incidents || [];
    if (incidents.length === 0) {
        wrap.innerHTML = '<div class="empty-state">No incidents found.</div>';
        return;
    }
    const SEV_ORDER = { critical: 0, warning: 1, watch: 2, nominal: 3 };
    const sorted = [...incidents].sort(
        (a, b) => (SEV_ORDER[a.severity] ?? 3) - (SEV_ORDER[b.severity] ?? 3)
    );
    wrap.innerHTML = `
        <table class="channel-table">
            <thead>
                <tr>
                    <th>First Seen</th><th>Satellite</th><th>Severity</th>
                    <th>Channels</th><th>Anomalies</th><th>Confidence</th>
                    <th>Root Cause</th><th>Status</th><th>Action</th>
                </tr>
            </thead>
            <tbody>
                ${sorted.map(inc => {
                    const ts      = formatDateTime(inc.first_anomaly_at);
                    const sev     = inc.severity || 'watch';
                    const chans   = (inc.channels || []).slice(0, 3).join(', ')
                                  + (inc.channels.length > 3 ? ` +${inc.channels.length - 3}` : '');
                    const conf    = ((inc.confidence || 0) * 100).toFixed(0) + '%';
                    const rootCause = (inc.root_cause_summary || '').replace(/"/g, '&quot;');
                    const statusBadge = inc.status === 'open'
                        ? '<span style="color:var(--warning)">Open</span>'
                        : inc.status === 'resolved'
                            ? '<span style="color:var(--nominal)">Resolved</span>'
                            : '<span style="color:var(--text-muted)">FP</span>';
                    const actionBtn = inc.status === 'open'
                        ? `<button class="btn-secondary" style="padding:2px 8px;font-size:0.8rem"
                               onclick="resolveIncident('${inc.id}')">Resolve</button>`
                        : '—';
                    return `<tr title="${rootCause}">
                        <td>${ts}</td>
                        <td>${inc.satellite_id}</td>
                        <td><span class="timeline-severity ${sev}">${sev.toUpperCase()}</span></td>
                        <td class="chan-param">${chans || '—'}</td>
                        <td style="text-align:center">${inc.anomaly_count}</td>
                        <td style="color:var(--accent)">${conf}</td>
                        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                            title="${rootCause}">${inc.root_cause_summary || '—'}</td>
                        <td>${statusBadge}</td>
                        <td>${actionBtn}</td>
                    </tr>`;
                }).join('')}
            </tbody>
        </table>`;
}

window.resolveIncident = async function(incidentId) {
    try {
        const resp = await fetch(`${API_BASE}/incidents/${incidentId}/status`, {
            method: 'PATCH',
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'resolved' }),
        });
        if (!resp.ok) { toast(`Could not resolve incident (${resp.status})`, 'error'); return; }
        toast('Incident resolved', 'success');
        await loadIncidents();
    } catch (e) {
        toast('Error: ' + e.message, 'error');
    }
};

// ============================================================
// Alerts Tab
// ============================================================
async function loadAlertHistory() {
    const satId = document.getElementById('alertSatFilter').value;
    const sev   = document.getElementById('alertSevFilter').value;
    let url = `${API_BASE}/alerts/history?limit=100`;
    if (satId) url += `&satellite_id=${encodeURIComponent(satId)}`;
    if (sev)   url += `&severity=${encodeURIComponent(sev)}`;

    try {
        const resp = await fetch(url, { headers: authHeaders() });
        if (!resp.ok) {
            document.getElementById('alertHistoryWrap').innerHTML =
                `<div class="empty-state">Alert history unavailable (${resp.status}). Configure alert delivery to enable.</div>`;
            return;
        }
        state.alertHistory = await resp.json();
        renderAlertHistory();
    } catch (e) {
        document.getElementById('alertHistoryWrap').innerHTML =
            '<div class="empty-state">Could not load alert history. Is the server running?</div>';
    }
}

function renderAlertHistory() {
    const wrap = document.getElementById('alertHistoryWrap');
    if (!state.alertHistory || state.alertHistory.length === 0) {
        wrap.innerHTML = '<div class="empty-state">No alerts dispatched yet.</div>';
        return;
    }
    wrap.innerHTML = `
        <table class="channel-table">
            <thead>
                <tr>
                    <th>Time</th><th>Satellite</th><th>Parameter</th>
                    <th>Severity</th><th>Status</th><th>Action</th>
                </tr>
            </thead>
            <tbody>
                ${state.alertHistory.map(a => {
                    const ts = formatDateTime(a.dispatched_at);
                    const acked = a.acknowledged_at
                        ? `<span class="ack-badge">ACK ${formatTime(a.acknowledged_at)}</span>`
                        : `<button class="ack-btn" onclick="acknowledgeAlert('${a.id}')">Acknowledge</button>`;
                    return `
                        <tr>
                            <td>${ts}</td>
                            <td>${a.satellite_id || '--'}</td>
                            <td class="chan-param">${a.parameter || '--'}</td>
                            <td><span class="timeline-severity ${a.severity}">${(a.severity || '').toUpperCase()}</span></td>
                            <td>${a.acknowledged_at ? '<span style="color:var(--nominal)">Acknowledged</span>' : '<span style="color:var(--text-muted)">Open</span>'}</td>
                            <td>${acked}</td>
                        </tr>`;
                }).join('')}
            </tbody>
        </table>`;
}

window.acknowledgeAlert = async function(id) {
    try {
        const resp = await fetch(`${API_BASE}/alerts/${id}/acknowledge`, { method: 'POST', headers: authHeaders() });
        if (!resp.ok) { toast(`Acknowledge failed: ${resp.status}`, 'error'); return; }
        toast('Alert acknowledged.', 'success');
        await loadAlertHistory();
    } catch (e) {
        toast('Failed to acknowledge: ' + e.message, 'error');
    }
};

async function loadAlertConfig() {
    try {
        const resp = await fetch(`${API_BASE}/alerts/config`, { headers: authHeaders() });
        if (!resp.ok) return; // not configured yet
        const cfg = await resp.json();
        if (cfg.webhook_url) document.getElementById('ac-webhook-url').value = cfg.webhook_url;
        if (cfg.email_to?.length)
            document.getElementById('ac-email-recipients').value = cfg.email_to.join(', ');
        if (cfg.smtp_host) document.getElementById('ac-smtp-host').value = cfg.smtp_host;
        if (cfg.smtp_port) document.getElementById('ac-smtp-port').value = cfg.smtp_port;
        if (cfg.min_severity) document.getElementById('ac-min-severity').value = cfg.min_severity;
        if (cfg.escalation_delay_s)
            document.getElementById('ac-escalation-delay').value = Math.round(cfg.escalation_delay_s / 60);
    } catch (e) { /* not critical */ }
}

async function saveAlertConfig() {
    const webhookUrl  = document.getElementById('ac-webhook-url').value.trim();
    const secret      = document.getElementById('ac-webhook-secret').value.trim();
    const emailRaw    = document.getElementById('ac-email-recipients').value.trim();
    const smtpHost    = document.getElementById('ac-smtp-host').value.trim();
    const smtpPort    = parseInt(document.getElementById('ac-smtp-port').value, 10);
    const minSev      = document.getElementById('ac-min-severity').value;
    const escalation  = parseInt(document.getElementById('ac-escalation-delay').value, 10);

    const body = { min_severity: minSev };
    if (webhookUrl) body.webhook_url = webhookUrl;
    if (secret)     body.webhook_secret = secret;
    if (emailRaw)   body.email_to = emailRaw.split(',').map(s => s.trim()).filter(Boolean);
    if (smtpHost)   body.smtp_host = smtpHost;
    if (!isNaN(smtpPort) && smtpPort > 0) body.smtp_port = smtpPort;
    if (!isNaN(escalation) && escalation > 0) body.escalation_delay_s = escalation * 60;

    try {
        const resp = await fetch(`${API_BASE}/alerts/config`, {
            method: 'PUT',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            toast(`Save failed: ${err.detail || resp.status}`, 'error');
            return;
        }
        toast('Alert delivery config saved.', 'success');
    } catch (e) {
        toast('Failed to save config: ' + e.message, 'error');
    }
}

// ============================================================
// Import Tab — Drop Zone Setup
// ============================================================
function setupDropZone(zoneId, inputId, fileKey, btnId, fileInfoId) {
    const zone  = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const btn   = document.getElementById(btnId);
    const info  = document.getElementById(fileInfoId);

    function setFile(file) {
        if (!file) return;
        state[fileKey] = file;
        zone.classList.add('has-file');
        info.style.display = 'flex';
        info.innerHTML = `📄 <strong>${file.name}</strong> &nbsp;<span style="color:var(--text-muted)">${formatBytes(file.size)}</span>`;
        btn.disabled = false;
    }

    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', () => setFile(input.files[0]));

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        setFile(e.dataTransfer.files[0]);
    });
}

// ============================================================
// Import Tab — XTCE Upload
// ============================================================
async function handleXtceUpload() {
    const satelliteId = document.getElementById('xtce-satellite-id').value.trim();
    const file = state.xtceFile;
    const btn = document.getElementById('xtceUploadBtn');
    const resultEl = document.getElementById('xtceResult');

    if (!satelliteId) { toast('Enter a Satellite ID first.', 'warning'); return; }
    if (!file) { toast('Select an XTCE XML file first.', 'warning'); return; }

    btn.disabled = true;
    resultEl.style.display = 'none';
    startProgress('xtceProgress', 'xtceProgressFill');

    try {
        const form = new FormData();
        form.append('satellite_id', satelliteId);
        form.append('file', file);

        const resp = await fetch(`${API_BASE}/parameters/import-xtce`, {
            method: 'POST',
            headers: authHeaders(),
            body: form,
        });

        stopProgress('xtceProgress', 'xtceProgressFill');

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            showImportResult('xtceResult', null, 'error', err.detail || `Error ${resp.status}`);
            toast('XTCE import failed.', 'error');
            return;
        }

        const data = await resp.json();
        showXtceResult(data);
        toast(`Imported ${data.parameters_imported} parameters.`, 'success');
        // Mark onboarding step 2 done
        document.getElementById('obStep2')?.classList.add('done');
    } catch (e) {
        stopProgress('xtceProgress', 'xtceProgressFill');
        showImportResult('xtceResult', null, 'error', 'Network error: ' + e.message);
        toast('Upload failed.', 'error');
    } finally {
        btn.disabled = false;
    }
}

function showXtceResult(data) {
    const el = document.getElementById('xtceResult');
    el.style.display = '';
    el.className = 'import-result result-success';

    // Count by subsystem
    const bySub = {};
    for (const p of (data.parameters || [])) {
        bySub[p.subsystem || 'unknown'] = (bySub[p.subsystem || 'unknown'] || 0) + 1;
    }
    const subRows = Object.entries(bySub)
        .sort((a, b) => b[1] - a[1])
        .map(([sub, n]) => `<div class="top-item" style="padding:4px 0">
            <span class="top-name">${sub.toUpperCase()}</span>
            <span class="top-count">${n}</span>
        </div>`).join('');

    el.innerHTML = `
        <div class="result-title">✓ XTCE Import Complete</div>
        <div class="result-stats">
            <div class="result-stat">
                <span class="result-stat-value">${data.parameters_imported}</span>
                <span class="result-stat-label">Parameters</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${Object.keys(bySub).length}</span>
                <span class="result-stat-label">Subsystems</span>
            </div>
        </div>
        ${subRows ? `<div style="margin-top:12px;font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">By Subsystem</div>${subRows}` : ''}`;
}

// ============================================================
// Import Tab — CSV Upload
// ============================================================
async function handleCsvUpload() {
    const satelliteId = document.getElementById('csv-satellite-id').value.trim();
    const resample    = document.getElementById('csv-resample').value.trim() || '1';
    const file = state.csvFile;
    const btn  = document.getElementById('csvUploadBtn');
    const resultEl = document.getElementById('csvResult');

    if (!satelliteId) { toast('Enter a Satellite ID first.', 'warning'); return; }
    if (!file) { toast('Select a CSV file first.', 'warning'); return; }

    btn.disabled = true;
    document.getElementById('csvAnalyzeBtn').style.display = 'none';
    resultEl.style.display = 'none';
    document.getElementById('csvProgressText').textContent = 'Uploading data...';
    startProgress('csvProgress', 'csvProgressFill');

    try {
        const form = new FormData();
        form.append('satellite_id', satelliteId);
        form.append('resample_minutes', resample);
        form.append('file', file);

        const resp = await fetch(`${API_BASE}/telemetry/upload`, {
            method: 'POST',
            headers: authHeaders(),
            body: form,
        });

        stopProgress('csvProgress', 'csvProgressFill');

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            showImportResult('csvResult', null, 'error', err.detail || `Error ${resp.status}`);
            toast('CSV upload failed.', 'error');
            return;
        }

        const data = await resp.json();
        // Store satellite id for the subsequent analyze step
        state.lastUploadedSatelliteId = satelliteId;
        showCsvResult(data);
        toast(`Upload complete: ${(data.total_rows_inserted || 0).toLocaleString()} rows written across ${data.channels_loaded || 0} channels. Click "Run Analysis" to detect anomalies.`, 'success');
        // Show the analyze button
        document.getElementById('csvAnalyzeBtn').style.display = '';
    } catch (e) {
        stopProgress('csvProgress', 'csvProgressFill');
        showImportResult('csvResult', null, 'error', 'Network error: ' + e.message);
        toast('Upload failed.', 'error');
    } finally {
        btn.disabled = false;
    }
}

async function handleRunAnalysis() {
    const satelliteId = state.lastUploadedSatelliteId
        || document.getElementById('csv-satellite-id').value.trim();
    if (!satelliteId) { toast('Enter a Satellite ID first.', 'warning'); return; }

    const analyzeBtn = document.getElementById('csvAnalyzeBtn');
    analyzeBtn.disabled = true;
    document.getElementById('csvProgressText').textContent = 'Running anomaly detection...';
    startProgress('csvProgress', 'csvProgressFill');

    try {
        const resp = await fetch(`${API_BASE}/telemetry/${encodeURIComponent(satelliteId)}/analyze`, {
            method: 'POST',
            headers: authHeaders(),
        });

        stopProgress('csvProgress', 'csvProgressFill');

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            showImportResult('csvResult', null, 'error', err.detail || `Error ${resp.status}`);
            toast('Analysis failed.', 'error');
            return;
        }

        const data = await resp.json();
        showAnalyzeResult(data);
        toast(`Analysis complete: ${data.total_anomalies} anomalies found across ${data.channels_analyzed} channels in ${data.elapsed_s}s.`, 'success');
        fetchAnomalies(); // refresh monitor tab
    } catch (e) {
        stopProgress('csvProgress', 'csvProgressFill');
        showImportResult('csvResult', null, 'error', 'Network error: ' + e.message);
        toast('Analysis failed.', 'error');
    } finally {
        analyzeBtn.disabled = false;
    }
}

function showCsvResult(data) {
    const el = document.getElementById('csvResult');
    el.style.display = '';
    el.className = 'import-result result-success';
    el.innerHTML = `
        <div class="result-title">✓ Upload Complete</div>
        <div class="result-stats">
            <div class="result-stat">
                <span class="result-stat-value">${(data.total_rows_inserted || 0).toLocaleString()}</span>
                <span class="result-stat-label">Rows Written</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${(data.channels_loaded || 0)}</span>
                <span class="result-stat-label">Channels</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${(data.channels_skipped || 0)}</span>
                <span class="result-stat-label">Skipped</span>
            </div>
        </div>
        <div style="margin-top:10px;font-size:11px;color:var(--text-secondary)">
            Data loaded — click <strong>Run Analysis</strong> to detect anomalies.
        </div>`;
}

function showAnalyzeResult(data) {
    const el = document.getElementById('csvResult');
    el.style.display = '';
    el.className = 'import-result result-success';
    const topChannels = Object.entries(data.anomalies_per_channel || {})
        .filter(([, n]) => n > 0)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5)
        .map(([ch, n]) => `<span style="opacity:0.8">${ch}</span>: <strong>${n}</strong>`)
        .join(' &nbsp;·&nbsp; ');
    el.innerHTML = `
        <div class="result-title">✓ Analysis Complete</div>
        <div class="result-stats">
            <div class="result-stat">
                <span class="result-stat-value">${data.total_anomalies}</span>
                <span class="result-stat-label">Anomalies</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${data.channels_analyzed}</span>
                <span class="result-stat-label">Channels</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${data.elapsed_s}s</span>
                <span class="result-stat-label">Time</span>
            </div>
        </div>
        ${topChannels ? `<div style="margin-top:10px;font-size:11px;color:var(--text-secondary)">${topChannels}</div>` : ''}
        <div style="margin-top:8px;font-size:11px;color:var(--text-secondary)">
            Refresh the Monitor tab to view all anomalies.
        </div>`;
}

function showImportResult(elId, _data, type, message) {
    const el = document.getElementById(elId);
    el.style.display = '';
    el.className = `import-result result-${type}`;
    el.innerHTML = `<div class="result-title">${type === 'error' ? '✗ ' : '✓ '}${message}</div>`;
}

// ============================================================
// Import Tab — Live Integrations (YAMCS / InfluxDB / REST API)
// ============================================================

function switchConnectorTab(tab) {
    ['api', 'yamcs', 'influx'].forEach(t => {
        document.getElementById(`connectorPanel${t.charAt(0).toUpperCase() + t.slice(1)}`)
            .style.display = t === tab ? '' : 'none';
        document.getElementById(`ctab${t.charAt(0).toUpperCase() + t.slice(1)}`)
            .classList.toggle('active', t === tab);
    });
    if (tab === 'api') _populateApiPushInfo();
}

function _populateApiPushInfo() {
    const endpoint = `${API_BASE}/telemetry`;
    const el = document.getElementById('apiPushEndpoint');
    if (el) el.value = endpoint;
    const curlEl = document.getElementById('apiPushCurl');
    if (curlEl) {
        curlEl.textContent =
`curl -X POST ${endpoint} \\
  -H "Authorization: Bearer <your-api-key>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "points": [{
      "satellite_id": "DOVE-001",
      "timestamp": "${new Date().toISOString()}",
      "subsystem": "eps",
      "parameter": "battery_voltage",
      "value": 12.4,
      "unit": "V"
    }]
  }'`;
    }
}

function copyToClipboard(inputId) {
    const el = document.getElementById(inputId);
    if (!el) return;
    navigator.clipboard?.writeText(el.value || el.textContent)
        .then(() => toast('Copied to clipboard.', 'success'))
        .catch(() => toast('Copy failed — select and copy manually.', 'warning'));
}

async function handleYamcsConnect() {
    const url    = document.getElementById('yamcs-url').value.trim();
    const inst   = document.getElementById('yamcs-instance').value.trim();
    const satId  = document.getElementById('yamcs-satellite-id').value.trim();
    const sub    = document.getElementById('yamcs-subsystem').value.trim() || 'yamcs';
    const start  = document.getElementById('yamcs-start').value.trim() || null;
    const stop   = document.getElementById('yamcs-stop').value.trim() || null;
    const apiKey = document.getElementById('yamcs-apikey').value.trim();
    const params = document.getElementById('yamcs-parameters').value
        .split('\n').map(s => s.trim()).filter(Boolean);

    if (!url)    { toast('Enter the YAMCS server URL.', 'warning'); return; }
    if (!inst)   { toast('Enter the YAMCS instance name.', 'warning'); return; }
    if (!satId)  { toast('Enter a Satellite ID.', 'warning'); return; }
    if (!params.length) { toast('Enter at least one parameter path.', 'warning'); return; }

    const btn = document.getElementById('yamcsConnectBtn');
    btn.disabled = true;
    startProgress('yamcsProgress', 'yamcsProgressFill');
    document.getElementById('yamcsResult').style.display = 'none';

    try {
        const body = { satellite_id: satId, yamcs_url: url, instance: inst,
                       parameters: params, subsystem: sub, api_key: apiKey };
        if (start) body.start = start;
        if (stop)  body.stop  = stop;

        const resp = await fetch(`${API_BASE}/connectors/yamcs`, {
            method: 'POST',
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        stopProgress('yamcsProgress', 'yamcsProgressFill');
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            showImportResult('yamcsResult', null, 'error', err.detail || `Error ${resp.status}`);
            toast('YAMCS fetch failed.', 'error');
            return;
        }
        const data = await resp.json();
        showConnectorResult('yamcsResult', data);
        toast(`YAMCS: ${data.total_rows_inserted.toLocaleString()} rows, ${data.total_anomalies} anomalies in ${data.elapsed_s}s.`, 'success');
        fetchAnomalies();
    } catch (e) {
        stopProgress('yamcsProgress', 'yamcsProgressFill');
        showImportResult('yamcsResult', null, 'error', 'Network error: ' + e.message);
        toast('YAMCS fetch failed.', 'error');
    } finally {
        btn.disabled = false;
    }
}

async function handleInfluxConnect() {
    const url    = document.getElementById('influx-url').value.trim();
    const org    = document.getElementById('influx-org').value.trim();
    const bucket = document.getElementById('influx-bucket').value.trim();
    const token  = document.getElementById('influx-token').value.trim();
    const meas   = document.getElementById('influx-measurement').value.trim();
    const satId  = document.getElementById('influx-satellite-id').value.trim();
    const sub    = document.getElementById('influx-subsystem').value.trim() || 'influxdb';
    const start  = document.getElementById('influx-start').value.trim() || '-30d';
    const stop   = document.getElementById('influx-stop').value.trim() || 'now()';
    const fields = document.getElementById('influx-fields').value
        .split('\n').map(s => s.trim()).filter(Boolean);

    if (!url)    { toast('Enter the InfluxDB URL.', 'warning'); return; }
    if (!org)    { toast('Enter the organization.', 'warning'); return; }
    if (!bucket) { toast('Enter the bucket name.', 'warning'); return; }
    if (!token)  { toast('Enter the API token.', 'warning'); return; }
    if (!meas)   { toast('Enter the measurement name.', 'warning'); return; }
    if (!satId)  { toast('Enter a Satellite ID.', 'warning'); return; }
    if (!fields.length) { toast('Enter at least one field name.', 'warning'); return; }

    const btn = document.getElementById('influxConnectBtn');
    btn.disabled = true;
    startProgress('influxProgress', 'influxProgressFill');
    document.getElementById('influxResult').style.display = 'none';

    try {
        const resp = await fetch(`${API_BASE}/connectors/influxdb`, {
            method: 'POST',
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
            body: JSON.stringify({
                satellite_id: satId, influxdb_url: url, org, bucket, token,
                measurement: meas, fields, subsystem: sub, start, stop,
            }),
        });
        stopProgress('influxProgress', 'influxProgressFill');
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            showImportResult('influxResult', null, 'error', err.detail || `Error ${resp.status}`);
            toast('InfluxDB fetch failed.', 'error');
            return;
        }
        const data = await resp.json();
        showConnectorResult('influxResult', data);
        toast(`InfluxDB: ${data.total_rows_inserted.toLocaleString()} rows, ${data.total_anomalies} anomalies in ${data.elapsed_s}s.`, 'success');
        fetchAnomalies();
    } catch (e) {
        stopProgress('influxProgress', 'influxProgressFill');
        showImportResult('influxResult', null, 'error', 'Network error: ' + e.message);
        toast('InfluxDB fetch failed.', 'error');
    } finally {
        btn.disabled = false;
    }
}

function showConnectorResult(elId, data) {
    const el = document.getElementById(elId);
    el.style.display = '';
    el.className = 'import-result result-success';
    el.innerHTML = `
        <div class="result-title">✓ Fetch &amp; Analysis Complete</div>
        <div class="result-stats">
            <div class="result-stat">
                <span class="result-stat-value">${(data.total_rows_inserted || 0).toLocaleString()}</span>
                <span class="result-stat-label">Rows</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${data.channels_loaded || 0}</span>
                <span class="result-stat-label">Channels</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${data.total_anomalies || 0}</span>
                <span class="result-stat-label">Anomalies</span>
            </div>
            <div class="result-stat">
                <span class="result-stat-value">${data.elapsed_s || 0}s</span>
                <span class="result-stat-label">Time</span>
            </div>
        </div>
        <div style="margin-top:8px;font-size:11px;color:var(--text-secondary)">
            Source: <code>${data.source || ''}</code> · Refresh Monitor tab to view anomalies.
        </div>`;
}

// ============================================================
// Import Tab — Progress Bar
// ============================================================
function startProgress(progressId, fillId) {
    const prog = document.getElementById(progressId);
    const fill = document.getElementById(fillId);
    prog.style.display = '';
    fill.classList.add('running');
    fill.style.width = '0%';
}

function stopProgress(progressId, fillId) {
    const fill = document.getElementById(fillId);
    fill.classList.remove('running');
    fill.style.width = '100%';
    setTimeout(() => { document.getElementById(progressId).style.display = 'none'; }, 600);
}

// ============================================================
// Export Functions
// ============================================================
function exportCsv() {
    const data = state.satFilter
        ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
        : state.anomalies;

    if (data.length === 0) { toast('No anomalies to export.', 'info'); return; }

    const headers = ['timestamp', 'satellite_id', 'parameter', 'value', 'severity', 'confidence', 'detectors', 'explanation'];
    const rows = data.map(a => [
        a.timestamp || '',
        a.satellite_id || '',
        a.parameter || '',
        a.value != null ? a.value : '',
        a.severity || '',
        a.confidence != null ? a.confidence.toFixed(4) : '',
        (a.detectors_triggered || []).join(';'),
        (a.explanation || '').replace(/,/g, ';'),
    ]);

    const csv = [headers, ...rows].map(r => r.map(v => `"${v}"`).join(',')).join('\n');
    downloadFile('sentinel_anomalies.csv', csv, 'text/csv');
    toast(`Exported ${data.length} anomalies as CSV.`, 'success');
}

function exportJson() {
    const data = state.satFilter
        ? state.anomalies.filter(a => a.satellite_id === state.satFilter)
        : state.anomalies;

    if (data.length === 0) { toast('No anomalies to export.', 'info'); return; }
    downloadFile('sentinel_anomalies.json', JSON.stringify(data, null, 2), 'application/json');
    toast(`Exported ${data.length} anomalies as JSON.`, 'success');
}

window.exportAnomaly = function() {
    const a = state.selectedAnomaly;
    if (!a) return;
    const name = `sentinel_${(a.satellite_id || 'unknown')}_${(a.parameter || 'param')}_${Date.now()}.json`;
    downloadFile(name, JSON.stringify(a, null, 2), 'application/json');
    toast('Anomaly exported.', 'success');
};

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
}

// ============================================================
// Satellite Filter
// ============================================================
function populateSatFilters() {
    const sats = state.satellites;

    // All satellite dropdowns (topbar + tabs) are now <select> elements
    ['globalSatFilter', 'chanSatFilter', 'alertSatFilter', 'anomSatFilter', 'incSatFilter'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const current = el.value;
        while (el.options.length > 1) el.remove(1);
        sats.forEach(sat => {
            const opt = document.createElement('option');
            opt.value = sat; opt.textContent = sat;
            el.appendChild(opt);
        });
        if (current) el.value = current;
    });
}

function rebuildSubsystemFilter() {
    const sel = document.getElementById('subsystemFilter');
    const current = sel.value;
    const subs = new Set();
    for (const a of state.anomalies) {
        const sub = a.subsystem || guessSubsystem(a.parameter);
        if (sub) subs.add(sub);
    }
    while (sel.options.length > 1) sel.remove(1);
    subs.forEach(sub => {
        const opt = document.createElement('option');
        opt.value = sub;
        opt.textContent = (SUBSYSTEM_META[sub]?.label || sub.toUpperCase());
        sel.appendChild(opt);
    });
    if (current) sel.value = current;
}

// ============================================================
// API Polling
// ============================================================
async function fetchHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        document.getElementById('uptime').textContent = formatDuration(data.uptime_seconds);
    } catch (e) { /* ignore */ }
}

// ── Pagination state ────────────────────────────────────────
// newestTs  : timestamp of the most-recently loaded anomaly (for poll)
// oldestTs  : timestamp of the oldest loaded anomaly (for infinite scroll cursor)
// allLoaded : true when the last page returned < PAGE_SIZE rows
// loading   : guard flag to prevent concurrent scroll fetches
const _PAGE_SIZE = 200;
let _paging = { newestTs: null, oldestTs: null, allLoaded: false, loading: false };

function _resetPaging() {
    state.anomalies = [];
    _paging = { newestTs: null, oldestTs: null, allLoaded: false, loading: false };
}

function _mergeAnomalies(incoming) {
    // Returns list of genuinely new items
    const existing = new Map(state.anomalies.map(a => [a.id, a]));
    const newOnes = incoming.filter(a => !existing.has(a.id));
    for (const a of newOnes) {
        state.anomalies.push(a);
        updateSubsystemStatus(a);
    }
    if (newOnes.length > 0) {
        state.anomalies.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        // Update cursors
        _paging.newestTs = state.anomalies[0]?.timestamp ?? null;
        _paging.oldestTs = state.anomalies[state.anomalies.length - 1]?.timestamp ?? null;
    }
    return newOnes;
}

// Initial full load — called on login / tenant switch / date filter change
async function fetchAnomalies() {
    _resetPaging();
    _paging.loading = true;
    try {
        const params = _buildFetchParams({});
        const resp = await fetch(`${API_BASE}/anomalies?${params}`, { headers: authHeaders() });
        if (!resp.ok) return;
        const data = await resp.json();
        if (!Array.isArray(data)) return;
        _mergeAnomalies(data);
        _paging.allLoaded = data.length < _PAGE_SIZE;
        _refreshUI();
    } catch (e) { /* ignore */ } finally {
        _paging.loading = false;
        _updateScrollSentinel();
    }
}

// Poll for NEW anomalies (called by interval) — does NOT reset the list
async function pollNewAnomalies() {
    if (!_paging.newestTs) { await fetchAnomalies(); return; }
    // If date filters are active, skip live polling (historical view)
    const df = document.getElementById('dateFrom')?.value;
    const dt = document.getElementById('dateTo')?.value;
    if (df || dt) return;
    try {
        const params = _buildFetchParams({ since: _paging.newestTs });
        const resp = await fetch(`${API_BASE}/anomalies?${params}`, { headers: authHeaders() });
        if (!resp.ok) return;
        const data = await resp.json();
        if (!Array.isArray(data) || data.length === 0) return;
        const newOnes = _mergeAnomalies(data);
        if (newOnes.length > 0 && newOnes.length <= 10) newOnes.forEach(a => addAlert(a));
        if (newOnes.length > 0) _refreshUI();
    } catch (e) { /* ignore */ }
}

// Infinite scroll — load next older page
async function loadOlderAnomalies() {
    if (_paging.loading || _paging.allLoaded || !_paging.oldestTs) return;
    _paging.loading = true;
    document.getElementById('timelineLoadingSpinner').style.display = 'block';
    try {
        const params = _buildFetchParams({ before: _paging.oldestTs });
        const resp = await fetch(`${API_BASE}/anomalies?${params}`, { headers: authHeaders() });
        if (!resp.ok) return;
        const data = await resp.json();
        if (!Array.isArray(data)) return;
        _mergeAnomalies(data);
        _paging.allLoaded = data.length < _PAGE_SIZE;
        _refreshUI();
    } catch (e) { /* ignore */ } finally {
        _paging.loading = false;
        document.getElementById('timelineLoadingSpinner').style.display = 'none';
        _updateScrollSentinel();
    }
}

function _buildFetchParams(extra = {}) {
    const p = new URLSearchParams({ limit: _PAGE_SIZE });
    const df = document.getElementById('dateFrom')?.value;
    const dt = document.getElementById('dateTo')?.value;
    if (df) p.set('date_from', new Date(df).toISOString());
    if (dt) { const d = new Date(dt); d.setHours(23,59,59,999); p.set('date_to', d.toISOString()); }
    if (extra.since)  p.set('since',  new Date(extra.since).toISOString());
    if (extra.before) p.set('before', new Date(extra.before).toISOString());
    return p.toString();
}

function _refreshUI() {
    renderTimeline();
    renderMetrics();
    renderSeverityBar();
    rebuildSubsystemFilter();
    maybeUpdateAnalysis();
    _updateLoadedInfo();
}

function _updateLoadedInfo() {
    const el = document.getElementById('timelineLoadedInfo');
    if (!el) return;
    // Derive total from severity counts (same source as the severity bar) — stays
    // consistent even when state.totalAnomalies is stale from a prior tenant context.
    const sc = state.severityCounts || {};
    const scTotal = (sc.critical || 0) + (sc.warning || 0) + (sc.watch || 0);
    const total = scTotal > 0 ? scTotal : state.totalAnomalies;
    const loaded = state.anomalies.length;
    if (total > loaded) {
        el.textContent = `Showing ${loaded.toLocaleString()} of ${total.toLocaleString()} anomalies — scroll down to load more`;
    } else if (loaded > 0) {
        el.textContent = `All ${loaded.toLocaleString()} anomalies loaded`;
    } else {
        el.textContent = '';
    }
}

function _updateScrollSentinel() {
    const sentinel = document.getElementById('timelineScrollSentinel');
    if (!sentinel) return;
    if (_paging.allLoaded) {
        sentinel.style.display = 'none';
    } else {
        sentinel.style.display = 'block';
    }
}

// Wire IntersectionObserver to the sentinel div
(function initInfiniteScroll() {
    const sentinel = document.getElementById('timelineScrollSentinel');
    if (!sentinel) return;
    const observer = new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) loadOlderAnomalies();
    }, { rootMargin: '200px' });
    observer.observe(sentinel);
})();

// Date filter helpers
function clearDateFilter() {
    const df = document.getElementById('dateFrom');
    const dt = document.getElementById('dateTo');
    if (df) df.value = '';
    if (dt) dt.value = '';
    fetchAnomalies();
}
document.getElementById('dateFrom')?.addEventListener('change', () => fetchAnomalies());
document.getElementById('dateTo')?.addEventListener('change', () => fetchAnomalies());

async function fetchSatellites() {
    try {
        const resp = await fetch(`${API_BASE}/satellites`, { headers: authHeaders() });
        if (!resp.ok) return;
        state.satellites = await resp.json();
        populateSatFilters();
        renderMetrics();
    } catch (e) { /* ignore */ }
}

async function fetchStats() {
    try {
        const resp = await fetch(`${API_BASE}/stats`, { headers: authHeaders() });
        if (!resp.ok) return;
        const data = await resp.json();
        state.totalPoints          = data.total_telemetry_points ?? 0;
        state.pointsLastHour       = data.points_last_hour ?? 0;
        state.totalAnomalies       = data.total_anomalies ?? 0;
        state.severityCounts       = data.anomaly_severity_counts ?? {};
        renderMetrics();
        renderSeverityBar();
    } catch (e) { /* ignore */ }
}

// ============================================================
// Utilities
// ============================================================
function formatTime(ts) {
    if (!ts) return '--:--';
    return new Date(ts).toLocaleTimeString('en-GB', {
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
}

function formatDateTime(ts) {
    if (!ts) return '--';
    return new Date(ts).toLocaleString('en-GB', {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
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
    if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
}

function formatBytes(n) {
    if (n < 1024)       return `${n} B`;
    if (n < 1048576)    return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1048576).toFixed(1)} MB`;
}

// ============================================================
// Auth Helpers
// ============================================================
function authHeaders(extra = {}) {
    const h = { ...extra };
    if (state.auth.accessToken) h['Authorization'] = `Bearer ${state.auth.accessToken}`;
    // Sentinel users scope all requests to a selected tenant via X-Tenant-ID header
    if (state.auth.user?.scope === 'sentinel' && state.auth.tenantContext) {
        h['X-Tenant-ID'] = state.auth.tenantContext;
    }
    return h;
}

// ============================================================
// Theme Management
// ============================================================
function initTheme() {
    const stored = localStorage.getItem('sentinel-theme') || 'system';
    applyTheme(stored);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if ((localStorage.getItem('sentinel-theme') || 'system') === 'system') applyTheme('system');
    });
}

function applyTheme(theme) {
    localStorage.setItem('sentinel-theme', theme);
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = theme === 'dark' || (theme === 'system' && prefersDark);
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    document.querySelectorAll('.theme-option').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.theme === theme);
    });
}

// ============================================================
// Auth — User Chip Population
// ============================================================
function populateUserChip(user) {
    const avatar      = document.getElementById('userAvatar');
    const emailEl     = document.getElementById('userEmail');
    const badgeEl     = document.getElementById('userRoleBadge');
    const ddName      = document.getElementById('ddName');
    const ddEmail     = document.getElementById('ddEmail');
    const ddTenant    = document.getElementById('ddTenant');
    const ddChangePw  = document.getElementById('ddChangePassword');
    const ddLogoutBtn = document.getElementById('ddLogout');

    if (user) {
        // Show display_name if set, otherwise derive a short name from email
        const displayName = user.display_name
            || (user.email ? user.email.split('@')[0].replace(/[._-]/g, ' ') : null)
            || user.role
            || '?';
        const initial = displayName[0].toUpperCase();
        avatar.textContent  = initial;
        emailEl.textContent = displayName;
        badgeEl.textContent = (user.role || '').toUpperCase();
        badgeEl.className   = `user-role-badge role-${(user.role || '').replaceAll('_', '-')}`;
        ddName.textContent  = displayName;
        ddEmail.textContent = user.email || (user.tenant_id ? `Tenant: ${user.tenant_id}` : 'Sentinel Staff');
        // Show tenant ID so customers always know what to type at login
        if (ddTenant) {
            ddTenant.textContent = user.tenant_id
                ? `TENANT ID: ${user.tenant_id}`
                : user.scope === 'sentinel' ? 'SENTINEL STAFF' : '';
        }
        // Show change-password only for JWT tenant users (scope≠sentinel)
        ddChangePw.style.display  = (state.auth.mode === 'jwt' && user.tenant_id) ? '' : 'none';
        ddLogoutBtn.innerHTML = '<span class="user-dropdown-icon">↩</span> Logout';
        ddLogoutBtn.classList.add('danger');
    } else {
        avatar.textContent  = '?';
        emailEl.textContent = 'Not signed in';
        badgeEl.textContent = '';
        badgeEl.className   = 'user-role-badge';
        ddName.textContent  = 'Not signed in';
        ddEmail.textContent = 'Authentication required';
        if (ddTenant) ddTenant.textContent = '';
        ddChangePw.style.display = 'none';
        ddLogoutBtn.innerHTML = '<span class="user-dropdown-icon">→</span> Sign In';
        ddLogoutBtn.classList.remove('danger');
    }
}

// ============================================================
// Auth — Init (restore session from localStorage)
// ============================================================
async function initAuth() {
    const stored        = localStorage.getItem('sentinel-access-token');
    const storedRefresh = localStorage.getItem('sentinel-refresh-token');

    if (stored) {
        state.auth.accessToken  = stored;
        state.auth.refreshToken = storedRefresh;
        try {
            const resp = await fetch(`${API_BASE}/auth/me`, {
                headers: { 'Authorization': `Bearer ${stored}` },
            });
            if (resp.ok) {
                const user = await resp.json();
                state.auth.user = user;
                state.auth.mode = 'jwt';
                // Restore sentinel tenant context from last session
                if (user.scope === 'sentinel') {
                    const saved = localStorage.getItem('sentinel-tenant-ctx');
                    if (saved) state.auth.tenantContext = saved;
                }
                populateUserChip(user);
                applyRoleGating(user.role, user.scope);
                return;
            }
        } catch (e) { /* network error — fall through */ }
        // Token expired or invalid — clear
        localStorage.removeItem('sentinel-access-token');
        localStorage.removeItem('sentinel-refresh-token');
        state.auth.accessToken  = null;
        state.auth.refreshToken = null;
    }
    // No valid session — require login
    populateUserChip(null);
    applyRoleGating('viewer', '');
    showLoginModal();
}

// ============================================================
// Auth — Login
// ============================================================
async function handleLogin() {
    const email      = document.getElementById('loginEmail').value.trim();
    const password   = document.getElementById('loginPassword').value;
    const tenantId   = (document.getElementById('loginTenantId')?.value || '').trim();
    const errorEl    = document.getElementById('loginError');
    const btn        = document.getElementById('loginBtn');

    if (!email || !password) { errorEl.textContent = 'Email and password are required.'; return; }

    btn.disabled    = true;
    btn.textContent = 'Signing in…';
    errorEl.textContent = '';

    try {
        // Strategy:
        //   1. If tenant_id supplied → try tenant login with that tenant only.
        //   2. If no tenant_id → try sentinel-login first, then 'default' tenant.
        //      This way Sentinel staff (most common single-box deployment) land first.
        let resp;
        if (tenantId) {
            // Explicit tenant supplied — single attempt
            resp = await fetch(`${API_BASE}/auth/login`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ email, password, tenant_id: tenantId }),
            });
        } else {
            // No tenant specified — try sentinel login first, then 'default' tenant
            resp = await fetch(`${API_BASE}/auth/sentinel-login`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ email, password }),
            });
            if (resp.status === 401) {
                resp = await fetch(`${API_BASE}/auth/login`, {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify({ email, password, tenant_id: 'default' }),
                });
            }
        }

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            errorEl.textContent = err.detail || 'Invalid email or password.';
            return;
        }

        const data = await resp.json();
        state.auth.accessToken  = data.access_token;
        state.auth.refreshToken = data.refresh_token;
        state.auth.mode         = 'jwt';
        localStorage.setItem('sentinel-access-token',  data.access_token);
        localStorage.setItem('sentinel-refresh-token', data.refresh_token);

        // Fetch user profile
        const meResp = await fetch(`${API_BASE}/auth/me`, {
            headers: { 'Authorization': `Bearer ${data.access_token}` },
        });
        if (meResp.ok) {
            state.auth.user = await meResp.json();
            populateUserChip(state.auth.user);
        }

        document.getElementById('loginModal').style.display = 'none';
        toast('Signed in successfully.', 'success');
        applyRoleGating(state.auth.user?.role || 'viewer', state.auth.user?.scope || '');
        fetchAnomalies();
        fetchSatellites();
        fetchStats();
    } catch (e) {
        errorEl.textContent = 'Network error. Please try again.';
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Sign In';
    }
}

function showLoginModal() {
    closeUserDropdown();
    document.getElementById('loginEmail').value    = '';
    document.getElementById('loginPassword').value = '';
    document.getElementById('loginPassword').type  = 'password';  // reset show/hide state
    const tidEl = document.getElementById('loginTenantId');
    if (tidEl) tidEl.value = '';
    document.getElementById('iconEyeOpen').style.display = '';
    document.getElementById('iconEyeOff').style.display  = 'none';
    document.getElementById('loginPwToggle').style.opacity = '0.6';
    document.getElementById('loginError').textContent = '';
    document.getElementById('loginModal').style.display = 'flex';
    setTimeout(() => document.getElementById('loginEmail').focus(), 50);
}

// ============================================================
// Auth — Logout
// ============================================================
async function handleLogout() {
    closeUserDropdown();

    try {
        if (state.auth.accessToken && state.auth.refreshToken) {
            await fetch(`${API_BASE}/auth/logout`, {
                method:  'POST',
                headers: {
                    'Authorization': `Bearer ${state.auth.accessToken}`,
                    'Content-Type':  'application/json',
                },
                body: JSON.stringify({ refresh_token: state.auth.refreshToken }),
            });
        }
    } catch (e) { /* clear locally regardless */ }

    state.auth.accessToken   = null;
    state.auth.refreshToken  = null;
    state.auth.user          = null;
    state.auth.mode          = null;
    state.auth.tenantContext = null;
    localStorage.removeItem('sentinel-access-token');
    localStorage.removeItem('sentinel-refresh-token');
    localStorage.removeItem('sentinel-tenant-ctx');
    const sw = document.getElementById('sentinelTenantSwitch');
    if (sw) sw.style.display = 'none';
    populateUserChip(null);
    applyRoleGating('viewer', '');
    toast('Signed out.', 'info');
    showLoginModal();
}

// ============================================================
// Auth — Change Password
// ============================================================
function openChangePassword() {
    closeUserDropdown();
    document.getElementById('cpCurrentPw').value   = '';
    document.getElementById('cpNewPw').value       = '';
    document.getElementById('cpConfirmPw').value   = '';
    document.getElementById('cpError').textContent = '';
    document.getElementById('changePwOverlay').classList.add('open');
    setTimeout(() => document.getElementById('cpCurrentPw').focus(), 50);
}

function closeChangePassword() {
    document.getElementById('changePwOverlay').classList.remove('open');
}

async function handleChangePassword() {
    const current  = document.getElementById('cpCurrentPw').value;
    const newPw    = document.getElementById('cpNewPw').value;
    const confirm  = document.getElementById('cpConfirmPw').value;
    const errorEl  = document.getElementById('cpError');
    const btn      = document.getElementById('cpSaveBtn');

    if (!current || !newPw || !confirm) { errorEl.textContent = 'All fields are required.'; return; }
    if (newPw !== confirm) { errorEl.textContent = 'New passwords do not match.'; return; }
    if (newPw.length < 8)  { errorEl.textContent = 'New password must be at least 8 characters.'; return; }

    btn.disabled    = true;
    btn.textContent = 'Changing…';
    errorEl.textContent = '';

    try {
        const resp = await fetch(`${API_BASE}/auth/change-password`, {
            method:  'POST',
            headers: {
                'Authorization': `Bearer ${state.auth.accessToken}`,
                'Content-Type':  'application/json',
            },
            body: JSON.stringify({ current_password: current, new_password: newPw }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            errorEl.textContent = err.detail || `Error ${resp.status}`;
            return;
        }
        closeChangePassword();
        toast('Password changed. Please sign in again.', 'success');
        // Server revokes all tokens — force re-login
        state.auth.accessToken  = null;
        state.auth.refreshToken = null;
        state.auth.user         = null;
        state.auth.mode         = null;
        localStorage.removeItem('sentinel-access-token');
        localStorage.removeItem('sentinel-refresh-token');
        populateUserChip(null);
        showLoginModal();
    } catch (e) {
        errorEl.textContent = 'Network error: ' + e.message;
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Change Password';
    }
}

// ============================================================
// Settings Modal
// ============================================================
function openSettings() {
    closeUserDropdown();
    document.getElementById('settingsApiBase').value  = API_BASE;
    document.getElementById('settingsAuthMode').value =
        state.auth.mode === 'jwt' ? 'JWT Bearer Token' : 'Demo / API Key';
    const theme = localStorage.getItem('sentinel-theme') || 'system';
    document.querySelectorAll('.theme-option').forEach(b => {
        b.classList.toggle('active', b.dataset.theme === theme);
    });
    const stored = localStorage.getItem('sentinel-poll-interval') || '10000';
    document.getElementById('pollInterval').value = stored;
    document.getElementById('settingsOverlay').classList.add('open');
}

function closeSettings() {
    document.getElementById('settingsOverlay').classList.remove('open');
}

function saveSettings() {
    const newInterval = parseInt(document.getElementById('pollInterval').value, 10);
    if (!isNaN(newInterval) && newInterval > 0) {
        localStorage.setItem('sentinel-poll-interval', String(newInterval));
        clearInterval(state.pollIntervals.anomaly);
        state.pollIntervals.anomaly = setInterval(pollNewAnomalies, newInterval);
    }
    closeSettings();
    toast('Settings saved.', 'success');
}

// ============================================================
// User Chip — Dropdown Toggle
// ============================================================
function closeUserDropdown() {
    document.getElementById('userDropdown').classList.remove('open');
}

// ============================================================
// Event Listeners
// ============================================================
document.getElementById('severityFilter').addEventListener('change', renderTimeline);
document.getElementById('subsystemFilter').addEventListener('change', renderTimeline);
// readonly on load blocks browser autofill; removed on first user focus
(function() {
    const s = document.getElementById('anomalySearch');
    s.addEventListener('focus', function() { s.removeAttribute('readonly'); }, { once: true });
    s.addEventListener('input', renderTimeline);
})();

// Satellite dropdown filter — <select> element
document.getElementById('globalSatFilter').addEventListener('change', e => {
    state.satFilter = e.target.value;
    renderTimeline();
    renderMetrics();
    renderSeverityBar();
    maybeUpdateAnalysis();
});

document.getElementById('clearAlerts').addEventListener('click', () => {
    document.getElementById('alertList').innerHTML = '<div class="empty-state">No alerts dispatched</div>';
});

document.getElementById('exportCsvBtn').addEventListener('click', exportCsv);
document.getElementById('exportJsonBtn').addEventListener('click', exportJson);

document.getElementById('refreshChannelsBtn').addEventListener('click', loadChannels);
document.getElementById('chanSatFilter').addEventListener('change', loadChannels);
document.getElementById('editorClose').addEventListener('click', () => {
    state.selectedChannel = null;
    document.getElementById('thresholdEditor').style.display = 'none';
    renderChannelTable();
});

document.getElementById('refreshIncidentsBtn').addEventListener('click', loadIncidents);
document.getElementById('incSatFilter').addEventListener('change', loadIncidents);
document.getElementById('incStatusFilter').addEventListener('change', loadIncidents);
document.getElementById('refreshAnomsBtn').addEventListener('click', loadAnomalies);
document.getElementById('anomSatFilter').addEventListener('change', loadAnomalies);
document.getElementById('anomSevFilter').addEventListener('change', loadAnomalies);
document.getElementById('anomMLFilter').addEventListener('change', loadAnomalies);
document.getElementById('refreshAlertsBtn').addEventListener('click', loadAlertHistory);
document.getElementById('alertSatFilter').addEventListener('change', loadAlertHistory);
document.getElementById('alertSevFilter').addEventListener('change', loadAlertHistory);
document.getElementById('saveAlertConfigBtn').addEventListener('click', saveAlertConfig);

document.getElementById('xtceUploadBtn').addEventListener('click', handleXtceUpload);
document.getElementById('csvUploadBtn').addEventListener('click', handleCsvUpload);
document.getElementById('csvAnalyzeBtn').addEventListener('click', handleRunAnalysis);
document.getElementById('yamcsConnectBtn').addEventListener('click', handleYamcsConnect);
document.getElementById('influxConnectBtn').addEventListener('click', handleInfluxConnect);
// Populate API push info when the Import tab is first shown
_populateApiPushInfo();

document.getElementById('rateChartRange').addEventListener('change', () => {
    if (state.chartsInitialized) updateRateChart();
});

// User chip — toggle dropdown
document.getElementById('userChip').addEventListener('click', e => {
    e.stopPropagation();
    document.getElementById('userDropdown').classList.toggle('open');
});

// Close dropdown on outside click
document.addEventListener('click', closeUserDropdown);

// Settings
document.getElementById('ddSettings').addEventListener('click', openSettings);
document.getElementById('settingsClose').addEventListener('click', closeSettings);
document.getElementById('settingsCancel').addEventListener('click', closeSettings);
document.getElementById('settingsSave').addEventListener('click', saveSettings);
document.getElementById('settingsOverlay').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeSettings();
});

// Theme picker buttons inside settings modal
document.querySelectorAll('.theme-option').forEach(btn => {
    btn.addEventListener('click', e => {
        e.stopPropagation();
        applyTheme(btn.dataset.theme);
    });
});

// Sentinel tenant switcher
document.getElementById('sentinelTenantSwitch').addEventListener('change', e => {
    handleSentinelTenantSwitch(e.target.value);
});

// Logout / Sign-In (ddLogout is reused for both actions)
document.getElementById('ddLogout').addEventListener('click', handleLogout);

// Change password
document.getElementById('ddChangePassword').addEventListener('click', openChangePassword);
document.getElementById('changePwClose').addEventListener('click', closeChangePassword);
document.getElementById('cpCancelBtn').addEventListener('click', closeChangePassword);
document.getElementById('cpSaveBtn').addEventListener('click', handleChangePassword);
document.getElementById('changePwOverlay').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeChangePassword();
});

// Login modal — form onsubmit handles Enter + button click; keep direct listener as fallback
document.getElementById('loginBtn').addEventListener('click', handleLogin);

function toggleLoginPw() {
    const inp  = document.getElementById('loginPassword');
    const show = inp.type === 'password';
    inp.type = show ? 'text' : 'password';
    document.getElementById('iconEyeOpen').style.display = show ? 'none' : '';
    document.getElementById('iconEyeOff').style.display  = show ? '' : 'none';
    document.getElementById('loginPwToggle').style.opacity = show ? '1' : '0.6';
    inp.focus();
}
// Admin — Create User modal
document.getElementById('addUserBtn').addEventListener('click', openCreateUser);
document.getElementById('createUserClose').addEventListener('click', closeCreateUser);
document.getElementById('createUserCancel').addEventListener('click', closeCreateUser);
document.getElementById('createUserSubmit').addEventListener('click', handleCreateUser);
document.getElementById('createUserModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeCreateUser();
});

// Admin — Generate Key modal
document.getElementById('generateKeyBtn').addEventListener('click', openGenerateKey);
document.getElementById('genKeyClose').addEventListener('click', closeGenerateKey);
document.getElementById('genKeyCancel').addEventListener('click', closeGenerateKey);
document.getElementById('genKeySubmit').addEventListener('click', handleGenerateKey);
document.getElementById('copyKeyBtn').addEventListener('click', () => {
    const text = document.getElementById('keyRevealBox').textContent;
    navigator.clipboard.writeText(text).then(() => toast('API key copied.', 'success'));
});
document.getElementById('generateKeyModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeGenerateKey();
});

// Admin — Create Tenant modal
document.getElementById('createTenantBtn').addEventListener('click', openCreateTenant);
document.getElementById('createTenantClose').addEventListener('click', closeCreateTenant);
document.getElementById('createTenantCancel').addEventListener('click', closeCreateTenant);
document.getElementById('createTenantSubmit').addEventListener('click', handleCreateTenant);
document.getElementById('createTenantModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeCreateTenant();
});

// Admin — Reset Password modal
document.getElementById('resetPwClose').addEventListener('click', closeResetPw);
document.getElementById('resetPwCancel').addEventListener('click', closeResetPw);
document.getElementById('resetPwSubmit').addEventListener('click', handleAdminResetPassword);
document.getElementById('resetPwModal').addEventListener('click', e => {
    if (e.target === e.currentTarget) closeResetPw();
});

// Admin — Simulator buttons
document.getElementById('startSimBtn').addEventListener('click', handleStartSimulation);
document.getElementById('injectFaultBtn').addEventListener('click', handleInjectFault);

// ============================================================
// Tab Switching
// ============================================================
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.tab === tab);
    });
    document.querySelectorAll('.tab-content').forEach(el => {
        el.style.display = el.id === `tab-${tab}` ? '' : 'none';
    });
    localStorage.setItem('sentinel-active-tab', tab);

    if (tab === 'analysis') {
        setTimeout(initCharts, 50);
    }
    if (tab === 'channels' && state.channels.length === 0) {
        loadChannels();
    }
    if (tab === 'alerts') {
        loadAnomalies();
        loadAlertHistory();
        loadAlertConfig();
        populateSatFilters();
    }
    if (tab === 'admin') {
        loadAdminTab();
    }
}

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// ============================================================
// Role-Based UI Gating
// ============================================================

// Tier mapping: 0=report_only … 4=admin; 5=sentinel (any sentinel role)
const ROLE_TIER = {
    report_only:    0,
    viewer:         1,
    operator:       2,
    tenant_manager: 3,
    admin:          4,
    developer:      5,
    sentinel_admin: 5,
    superuser:      5,
};

function applyRoleGating(role, scope) {
    const tier = scope === 'sentinel' ? 5 : (ROLE_TIER[role] ?? 0);
    const isSentinel = scope === 'sentinel';

    // Admin tab: visible for tier ≥ 3 (tenant_manager, admin) or sentinel
    const adminTab = document.getElementById('adminTabBtn');
    if (adminTab) adminTab.style.display = (tier >= 3 || isSentinel) ? '' : 'none';

    // Sentinel-only sections inside Admin tab
    document.getElementById('adminTenants')?.style.setProperty('display', isSentinel ? '' : 'none');
    document.getElementById('adminSimulator')?.style.setProperty('display', isSentinel ? '' : 'none');

    // Sentinel tenant switcher — only shown for sentinel scope users
    const switcher = document.getElementById('sentinelTenantSwitch');
    if (switcher) {
        switcher.style.display = isSentinel ? '' : 'none';
        if (isSentinel) {
            // Populate switcher with known tenants (load if needed)
            _populateSentinelTenantSwitcher();
        }
    }

    // Channels tab: save/reset buttons — operator+ (tier ≥ 2)
    _gateButtons(['saveChannelConfig', 'resetChannelConfig'], tier >= 2);

    // Import tab: upload buttons — operator+ (tier ≥ 2)
    _gateButtons(['xtceUploadBtn', 'csvUploadBtn'], tier >= 2);

    // Alerts tab: alert config save — admin+ (tier ≥ 4)
    _gateButtons(['saveAlertConfigBtn'], tier >= 4 || isSentinel);

    // Delete alert config button — find by text if no id
    document.querySelectorAll('[data-min-role="admin"]').forEach(el => {
        el.classList.toggle('role-hidden', tier < 4 && !isSentinel);
    });
}

function _gateButtons(ids, allowed) {
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        if (allowed) {
            el.classList.remove('role-hidden');
            el.disabled = false;
        } else {
            el.classList.add('role-hidden');
            el.disabled = true;
        }
    });
}

function _gateClass(attr, allowed) {
    document.querySelectorAll(`[${attr}]`).forEach(el => {
        el.classList.toggle('role-hidden', !allowed);
    });
}

// ============================================================
// Sentinel Tenant Switcher
// ============================================================

/**
 * Populates the sentinel tenant switcher dropdown.
 * Fetches /tenants (sentinel_admin can always call this).
 * Preserves any currently-selected tenant context.
 */
async function _populateSentinelTenantSwitcher() {
    const switcher = document.getElementById('sentinelTenantSwitch');
    if (!switcher) return;
    try {
        const resp = await fetch(`${API_BASE}/tenants`, { headers: authHeaders() });
        if (!resp.ok) return;
        const tenants = await resp.json();
        // Rebuild options
        const current = state.auth.tenantContext || '';
        switcher.innerHTML = '<option value="">— Select Tenant —</option>' +
            tenants.map(t => `<option value="${t.id}"${t.id === current ? ' selected' : ''}>${t.name} (${t.id})</option>`).join('');
    } catch (e) { /* silently ignore — switcher stays empty */ }
}

/**
 * Called when sentinel user picks a tenant from the topbar switcher.
 * Updates tenantContext and re-fetches all data so the dashboard
 * shows that tenant's anomalies, satellites, stats, and channels.
 */
async function handleSentinelTenantSwitch(tenantId) {
    state.auth.tenantContext = tenantId || null;
    // Persist for session
    if (tenantId) {
        localStorage.setItem('sentinel-tenant-ctx', tenantId);
    } else {
        localStorage.removeItem('sentinel-tenant-ctx');
    }
    // Re-fetch all data scoped to new tenant
    fetchAnomalies();
    fetchSatellites();
    fetchStats();
    loadChannels();
    loadAlertHistory();
    loadAlertConfig();
    // Refresh admin panel if open
    if (document.getElementById('tab-admin')?.style.display !== 'none') {
        loadAdminTab();
    }
    toast(tenantId ? `Viewing tenant: ${tenantId}` : 'No tenant selected — pick a tenant to view data.', 'info');
}

// ============================================================
// Admin Tab — State
// ============================================================
const adminState = {
    users:    [],
    keys:     [],
    tenants:  [],
    resetTarget: null,  // { id, email } for reset password modal
};

// ============================================================
// Admin Tab — Load & Render Users
// ============================================================
async function loadUsers() {
    try {
        const resp = await fetch(`${API_BASE}/users`, { headers: authHeaders() });
        if (!resp.ok) { document.getElementById('usersTableBody').innerHTML = `<tr><td colspan="6" class="empty-state">Error ${resp.status}: ${resp.statusText}</td></tr>`; return; }
        adminState.users = await resp.json();
        renderUsersTable();
    } catch (e) {
        document.getElementById('usersTableBody').innerHTML = `<tr><td colspan="6" class="empty-state">Failed to load users</td></tr>`;
    }
}

function renderUsersTable() {
    const search = (document.getElementById('userSearch')?.value || '').toLowerCase();
    const filtered = adminState.users.filter(u =>
        !search || u.email?.toLowerCase().includes(search) || u.role?.toLowerCase().includes(search)
    );
    const tbody = document.getElementById('usersTableBody');
    if (!filtered.length) {
        const msg = adminState.users.length
            ? 'No users match the search filter'
            : 'No team members yet — use + Add User to invite your first user';
        tbody.innerHTML = `<tr><td colspan="6" class="empty-state">${msg}</td></tr>`;
        return;
    }
    tbody.innerHTML = filtered.map(u => {
        const displayName = u.display_name || u.email || '?';
        const initial = displayName[0].toUpperCase();
        const statusClass = u.active ? 'active' : 'inactive';
        const statusLabel = u.active ? '● Active' : '○ Inactive';
        const lastLogin = u.last_login ? formatDateTime(u.last_login) : 'Never';
        const deactivateLabel = u.active ? 'Deactivate' : 'Reactivate';
        const deactivateClass = u.active ? 'danger' : 'success';
        const roles = ['viewer','operator','tenant_manager','admin'];
        const roleOptions = roles.map(r =>
            `<option value="${r}" ${r === u.role ? 'selected' : ''}>${r}</option>`
        ).join('');
        // Use data-* attributes to avoid XSS in inline onclick handlers
        return `
        <tr>
            <td><div class="user-avatar-sm">${initial}</div></td>
            <td>
                <div style="color:var(--text-primary);font-weight:600">${displayName}</div>
                <div style="color:var(--text-muted);font-size:10px">${u.email}</div>
                ${u.phone ? `<div style="color:var(--text-muted);font-size:10px">${u.phone}</div>` : ''}
            </td>
            <td>
                <select class="role-select" data-user-id="${u.id}">
                    ${roleOptions}
                </select>
            </td>
            <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
            <td>${lastLogin}</td>
            <td>
                <button class="btn-action" data-action="reset-pw" data-user-id="${u.id}" data-user-email="${u.email}">Reset PW</button>
                <button class="btn-action ${deactivateClass}" data-action="${u.active ? 'deactivate' : 'reactivate'}" data-user-id="${u.id}">${deactivateLabel}</button>
            </td>
        </tr>`;
    }).join('');
}

async function handleRoleChange(userId, newRole) {
    try {
        const resp = await fetch(`${API_BASE}/users/${userId}/role`, {
            method: 'PATCH',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ role: newRole }),
        });
        if (!resp.ok) { toast(`Role change failed: ${resp.status}`, 'error'); return; }
        toast('Role updated.', 'success');
        await loadUsers();
    } catch (e) { toast('Failed to change role: ' + e.message, 'error'); }
}

async function handleDeactivateUser(userId) {
    try {
        const resp = await fetch(`${API_BASE}/users/${userId}/deactivate`, {
            method: 'POST',
            headers: authHeaders(),
        });
        if (!resp.ok) { toast(`Deactivate failed: ${resp.status}`, 'error'); return; }
        toast('User deactivated.', 'success');
        await loadUsers();
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
}

async function handleReactivateUser(userId) {
    try {
        const resp = await fetch(`${API_BASE}/users/${userId}/reactivate`, {
            method: 'POST',
            headers: authHeaders(),
        });
        if (!resp.ok) { toast(`Reactivate failed: ${resp.status}`, 'error'); return; }
        toast('User reactivated.', 'success');
        await loadUsers();
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
}

// Event delegation for users table — avoids XSS from inline onclick with user data
document.getElementById('usersTableBody').addEventListener('change', e => {
    const sel = e.target.closest('select.role-select');
    if (sel) handleRoleChange(sel.dataset.userId, sel.value);
});
document.getElementById('usersTableBody').addEventListener('click', e => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const { action, userId, userEmail } = btn.dataset;
    if (action === 'reset-pw')   openResetPw(userId, userEmail);
    if (action === 'deactivate') handleDeactivateUser(userId);
    if (action === 'reactivate') handleReactivateUser(userId);
});

// Keep these on window for backward compatibility (called from other code paths)
window.handleRoleChange     = handleRoleChange;
window.handleDeactivateUser = handleDeactivateUser;
window.handleReactivateUser = handleReactivateUser;
window.openResetPw          = openResetPw;

// ============================================================
// Admin Tab — Create User Modal
// ============================================================
function openCreateUser() {
    const dn = document.getElementById('newUserDisplayName');
    if (dn) dn.value = '';
    document.getElementById('newUserEmail').value    = '';
    const ph = document.getElementById('newUserPhone');
    if (ph) ph.value = '';
    document.getElementById('newUserPassword').value = '';
    document.getElementById('newUserRole').value     = 'viewer';
    document.getElementById('createUserError').textContent = '';
    document.getElementById('createUserModal').classList.add('open');
    setTimeout(() => (dn || document.getElementById('newUserEmail')).focus(), 50);
}

function closeCreateUser() {
    document.getElementById('createUserModal').classList.remove('open');
}

async function handleCreateUser() {
    const displayName = document.getElementById('newUserDisplayName')?.value.trim() || '';
    const email       = document.getElementById('newUserEmail').value.trim();
    const phone       = document.getElementById('newUserPhone')?.value.trim() || '';
    const password    = document.getElementById('newUserPassword').value;
    const role        = document.getElementById('newUserRole').value;
    const errorEl     = document.getElementById('createUserError');
    const btn         = document.getElementById('createUserSubmit');

    if (!email || !password) { errorEl.textContent = 'Email and password are required.'; return; }
    if (password.length < 8) { errorEl.textContent = 'Password must be at least 8 characters.'; return; }

    btn.disabled = true;
    errorEl.textContent = '';
    try {
        const body = { email, password, role };
        if (displayName) body.display_name = displayName;
        if (phone)        body.phone = phone;
        const resp = await fetch(`${API_BASE}/users`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            errorEl.textContent = err.detail || `Error ${resp.status}`;
            return;
        }
        closeCreateUser();
        toast(`User ${email} created.`, 'success');
        await loadUsers();
    } catch (e) {
        errorEl.textContent = 'Network error: ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

// ============================================================
// Admin Tab — API Key Management
// ============================================================
async function loadApiKeys() {
    try {
        const resp = await fetch(`${API_BASE}/keys`, { headers: authHeaders() });
        if (!resp.ok) { document.getElementById('keysTableBody').innerHTML = `<tr><td colspan="5" class="empty-state">Error ${resp.status}</td></tr>`; return; }
        adminState.keys = await resp.json();
        renderApiKeysTable();
    } catch (e) {
        document.getElementById('keysTableBody').innerHTML = `<tr><td colspan="5" class="empty-state">Failed to load keys</td></tr>`;
    }
}

function renderApiKeysTable() {
    const tbody = document.getElementById('keysTableBody');
    if (!adminState.keys.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No API keys — generate one to allow machine access.</td></tr>';
        return;
    }
    tbody.innerHTML = adminState.keys.map(k => `
        <tr>
            <td style="color:var(--text-primary)">${k.label}</td>
            <td><code>${k.hash_prefix}…</code></td>
            <td>${k.created_at ? formatDateTime(k.created_at) : '—'}</td>
            <td>${k.last_used_at ? formatDateTime(k.last_used_at) : 'Never'}</td>
            <td><button class="btn-action danger" onclick="handleRevokeKey('${k.hash_prefix}')">Revoke</button></td>
        </tr>`
    ).join('');
}

window.handleRevokeKey = async function(prefix) {
    if (!confirm(`Revoke key ${prefix}…? This cannot be undone.`)) return;
    try {
        const resp = await fetch(`${API_BASE}/keys/${prefix}`, {
            method: 'DELETE',
            headers: authHeaders(),
        });
        if (!resp.ok) { toast(`Revoke failed: ${resp.status}`, 'error'); return; }
        toast('API key revoked.', 'success');
        await loadApiKeys();
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
};

function openGenerateKey() {
    document.getElementById('newKeyLabel').value   = '';
    document.getElementById('genKeyError').textContent = '';
    document.getElementById('genKeyForm').style.display = '';
    document.getElementById('keyRevealSection').style.display = 'none';
    document.getElementById('genKeyFooter').style.display = '';
    document.getElementById('generateKeyModal').classList.add('open');
    setTimeout(() => document.getElementById('newKeyLabel').focus(), 50);
}

function closeGenerateKey() {
    document.getElementById('generateKeyModal').classList.remove('open');
}

async function handleGenerateKey() {
    const label   = document.getElementById('newKeyLabel').value.trim();
    const errorEl = document.getElementById('genKeyError');
    const btn     = document.getElementById('genKeySubmit');
    if (!label) { errorEl.textContent = 'A label is required.'; return; }

    btn.disabled = true;
    errorEl.textContent = '';
    try {
        const resp = await fetch(`${API_BASE}/keys`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ label }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            errorEl.textContent = err.detail || `Error ${resp.status}`;
            return;
        }
        const data = await resp.json();
        // Show key reveal (one-time)
        document.getElementById('genKeyForm').style.display = 'none';
        document.getElementById('keyRevealSection').style.display = '';
        document.getElementById('keyRevealBox').textContent = data.key;
        document.getElementById('genKeyFooter').style.display = 'none';
        await loadApiKeys();
    } catch (e) {
        errorEl.textContent = 'Network error: ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

// ============================================================
// Admin Tab — Tenant Management (sentinel_admin only)
// ============================================================
async function loadTenants() {
    try {
        const resp = await fetch(`${API_BASE}/tenants`, { headers: authHeaders() });
        if (!resp.ok) { document.getElementById('tenantsTableBody').innerHTML = `<tr><td colspan="6" class="empty-state">Error ${resp.status}</td></tr>`; return; }
        adminState.tenants = await resp.json();
        renderTenantsTable();
    } catch (e) {
        document.getElementById('tenantsTableBody').innerHTML = `<tr><td colspan="6" class="empty-state">Failed to load tenants</td></tr>`;
    }
}

function renderTenantsTable() {
    const tbody  = document.getElementById('tenantsTableBody');
    const search = (document.getElementById('tenantSearch')?.value || '').toLowerCase();
    const tenants = search
        ? adminState.tenants.filter(t =>
            t.id?.toLowerCase().includes(search) || t.name?.toLowerCase().includes(search))
        : adminState.tenants;
    if (!tenants.length) {
        tbody.innerHTML = `<tr><td colspan="6" class="empty-state">${search ? 'No tenants match the search.' : 'No tenants found.'}</td></tr>`;
        return;
    }
    tbody.innerHTML = tenants.map(t => {
        const statusClass = t.active ? 'active' : 'inactive';
        const statusLabel = t.active ? '● Active' : '○ Inactive';
        const toggleFn  = `handleToggleTenant('${t.id}', ${!t.active})`;
        const toggleLabel = t.active ? 'Disable' : 'Enable';
        return `
        <tr>
            <td style="color:var(--text-primary);font-family:monospace">${t.id}</td>
            <td>${t.name}</td>
            <td><span class="role-badge">${t.plan}</span></td>
            <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
            <td>${t.created_at ? formatDateTime(t.created_at) : '—'}</td>
            <td><button class="btn-action" onclick="${toggleFn}">${toggleLabel}</button></td>
        </tr>`;
    }).join('');
}

window.handleToggleTenant = async function(tenantId, active) {
    try {
        const resp = await fetch(`${API_BASE}/tenants/${tenantId}`, {
            method: 'PATCH',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ active }),
        });
        if (!resp.ok) { toast(`Update failed: ${resp.status}`, 'error'); return; }
        toast(`Tenant ${active ? 'enabled' : 'disabled'}.`, 'success');
        await loadTenants();
    } catch (e) { toast('Failed: ' + e.message, 'error'); }
};

/**
 * Converts a display name to a valid tenant ID slug.
 * Called live as the user types the tenant name.
 */
window.autoSlugTenantId = function(name) {
    const slug = name
        .toLowerCase()
        .normalize('NFD').replace(/[\u0300-\u036f]/g, '')   // strip accents
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 64);
    document.getElementById('newTenantId').value = slug;
    const autoLabel = document.getElementById('tenantIdAutoLabel');
    if (autoLabel) autoLabel.style.display = slug ? '' : 'none';
};

function openCreateTenant() {
    document.getElementById('newTenantId').value   = '';
    document.getElementById('newTenantName').value = '';
    document.getElementById('newTenantPlan').value = 'free';
    document.getElementById('createTenantError').textContent = '';
    const autoLabel = document.getElementById('tenantIdAutoLabel');
    if (autoLabel) autoLabel.style.display = 'none';
    document.getElementById('createTenantModal').classList.add('open');
    setTimeout(() => document.getElementById('newTenantName').focus(), 50);
}

function closeCreateTenant() {
    document.getElementById('createTenantModal').classList.remove('open');
}

async function handleCreateTenant() {
    const id      = document.getElementById('newTenantId').value.trim().toLowerCase();
    const name    = document.getElementById('newTenantName').value.trim();
    const plan    = document.getElementById('newTenantPlan').value;
    const errorEl = document.getElementById('createTenantError');
    const btn     = document.getElementById('createTenantSubmit');

    if (!id || !name) { errorEl.textContent = 'Tenant ID and name are required.'; return; }
    if (!/^[a-z0-9-]{2,64}$/.test(id)) { errorEl.textContent = 'ID: lowercase letters, digits, hyphens only.'; return; }

    btn.disabled = true;
    errorEl.textContent = '';
    try {
        const resp = await fetch(`${API_BASE}/tenants`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ id, name, plan }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            errorEl.textContent = err.detail || `Error ${resp.status}`;
            return;
        }
        closeCreateTenant();
        toast(`Tenant "${name}" created.`, 'success');
        await loadTenants();
    } catch (e) {
        errorEl.textContent = 'Network error: ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

// ============================================================
// Admin Tab — Simulator
// ============================================================
async function handleStartSimulation() {
    const satelliteId = document.getElementById('simSatId').value.trim();
    const duration    = parseFloat(document.getElementById('simDuration').value);
    const rate        = parseFloat(document.getElementById('simRate').value);
    if (!satelliteId) { toast('Enter a Satellite ID first.', 'warning'); return; }

    const statusEl = document.getElementById('simStatus');
    statusEl.style.display = '';
    statusEl.textContent = `Starting simulation for ${satelliteId}…`;

    try {
        const resp = await fetch(`${API_BASE}/simulate/start`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ satellite_id: satelliteId, duration_seconds: duration, rate_hz: rate }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) { statusEl.textContent = `Error: ${data.detail || resp.status}`; toast('Simulation start failed.', 'error'); return; }
        statusEl.textContent = data.message || 'Simulation started.';
        toast('Simulation started.', 'success');
    } catch (e) {
        statusEl.textContent = 'Network error: ' + e.message;
        toast('Failed to start simulation.', 'error');
    }
}

async function handleInjectFault() {
    const satelliteId = document.getElementById('simSatId').value.trim();
    const scenario    = document.getElementById('simScenario').value;
    if (!satelliteId) { toast('Enter a Satellite ID first.', 'warning'); return; }
    if (!scenario)    { toast('Select a fault scenario.', 'warning'); return; }

    const statusEl = document.getElementById('simStatus');
    statusEl.style.display = '';
    statusEl.textContent = `Injecting fault "${scenario}" into ${satelliteId}…`;

    try {
        const resp = await fetch(`${API_BASE}/simulate/inject`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ satellite_id: satelliteId, scenario }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) { statusEl.textContent = `Error: ${data.detail || resp.status}`; toast('Fault injection failed.', 'error'); return; }
        statusEl.textContent = data.message || 'Fault injected.';
        toast(`Fault "${scenario}" injected.`, 'success');
    } catch (e) {
        statusEl.textContent = 'Network error: ' + e.message;
        toast('Failed to inject fault.', 'error');
    }
}

// ============================================================
// Admin Tab — Reset Password Modal
// ============================================================
function openResetPw(userId, email) {
    adminState.resetTarget = { id: userId, email };
    document.getElementById('resetPwEmail').value   = email;
    document.getElementById('resetPwNew').value     = '';
    document.getElementById('resetPwConfirm').value = '';
    document.getElementById('resetPwError').textContent = '';
    document.getElementById('resetPwModal').classList.add('open');
    setTimeout(() => document.getElementById('resetPwNew').focus(), 50);
}

function closeResetPw() {
    document.getElementById('resetPwModal').classList.remove('open');
    adminState.resetTarget = null;
}

async function handleAdminResetPassword() {
    const target   = adminState.resetTarget;
    const newPw    = document.getElementById('resetPwNew').value;
    const confirm  = document.getElementById('resetPwConfirm').value;
    const errorEl  = document.getElementById('resetPwError');
    const btn      = document.getElementById('resetPwSubmit');

    if (!target) return;
    if (!newPw || !confirm) { errorEl.textContent = 'Both fields are required.'; return; }
    if (newPw !== confirm)  { errorEl.textContent = 'Passwords do not match.'; return; }
    if (newPw.length < 8)   { errorEl.textContent = 'Minimum 8 characters.'; return; }

    btn.disabled = true;
    errorEl.textContent = '';
    try {
        const resp = await fetch(`${API_BASE}/users/${target.id}/reset-password`, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ new_password: newPw }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            errorEl.textContent = err.detail || `Error ${resp.status}`;
            return;
        }
        closeResetPw();
        toast(`Password reset for ${target.email}.`, 'success');
    } catch (e) {
        errorEl.textContent = 'Network error: ' + e.message;
    } finally {
        btn.disabled = false;
    }
}

// ============================================================
// Admin Tab — Master Load Function
// ============================================================
async function loadAdminTab() {
    const user  = state.auth.user;
    const scope = user?.scope || '';
    await loadUsers();
    await loadApiKeys();
    if (scope === 'sentinel') await loadTenants();
}

// ============================================================
// Init
// ============================================================

// 1. Apply theme before first paint (prevents flash-of-wrong-theme)
initTheme();

// 2. Restore auth session, then fire all initial data loads once the token is ready.
//    Without this await, fetchAnomalies/fetchSatellites/fetchStats would fire with
//    no Authorization header (race condition) and silently get 401 back.
initAuth().then(() => {
    fetchAnomalies();
    fetchSatellites();
    fetchStats();
    // Restore last active tab (must be after auth so role-gated tabs are visible)
    const savedTab = localStorage.getItem('sentinel-active-tab');
    if (savedTab && document.querySelector(`.tab-btn[data-tab="${savedTab}"]`)) {
        switchTab(savedTab);
    }
});

// 3. Connect live feed (WebSocket doesn't need auth)
connectWebSocket();
fetchHealth();

// 4. File drop zones
setupDropZone('xtceDropZone', 'xtceFileInput', 'xtceFile', 'xtceUploadBtn', 'xtceFileInfo');
setupDropZone('csvDropZone',  'csvFileInput',  'csvFile',  'csvUploadBtn',  'csvFileInfo');

// 5. Initial subsystem grid
renderSubsystems();

// 6. Periodic refresh — stored IDs so poll interval can be changed in Settings
const _storedPollMs = parseInt(localStorage.getItem('sentinel-poll-interval') || '10000', 10);
state.pollIntervals.health     = setInterval(fetchHealth,     15000);
state.pollIntervals.anomaly    = setInterval(pollNewAnomalies, _storedPollMs);
state.pollIntervals.satellites = setInterval(fetchSatellites, 30000);
state.pollIntervals.stats      = setInterval(fetchStats,      30000);
