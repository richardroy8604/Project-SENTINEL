/**
 * ULTRON TACTICAL REMOTE COMPANION — CLIENT APPLICATION
 * High-performance WebSocket event listener & responsive HUD
 */

(function () {
  'use strict';

  // DOM Elements
  const wsStatusEl = document.getElementById('ws-status');
  const mqttStatusEl = document.getElementById('mqtt-status');
  const securityStateEl = document.getElementById('security-state');
  const personCountEl = document.getElementById('person-count');
  const incidentBannerEl = document.getElementById('incident-banner');
  const incidentTextEl = document.getElementById('incident-text');
  const incidentCloseBtn = document.getElementById('incident-close-btn');

  // Hero Snapshot Elements
  const heroImageEl = document.getElementById('hero-image');
  const heroImageEmptyEl = document.getElementById('hero-image-empty');
  const heroTriggerEl = document.getElementById('hero-trigger');
  const heroThreatEl = document.getElementById('hero-threat');
  const heroTimeEl = document.getElementById('hero-time');
  const heroDescEl = document.getElementById('hero-desc');
  const heroExpandBtn = document.getElementById('hero-expand-btn');
  const snapshotStripEl = document.getElementById('snapshot-strip');
  const personsListEl = document.getElementById('persons-list');

  // Chat Elements
  const chatStreamEl = document.getElementById('chat-stream');
  const speakingIndicatorEl = document.getElementById('speaking-indicator');

  // Lightbox Elements
  const lightboxModalEl = document.getElementById('lightbox-modal');
  const lightboxImgEl = document.getElementById('lightbox-img');
  const lightboxCloseBtn = document.getElementById('lightbox-close');

  let currentSnapshotSrc = '';
  let ws = null;
  let pingInterval = null;
  let reconnectTimeout = null;

  // ── Formatters ──────────────────────────────────────────────────
  function formatTimestamp(ts) {
    if (!ts) return '--:--:--';
    const date = new Date(ts * 1000);
    return date.toTimeString().split(' ')[0];
  }

  function formatDwell(sec) {
    if (sec < 60) return `${Math.round(sec)}s`;
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m}m ${s}s`;
  }

  // ── UI Updaters ─────────────────────────────────────────────────
  function setSecurityState(state) {
    if (!securityStateEl) return;
    const s = (state || 'PATROL').toUpperCase();
    securityStateEl.className = 'status-pill';
    
    if (s === 'PATROL') {
      securityStateEl.classList.add('state-patrol');
    } else if (s === 'ALERT') {
      securityStateEl.classList.add('state-alert');
    } else if (s === 'SUSPICIOUS') {
      securityStateEl.classList.add('state-suspicious');
    } else if (s === 'CRITICAL') {
      securityStateEl.classList.add('state-critical');
    } else {
      securityStateEl.classList.add('state-patrol');
    }

    const dot = securityStateEl.querySelector('.pill-dot');
    securityStateEl.innerHTML = '';
    if (dot) securityStateEl.appendChild(dot);
    securityStateEl.append(` STATE: ${s}`);
  }

  function setPersonCount(count) {
    if (!personCountEl) return;
    const c = count || 0;
    personCountEl.textContent = `VISITORS: ${c}`;
  }

  function setMQTTStatus(connected, broker) {
    if (!mqttStatusEl) return;
    mqttStatusEl.className = 'status-pill';
    if (connected) {
      mqttStatusEl.classList.add('status-online');
      mqttStatusEl.innerHTML = `<span class="pill-dot"></span> MQTT: CONNECTED`;
    } else {
      mqttStatusEl.classList.add('status-offline');
      mqttStatusEl.innerHTML = `<span class="pill-dot"></span> MQTT: STANDBY`;
    }
  }

  function setWSStatus(connected) {
    if (!wsStatusEl) return;
    wsStatusEl.className = 'status-pill';
    if (connected) {
      wsStatusEl.classList.add('status-online');
      wsStatusEl.innerHTML = `<span class="pill-dot"></span> LIVE LINK`;
    } else {
      wsStatusEl.classList.add('status-offline');
      wsStatusEl.innerHTML = `<span class="pill-dot"></span> RECONNECTING...`;
    }
  }

  function triggerIncidentAlert(text, severity) {
    if (!incidentBannerEl || !incidentTextEl) return;
    incidentTextEl.textContent = text;
    incidentBannerEl.classList.add('active');
    
    // Sound/vibrate if supported
    if (navigator.vibrate) {
      navigator.vibrate([200, 100, 200]);
    }

    setTimeout(() => {
      incidentBannerEl.classList.remove('active');
    }, 9000);
  }

  if (incidentCloseBtn) {
    incidentCloseBtn.addEventListener('click', () => {
      incidentBannerEl.classList.remove('active');
    });
  }

  // ── Snapshot Management ─────────────────────────────────────────
  function displayHeroSnapshot(snap) {
    if (!snap) return;
    const src = snap.base64 || snap.url;
    if (!src) return;

    currentSnapshotSrc = src;
    heroImageEl.src = src;
    heroImageEl.style.display = 'block';
    if (heroImageEmptyEl) heroImageEmptyEl.style.display = 'none';

    // Update Pills
    if (heroTriggerEl) heroTriggerEl.textContent = `EVENT: ${snap.trigger || 'MANUAL'}`;
    
    const threat = (snap.threat_level || 'NOMINAL').toUpperCase();
    if (heroThreatEl) {
      heroThreatEl.textContent = threat;
      heroThreatEl.className = `hud-pill ${threat === 'CRITICAL' ? 'threat-critical' : 'threat-nominal'}`;
    }

    if (heroTimeEl) heroTimeEl.textContent = formatTimestamp(snap.timestamp);
    if (heroDescEl) heroDescEl.textContent = snap.description || 'Forensic action capture';

    // Flash wrapper if critical
    const wrapper = document.querySelector('.hero-image-wrapper');
    if (wrapper && threat === 'CRITICAL') {
      wrapper.style.borderColor = '#ff1744';
      wrapper.style.boxShadow = '0 0 20px rgba(255, 23, 68, 0.4)';
      setTimeout(() => {
        wrapper.style.borderColor = '';
        wrapper.style.boxShadow = '';
      }, 3000);
    }
  }

  function addSnapshotThumbnail(snap, prepend = true) {
    if (!snapshotStripEl || !snap) return;
    const src = snap.base64 || snap.url;
    if (!src) return;

    const card = document.createElement('div');
    card.className = 'snapshot-thumb-card';
    card.innerHTML = `
      <img src="${src}" class="snapshot-thumb-img" alt="${snap.trigger || 'Snapshot'}" />
      <span class="snapshot-thumb-badge">${snap.trigger || 'SNAP'}</span>
    `;

    card.addEventListener('click', () => {
      displayHeroSnapshot(snap);
      document.querySelectorAll('.snapshot-thumb-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
    });

    if (prepend && snapshotStripEl.firstChild) {
      snapshotStripEl.insertBefore(card, snapshotStripEl.firstChild);
    } else {
      snapshotStripEl.appendChild(card);
    }

    // Keep strip capped at 18
    while (snapshotStripEl.children.length > 18) {
      snapshotStripEl.removeChild(snapshotStripEl.lastChild);
    }
  }

  function updateTrackedPersons(persons) {
    if (!personsListEl) return;
    personsListEl.innerHTML = '';

    if (!persons || persons.length === 0) {
      personsListEl.innerHTML = `
        <div class="person-row" style="color: var(--text-dim); justify-content: center;">
          <span>No subjects in perimeter</span>
        </div>
      `;
      return;
    }

    persons.forEach(p => {
      const row = document.createElement('div');
      row.className = 'person-row';

      let badgesHtml = '';
      if (p.phone) {
        badgesHtml += `<span class="badge-tag phone">PHONE</span>`;
      }
      if (p.weapon) {
        badgesHtml += `<span class="badge-tag weapon">ARMED: ${p.weapon}</span>`;
      }

      row.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="person-id-badge">ID #${p.id}</span>
          <span style="color: var(--text-muted); font-size: 0.7rem;">dwell: ${formatDwell(p.dwell_s)}</span>
        </div>
        <div class="person-badges">
          ${badgesHtml || '<span style="color: var(--text-dim); font-size: 0.65rem;">CLEAR</span>'}
        </div>
      `;
      personsListEl.appendChild(row);
    });
  }

  // ── Chat Stream Management ──────────────────────────────────────
  function appendChatMessage(msg) {
    if (!chatStreamEl || !msg) return;

    const role = (msg.role || 'visitor').toLowerCase();
    const isUltron = role === 'ultron' || role === 'assistant';

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${isUltron ? 'ultron' : 'visitor'}`;

    const authorText = isUltron ? 'ULTRON SENTINEL' : 'VISITOR';
    const timeText = formatTimestamp(msg.timestamp);
    const latencyBadge = msg.latency_ms ? `<span class="bubble-badge">${Math.round(msg.latency_ms)}ms</span>` : '';
    const autoBadge = msg.autonomous ? `<span class="bubble-badge" style="border: 1px solid #ffd600; color: #ffd600;">AUTONOMOUS</span>` : '';

    bubble.innerHTML = `
      <div class="bubble-meta">
        <span class="bubble-author">${authorText}</span>
        ${latencyBadge}
        ${autoBadge}
        <span>${timeText}</span>
      </div>
      <div class="bubble-content">${escapeHTML(msg.text || '')}</div>
    `;

    chatStreamEl.appendChild(bubble);
    chatStreamEl.scrollTop = chatStreamEl.scrollHeight;
  }

  function escapeHTML(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ── Lightbox Interaction ────────────────────────────────────────
  function openLightbox(src) {
    if (!src || !lightboxModalEl || !lightboxImgEl) return;
    lightboxImgEl.src = src;
    lightboxModalEl.classList.add('active');
  }

  if (heroImageEl) {
    heroImageEl.addEventListener('click', () => openLightbox(currentSnapshotSrc));
  }
  if (heroExpandBtn) {
    heroExpandBtn.addEventListener('click', () => openLightbox(currentSnapshotSrc));
  }
  if (lightboxCloseBtn) {
    lightboxCloseBtn.addEventListener('click', () => lightboxModalEl.classList.remove('active'));
  }
  if (lightboxModalEl) {
    lightboxModalEl.addEventListener('click', (e) => {
      if (e.target === lightboxModalEl) lightboxModalEl.classList.remove('active');
    });
  }

  // ── WebSocket Connection & Heartbeat ────────────────────────────
  function connectWebSocket() {
    clearTimeout(reconnectTimeout);
    clearInterval(pingInterval);

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      console.error('[ULTRON WS] Socket creation failed:', e);
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      console.log('[ULTRON WS] Connected to Ultron Edge Hub');
      setWSStatus(true);

      // Heartbeat ping every 15 seconds
      pingInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 15000);
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleServerMessage(payload);
      } catch (err) {
        console.warn('[ULTRON WS] Parse error:', err);
      }
    };

    ws.onclose = () => {
      console.log('[ULTRON WS] Connection closed');
      setWSStatus(false);
      clearInterval(pingInterval);
      scheduleReconnect();
    };

    ws.onerror = (err) => {
      console.error('[ULTRON WS] Error:', err);
      ws.close();
    };
  }

  function scheduleReconnect() {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = setTimeout(() => {
      console.log('[ULTRON WS] Attempting reconnect...');
      connectWebSocket();
    }, 2500);
  }

  // ── Server Event Dispatcher ─────────────────────────────────────
  function handleServerMessage(msg) {
    const { type, data } = msg;

    switch (type) {
      case 'init':
        setSecurityState(data.security_state);
        setPersonCount(data.person_count);
        setMQTTStatus(data.mqtt_connected, data.mqtt_broker);
        updateTrackedPersons(data.persons);

        if (data.latest_snapshot) {
          displayHeroSnapshot(data.latest_snapshot);
        }
        if (data.recent_snapshots && Array.isArray(data.recent_snapshots)) {
          if (snapshotStripEl) snapshotStripEl.innerHTML = '';
          data.recent_snapshots.forEach(s => addSnapshotThumbnail(s, false));
        }
        if (data.chat_history && Array.isArray(data.chat_history)) {
          if (chatStreamEl) chatStreamEl.innerHTML = '';
          data.chat_history.forEach(m => appendChatMessage(m));
        }
        break;

      case 'state_changed':
        setSecurityState(data.security_state);
        setPersonCount(data.person_count);
        break;

      case 'chat_message':
        appendChatMessage(data);
        break;

      case 'snapshot_captured':
        displayHeroSnapshot(data);
        addSnapshotThumbnail(data, true);
        break;

      case 'alert':
        const alertTxt = `[${data.severity}] ${data.alert_type}: ${data.weapon || data.type || ''}`;
        triggerIncidentAlert(alertTxt, data.severity);
        break;

      case 'speaking_status':
        if (speakingIndicatorEl) {
          if (data.is_speaking) {
            speakingIndicatorEl.classList.add('active');
          } else {
            speakingIndicatorEl.classList.remove('active');
          }
        }
        break;

      case 'pong':
        // Heartbeat ACK
        break;

      default:
        break;
    }
  }

  // Boot
  document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
  });

})();
