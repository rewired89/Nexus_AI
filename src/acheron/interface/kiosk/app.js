/**
 * Nexus Kiosk — Client-side logic for the Acheron voice interface.
 *
 * Connects to the WebSocket server at /ws and handles:
 *   - Voice recording via MediaRecorder (16-bit PCM)
 *   - Text input fallback
 *   - Response rendering with epistemic coloring
 *   - 2D avatar animation (canvas-based; Three.js GLB optional)
 *   - Audio playback with lip-sync
 */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ws = null;
let isRecording = false;
let mediaRecorder = null;
let audioContext = null;
let audioChunks = [];
let avatarState = "idle";
let avatarParams = {};

const responsePanel = document.getElementById("response-panel");
const queryInput = document.getElementById("query-input");
const micBtn = document.getElementById("mic-btn");
const sendBtn = document.getElementById("send-btn");
const modeSelect = document.getElementById("mode-select");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const avatarCanvas = document.getElementById("avatar-canvas");
const avatarLabel = document.getElementById("avatar-state-label");
const avatarGlow = document.getElementById("avatar-glow");

// ---------------------------------------------------------------------------
// WebSocket connection
// ---------------------------------------------------------------------------

function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws`;
    ws = new WebSocket(url);

    ws.onopen = () => {
        setStatus("active", "Connected");
    };

    ws.onclose = () => {
        setStatus("inactive", "Disconnected");
        // Reconnect after 3 seconds.
        setTimeout(connect, 3000);
    };

    ws.onerror = () => {
        setStatus("error", "Connection error");
    };

    ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
            // Binary: TTS audio.
            playAudio(event.data);
            return;
        }

        let msg;
        try {
            msg = JSON.parse(event.data);
        } catch {
            return;
        }

        handleMessage(msg);
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case "status":
            setStatus("active", msg.message);
            break;

        case "pong":
            break;

        case "transcription":
            appendMessage("user", msg.text, "Voice input");
            break;

        case "response":
            renderResponse(msg);
            break;

        case "avatar":
            updateAvatar(msg);
            break;

        case "error":
            appendMessage("error", msg.message, "Error");
            setAvatarState("idle");
            break;
    }
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function setStatus(state, text) {
    if (statusDot) {
        statusDot.className = `status-dot ${state}`;
    }
    if (statusText) {
        statusText.textContent = text;
    }
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function appendMessage(type, content, label) {
    const div = document.createElement("div");
    div.className = `message ${type}`;

    if (label) {
        const labelEl = document.createElement("span");
        labelEl.className = `label ${type}`;
        labelEl.textContent = label;
        div.appendChild(labelEl);
    }

    const body = document.createElement("div");
    body.innerHTML = formatMarkdown(content);
    div.appendChild(body);

    responsePanel.appendChild(div);
    responsePanel.scrollTop = responsePanel.scrollHeight;
}

function renderResponse(msg) {
    const div = document.createElement("div");
    div.className = "message system";

    // Mode label
    if (msg.mode) {
        const label = document.createElement("span");
        label.className = "label mode";
        label.textContent = msg.mode.toUpperCase();
        div.appendChild(label);
    }

    // Answer body
    const body = document.createElement("div");
    body.innerHTML = formatMarkdown(msg.answer || "No response.");
    div.appendChild(body);

    // Sources
    if (msg.sources && msg.sources.length > 0) {
        const srcDiv = document.createElement("div");
        srcDiv.className = "sources-list";
        srcDiv.innerHTML = "<strong>Sources:</strong>";
        msg.sources.forEach((src) => {
            const item = document.createElement("div");
            item.className = "source-item";
            item.innerHTML = `
                <span class="source-score">${src.score}</span>
                <span>${escapeHtml(src.title || "Untitled")}${
                    src.doi ? ` (${src.doi})` : ""
                }</span>
            `;
            srcDiv.appendChild(item);
        });
        div.appendChild(srcDiv);
    }

    responsePanel.appendChild(div);
    responsePanel.scrollTop = responsePanel.scrollHeight;
    setAvatarState("idle");
}

function formatMarkdown(text) {
    if (!text) return "";
    // Minimal markdown: headers, bold, code blocks, lists.
    let html = escapeHtml(text);
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, "<pre><code>$2</code></pre>");
    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Headers
    html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    // List items
    html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
    // Newlines
    html = html.replace(/\n/g, "<br>");
    return html;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Text input
// ---------------------------------------------------------------------------

function sendTextQuery() {
    const query = queryInput.value.trim();
    if (!query || !ws || ws.readyState !== WebSocket.OPEN) return;

    appendMessage("user", query, "Query");
    appendThinkingIndicator();

    ws.send(JSON.stringify({ type: "text", query: query }));
    queryInput.value = "";
}

function appendThinkingIndicator() {
    const div = document.createElement("div");
    div.className = "message system";
    div.id = "thinking-indicator";
    div.innerHTML = '<div class="thinking-dots"><span></span><span></span><span></span></div>';
    responsePanel.appendChild(div);
    responsePanel.scrollTop = responsePanel.scrollHeight;
}

function removeThinkingIndicator() {
    const el = document.getElementById("thinking-indicator");
    if (el) el.remove();
}

// ---------------------------------------------------------------------------
// Voice recording
// ---------------------------------------------------------------------------

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            },
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000,
        });

        const source = audioContext.createMediaStreamSource(stream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const float32 = e.inputBuffer.getChannelData(0);
            // Convert float32 → int16 PCM.
            const int16 = new Int16Array(float32.length);
            for (let i = 0; i < float32.length; i++) {
                const s = Math.max(-1, Math.min(1, float32[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            // Send raw PCM to server.
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(int16.buffer);
            }
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

        // Tell server we're starting.
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "audio_start" }));
        }

        isRecording = true;
        micBtn.classList.add("recording");
        micBtn.textContent = "\u23F9"; // stop icon
        setAvatarState("listening");

        // Store refs for cleanup.
        mediaRecorder = { stream, source, processor };
    } catch (err) {
        console.error("Microphone access denied:", err);
        setStatus("error", "Microphone access denied");
    }
}

function stopRecording() {
    isRecording = false;
    micBtn.classList.remove("recording");
    micBtn.textContent = "\uD83C\uDF99"; // microphone icon

    if (mediaRecorder) {
        mediaRecorder.processor.disconnect();
        mediaRecorder.source.disconnect();
        mediaRecorder.stream.getTracks().forEach((t) => t.stop());
        mediaRecorder = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    // Tell server to process.
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "audio_end" }));
    }

    appendThinkingIndicator();
    setAvatarState("thinking");
}

// ---------------------------------------------------------------------------
// Audio playback
// ---------------------------------------------------------------------------

function playAudio(blob) {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => {
        URL.revokeObjectURL(url);
        setAvatarState("idle");
    };
    audio.play().catch((err) => {
        console.error("Audio playback failed:", err);
    });
}

// ---------------------------------------------------------------------------
// Avatar (2D canvas fallback)
// ---------------------------------------------------------------------------

let canvasCtx = null;
let animFrameId = null;

function initAvatar() {
    if (!avatarCanvas) return;
    avatarCanvas.width = avatarCanvas.offsetWidth * (window.devicePixelRatio || 1);
    avatarCanvas.height = avatarCanvas.offsetHeight * (window.devicePixelRatio || 1);
    canvasCtx = avatarCanvas.getContext("2d");
    animateAvatar();
}

function setAvatarState(state) {
    avatarState = state;
    if (avatarLabel) {
        avatarLabel.textContent = state;
    }
    if (avatarGlow) {
        avatarGlow.classList.toggle(
            "active",
            state === "thinking" || state === "speaking"
        );
    }
}

function updateAvatar(params) {
    avatarParams = params;
    setAvatarState(params.state || "idle");
    removeThinkingIndicator();
}

function animateAvatar() {
    if (!canvasCtx) return;
    const ctx = canvasCtx;
    const w = avatarCanvas.width;
    const h = avatarCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const t = Date.now() / 1000;

    ctx.clearRect(0, 0, w, h);

    // Core orb.
    const baseRadius = Math.min(w, h) * 0.18;
    let radius = baseRadius;
    let coreColor = "#00b4d8";
    let glowAlpha = 0.1;

    if (avatarState === "listening") {
        radius = baseRadius + Math.sin(t * 3) * 4;
        coreColor = "#00e5ff";
        glowAlpha = 0.2;
    } else if (avatarState === "thinking") {
        const glow = avatarParams.glow_intensity || 0.5;
        radius = baseRadius + Math.sin(t * 2) * 8;
        glowAlpha = 0.15 + glow * 0.25;
        coreColor = "#a855f7";
    } else if (avatarState === "speaking") {
        const mouth = avatarParams.mouth_open || 0;
        radius = baseRadius + mouth * 12;
        glowAlpha = 0.2 + mouth * 0.15;
        coreColor = "#22c55e";
    } else if (avatarState === "error") {
        coreColor = "#ef4444";
        glowAlpha = 0.3;
    } else {
        // Idle: gentle breathing.
        radius = baseRadius + Math.sin(t * 0.8) * 2;
    }

    // Outer glow rings.
    for (let i = 3; i >= 1; i--) {
        const r = radius + i * 20;
        const alpha = glowAlpha * (1 - i * 0.25);
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 180, 216, ${alpha})`;
        ctx.fill();
    }

    // Core circle.
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    grad.addColorStop(0, coreColor);
    grad.addColorStop(0.7, coreColor + "88");
    grad.addColorStop(1, coreColor + "00");
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Inner bright core.
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.3, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.globalAlpha = 0.6;
    ctx.fill();
    ctx.globalAlpha = 1.0;

    // Speaking: mouth wave.
    if (avatarState === "speaking") {
        const mouth = avatarParams.mouth_open || 0;
        if (mouth > 0.05) {
            ctx.beginPath();
            ctx.strokeStyle = "#22c55e";
            ctx.lineWidth = 2;
            const waveW = radius * 1.5;
            for (let x = -waveW; x <= waveW; x += 2) {
                const y = Math.sin(x * 0.05 + t * 8) * mouth * 15;
                if (x === -waveW) {
                    ctx.moveTo(cx + x, cy + radius * 0.6 + y);
                } else {
                    ctx.lineTo(cx + x, cy + radius * 0.6 + y);
                }
            }
            ctx.stroke();
        }
    }

    // Blink effect (brief flash).
    if (avatarParams.blink) {
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
        ctx.fill();
        avatarParams.blink = false;
    }

    animFrameId = requestAnimationFrame(animateAvatar);
}

// ---------------------------------------------------------------------------
// Mode switching
// ---------------------------------------------------------------------------

function onModeChange() {
    const mode = modeSelect.value;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "mode", mode: mode }));
    }
}

// ---------------------------------------------------------------------------
// Event bindings
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    connect();
    initAvatar();

    if (sendBtn) {
        sendBtn.addEventListener("click", sendTextQuery);
    }

    if (queryInput) {
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendTextQuery();
            }
        });
    }

    if (micBtn) {
        micBtn.addEventListener("click", toggleRecording);
    }

    if (modeSelect) {
        modeSelect.addEventListener("change", onModeChange);
    }

    // Keep-alive ping every 30s.
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
        }
    }, 30000);

    // Resize avatar canvas on window resize.
    window.addEventListener("resize", () => {
        if (avatarCanvas) {
            avatarCanvas.width =
                avatarCanvas.offsetWidth * (window.devicePixelRatio || 1);
            avatarCanvas.height =
                avatarCanvas.offsetHeight * (window.devicePixelRatio || 1);
        }
    });
});
