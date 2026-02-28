/**
 * Nexus Kiosk — Conversational voice interface for Acheron.
 *
 * Uses the browser's built-in SpeechRecognition API for real-time
 * speech-to-text (like ChatGPT voice mode).  No server-side STT needed.
 *
 *   - Click mic → browser starts listening continuously
 *   - Words appear in real-time as you speak (interim results)
 *   - Pause speaking → final transcript sent to server → response
 *   - Say "stop" / "Nexus stop" to interrupt
 *   - Speak over Nexus to interrupt playback
 *   - Two voice profiles: male (soft) and female (warm)
 *   - Text input fallback always available
 */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ws = null;
let avatarState = "idle";
let avatarParams = {};

// Speech recognition
let recognition = null;
let micActive = false;
let micEnabledByUser = false;
let isRecognizing = false;

// Transcript assembly
let interimTranscript = "";
let finalTranscript = "";
let silenceTimer = null;
const SILENCE_TIMEOUT_MS = 2000; // 2s pause → send query

// Playback state
let currentAudio = null;
let isSpeaking = false; // Nexus is speaking

// DOM references (set in DOMContentLoaded)
let responsePanel, queryInput, micBtn, sendBtn, modeSelect, voiceSelect;
let statusDot, statusText, avatarCanvas, avatarLabel, avatarGlow;
let listeningIndicator, interimDisplay;

// ---------------------------------------------------------------------------
// Browser SpeechRecognition setup
// ---------------------------------------------------------------------------

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

function initSpeechRecognition() {
    if (!SpeechRecognition) {
        console.warn("SpeechRecognition API not supported in this browser");
        setStatus("error", "Voice not supported — use Chrome, Edge, or Safari");
        return false;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        isRecognizing = true;
        console.log("SpeechRecognition started");
    };

    recognition.onresult = (event) => {
        interimTranscript = "";
        finalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        // Show interim text in real-time (like ChatGPT).
        showInterimTranscript(interimTranscript || finalTranscript);
        setAvatarState("listening");
        setStatus("active", "Hearing you...");

        // If Nexus is currently speaking and user talks, interrupt.
        if (isSpeaking && (interimTranscript.length > 3 || finalTranscript.length > 0)) {
            stopAudio();
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "interrupt" }));
            }
        }

        // Reset silence timer — user is still talking.
        clearSilenceTimer();

        if (finalTranscript) {
            // Browser gave us a final result — process it.
            onSpeechFinalized(finalTranscript.trim());
        } else {
            // Still interim — start silence timer.
            startSilenceTimer();
        }
    };

    recognition.onerror = (event) => {
        console.error("SpeechRecognition error:", event.error);

        if (event.error === "not-allowed") {
            setStatus("error", "Microphone access denied");
            stopMic();
            return;
        }

        if (event.error === "no-speech") {
            // Normal — just means silence, restart.
            return;
        }

        if (event.error === "network") {
            setStatus("error", "Speech recognition network error — retrying...");
        }

        // For aborted/other errors, will auto-restart via onend.
    };

    recognition.onend = () => {
        isRecognizing = false;
        console.log("SpeechRecognition ended");

        // Auto-restart if mic should be active (continuous mode).
        if (micActive && micEnabledByUser) {
            try {
                recognition.start();
            } catch (e) {
                console.warn("Failed to restart recognition:", e);
                setTimeout(() => {
                    if (micActive && micEnabledByUser) {
                        try { recognition.start(); } catch (e2) { /* give up */ }
                    }
                }, 500);
            }
        }
    };

    return true;
}

// ---------------------------------------------------------------------------
// Silence timer — sends the query after user pauses
// ---------------------------------------------------------------------------

function startSilenceTimer() {
    clearSilenceTimer();
    silenceTimer = setTimeout(() => {
        // User stopped talking — send whatever interim text we have.
        if (interimTranscript.trim()) {
            onSpeechFinalized(interimTranscript.trim());
            interimTranscript = "";
        }
    }, SILENCE_TIMEOUT_MS);
}

function clearSilenceTimer() {
    if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
    }
}

// ---------------------------------------------------------------------------
// Speech finalized — send to server
// ---------------------------------------------------------------------------

function onSpeechFinalized(text) {
    clearSilenceTimer();
    hideInterimTranscript();

    if (!text) return;

    // Check for interrupt commands locally (instant response).
    if (isInterruptCommand(text)) {
        stopAudio();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "interrupt" }));
        }
        appendMessage("user", text, "You");
        appendMessage("system", "Understood — stopped.", "NEXUS");
        setAvatarState("idle");
        setListeningActive(true);
        return;
    }

    // Send as text query.
    appendMessage("user", text, "You");
    appendThinkingIndicator();
    setAvatarState("thinking");

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "text", query: text }));
    }
}

function isInterruptCommand(text) {
    const cleaned = text.trim().toLowerCase();
    const words = cleaned.split(/\s+/);
    if (words.length > 4) return false;
    return /\b(stop|nexus stop|shut up|be quiet|quiet|enough|cancel|hold on|wait|pause|never\s?mind|nevermind)\b/i.test(cleaned);
}

// ---------------------------------------------------------------------------
// Interim transcript display
// ---------------------------------------------------------------------------

function showInterimTranscript(text) {
    if (!interimDisplay) return;
    interimDisplay.textContent = text;
    interimDisplay.classList.add("visible");
}

function hideInterimTranscript() {
    if (!interimDisplay) return;
    interimDisplay.textContent = "";
    interimDisplay.classList.remove("visible");
}

// ---------------------------------------------------------------------------
// Microphone control
// ---------------------------------------------------------------------------

function startMic() {
    if (micActive) return;
    if (!recognition && !initSpeechRecognition()) return;

    try {
        recognition.start();
        micActive = true;

        if (micBtn) {
            micBtn.classList.add("active");
            micBtn.title = "Microphone active (click to mute)";
        }
        setListeningActive(true);
        setStatus("active", "Listening — speak naturally");
        console.log("Mic activated — browser SpeechRecognition");
    } catch (err) {
        console.error("Failed to start speech recognition:", err);
        setStatus("error", "Could not start voice recognition");
    }
}

function stopMic() {
    micActive = false;
    clearSilenceTimer();
    hideInterimTranscript();

    if (recognition && isRecognizing) {
        try {
            recognition.stop();
        } catch (e) {
            // Already stopped.
        }
    }

    if (micBtn) {
        micBtn.classList.remove("active");
        micBtn.title = "Microphone muted (click to unmute)";
    }
    setListeningActive(false);
}

function toggleMic() {
    if (micActive) {
        stopMic();
        micEnabledByUser = false;
        setStatus("active", "Microphone muted");
    } else {
        micEnabledByUser = true;
        startMic();
    }
}

// ---------------------------------------------------------------------------
// WebSocket connection
// ---------------------------------------------------------------------------

function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws`;
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        setStatus("active", "Connected");
        // Re-enable mic on reconnect if user had it on.
        if (micEnabledByUser && !micActive) {
            startMic();
        }
    };

    ws.onclose = () => {
        setStatus("inactive", "Disconnected — reconnecting...");
        setTimeout(connect, 3000);
    };

    ws.onerror = () => {
        setStatus("error", "Connection error");
    };

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            // Binary: TTS audio response.
            playAudio(new Blob([event.data], { type: "audio/wav" }));
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
            if (!micActive) {
                setStatus("active", msg.message);
            }
            break;

        case "listening":
            if (!micActive) {
                setAvatarState("idle");
            }
            setListeningActive(true);
            removeThinkingIndicator();
            break;

        case "pong":
            break;

        case "transcription":
            // Server-side transcription (fallback) — show it.
            appendMessage("user", msg.text, "You");
            removeThinkingIndicator();
            appendThinkingIndicator();
            break;

        case "response":
            removeThinkingIndicator();
            renderResponse(msg);
            setAvatarState("idle");
            break;

        case "interrupted":
            removeThinkingIndicator();
            stopAudio();
            setAvatarState("idle");
            appendMessage("system", "Interrupted.", "NEXUS");
            break;

        case "avatar":
            updateAvatar(msg);
            break;

        case "error":
            removeThinkingIndicator();
            appendMessage("error", msg.message, "Error");
            setAvatarState("idle");
            break;
    }
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function setStatus(state, text) {
    if (statusDot) statusDot.className = `status-dot ${state}`;
    if (statusText) statusText.textContent = text;
}

function setListeningActive(active) {
    if (listeningIndicator) {
        listeningIndicator.classList.toggle("active", active && micActive);
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

    if (msg.mode) {
        const label = document.createElement("span");
        label.className = "label mode";
        label.textContent = msg.mode.toUpperCase();
        div.appendChild(label);
    }

    const body = document.createElement("div");
    body.innerHTML = formatMarkdown(msg.answer || "No response.");
    div.appendChild(body);

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
}

function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, "<pre><code>$2</code></pre>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
    html = html.replace(/\n/g, "<br>");
    return html;
}

function escapeHtml(text) {
    const el = document.createElement("div");
    el.textContent = text;
    return el.innerHTML;
}

function appendThinkingIndicator() {
    removeThinkingIndicator();
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
// Text input
// ---------------------------------------------------------------------------

function sendTextQuery() {
    const query = queryInput.value.trim();
    if (!query || !ws || ws.readyState !== WebSocket.OPEN) return;

    appendMessage("user", query, "You");
    appendThinkingIndicator();
    setAvatarState("thinking");

    ws.send(JSON.stringify({ type: "text", query: query }));
    queryInput.value = "";
}

// ---------------------------------------------------------------------------
// Audio playback (interruptible)
// ---------------------------------------------------------------------------

function playAudio(blob) {
    stopAudio(); // Stop any previous playback.

    const url = URL.createObjectURL(blob);
    currentAudio = new Audio(url);
    isSpeaking = true;
    setAvatarState("speaking");

    currentAudio.onended = () => {
        URL.revokeObjectURL(url);
        isSpeaking = false;
        currentAudio = null;
        setAvatarState("idle");
        setListeningActive(true);
    };

    currentAudio.onerror = () => {
        URL.revokeObjectURL(url);
        isSpeaking = false;
        currentAudio = null;
        setAvatarState("idle");
    };

    currentAudio.play().catch((err) => {
        console.error("Audio playback failed:", err);
        isSpeaking = false;
        currentAudio = null;
    });
}

function stopAudio() {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    isSpeaking = false;
}

// ---------------------------------------------------------------------------
// Avatar (2D canvas)
// ---------------------------------------------------------------------------

let canvasCtx = null;

function initAvatar() {
    if (!avatarCanvas) return;
    avatarCanvas.width = avatarCanvas.offsetWidth * (window.devicePixelRatio || 1);
    avatarCanvas.height = avatarCanvas.offsetHeight * (window.devicePixelRatio || 1);
    canvasCtx = avatarCanvas.getContext("2d");
    animateAvatar();
}

function setAvatarState(state) {
    avatarState = state;
    if (avatarLabel) avatarLabel.textContent = state;
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

    // Speaking: sound wave.
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

    // Listening: gentle pulse effect.
    if (avatarState === "listening") {
        const pulseRadius = radius + 10 + Math.sin(t * 4) * 5;
        ctx.beginPath();
        ctx.arc(cx, cy, pulseRadius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 229, 255, 0.3)";
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Blink effect.
    if (avatarParams.blink) {
        ctx.beginPath();
        ctx.arc(cx, cy, radius * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
        ctx.fill();
        avatarParams.blink = false;
    }

    requestAnimationFrame(animateAvatar);
}

// ---------------------------------------------------------------------------
// Mode & voice switching
// ---------------------------------------------------------------------------

function onModeChange() {
    const mode = modeSelect.value;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "mode", mode: mode }));
    }
}

function onVoiceChange() {
    const profile = voiceSelect.value;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "voice", profile: profile }));
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    // Grab DOM refs.
    responsePanel = document.getElementById("response-panel");
    queryInput = document.getElementById("query-input");
    micBtn = document.getElementById("mic-btn");
    sendBtn = document.getElementById("send-btn");
    modeSelect = document.getElementById("mode-select");
    voiceSelect = document.getElementById("voice-select");
    statusDot = document.getElementById("status-dot");
    statusText = document.getElementById("status-text");
    avatarCanvas = document.getElementById("avatar-canvas");
    avatarLabel = document.getElementById("avatar-state-label");
    avatarGlow = document.getElementById("avatar-glow");
    listeningIndicator = document.getElementById("listening-indicator");
    interimDisplay = document.getElementById("interim-transcript");

    // Check browser support.
    if (!SpeechRecognition) {
        appendMessage("error",
            "Your browser does not support voice recognition. " +
            "Please use Chrome, Edge, or Safari. You can still type queries below.",
            "Error"
        );
    }

    connect();
    initAvatar();

    if (sendBtn) sendBtn.addEventListener("click", sendTextQuery);

    if (queryInput) {
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendTextQuery();
            }
        });
    }

    if (micBtn) micBtn.addEventListener("click", toggleMic);
    if (modeSelect) modeSelect.addEventListener("change", onModeChange);
    if (voiceSelect) voiceSelect.addEventListener("change", onVoiceChange);

    // Keep-alive ping.
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
        }
    }, 30000);

    // Resize canvas.
    window.addEventListener("resize", () => {
        if (avatarCanvas) {
            avatarCanvas.width = avatarCanvas.offsetWidth * (window.devicePixelRatio || 1);
            avatarCanvas.height = avatarCanvas.offsetHeight * (window.devicePixelRatio || 1);
        }
    });
});
