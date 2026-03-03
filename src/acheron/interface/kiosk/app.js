/**
 * Nexus Kiosk — Conversational voice interface for Nexus.
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
let awaitingServerAudio = false; // expecting audio from server
let serverAudioTimer = null;

// Audio queue — sentence-sized WAV chunks arrive progressively.
const audioQueue = [];
let audioPlaying = false;

// Streaming text state — accumulates response_chunk fragments.
let streamingDiv = null;
let streamingBody = null;
let streamingText = "";

// Emotion detection (camera)
let cameraStream = null;
let cameraVideo = null;
let emotionCanvas = null;
let emotionCtx = null;
let emotionTimer = null;
let cameraActive = false;
let currentEmotion = null; // latest EmotionalState from server
const EMOTION_CAPTURE_MS = 500; // capture a frame every 500ms (2 FPS)

// DOM references (set in DOMContentLoaded)
let responsePanel, queryInput, micBtn, sendBtn, modeSelect, voiceSelect;
let statusDot, statusText, avatarCanvas, avatarLabel, avatarGlow;
let listeningIndicator, interimDisplay, cameraBtn, emotionBadge;

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

function capitalizeFirst(str) {
    if (!str) return str;
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function onSpeechFinalized(text) {
    clearSilenceTimer();
    hideInterimTranscript();

    if (!text) return;

    // Capitalize first letter like a normal sentence.
    text = capitalizeFirst(text);

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

    // Check for voice mode commands:
    //   "Nexus analyze ...", "Analyze ...", "Analyze that"
    //   "Nexus query ...",   "Query ...",   "Query that"
    //   "Nexus discover ...", "Discover ...", "Discover that"
    const modeResult = extractVoiceMode(text);
    if (modeResult.modeChanged) {
        // Switch the mode dropdown + notify server.
        if (modeSelect) {
            modeSelect.value = modeResult.mode;
            onModeChange();
        }
    }

    // Use the query (with mode prefix stripped) or the original text.
    const query = modeResult.query || text;

    // If the voice command was just a mode switch with no query
    // (e.g., "Nexus analyze" by itself), confirm and return.
    if (modeResult.modeChanged && !modeResult.query) {
        appendMessage("user", text, "You");
        appendMessage("system",
            "Switched to " + modeResult.mode + " mode.",
            "NEXUS"
        );
        setAvatarState("idle");
        return;
    }

    // Auto-send — Nexus should feel like a conversation, not a form.
    if (ws && ws.readyState === WebSocket.OPEN) {
        stopAudio();
        appendMessage("user", query, "You");
        appendThinkingIndicator();
        setAvatarState("thinking");
        ws.send(JSON.stringify({ type: "text", query: query }));
        // Also update the input box so the user can see what was sent.
        if (queryInput) queryInput.value = "";
    }
}

function extractVoiceMode(text) {
    const lower = text.toLowerCase().trim();

    // Patterns:  "nexus analyze ...", "analyze ...", "analyze that/this"
    const modes = [
        { words: ["nexus analyze", "nexus analyse"], mode: "analyze" },
        { words: ["analyze", "analyse"],             mode: "analyze" },
        { words: ["nexus discover"],                 mode: "discover" },
        { words: ["discover"],                       mode: "discover" },
        { words: ["nexus query"],                    mode: "query" },
    ];

    for (const m of modes) {
        for (const prefix of m.words) {
            if (lower.startsWith(prefix)) {
                let rest = text.slice(prefix.length).trim();

                // "Analyze that" / "Analyze this" = mode switch only.
                const restLower = rest.toLowerCase();
                if (restLower === "that" || restLower === "this" ||
                    restLower === "it" || restLower === "mode" ||
                    restLower === "") {
                    return { modeChanged: true, mode: m.mode, query: "" };
                }

                // Strip leading filler: "Analyze, what is ..." → "what is ..."
                rest = rest.replace(/^[,.:;]\s*/, "");

                return { modeChanged: true, mode: m.mode, query: capitalizeFirst(rest) };
            }
        }
    }

    return { modeChanged: false, mode: null, query: "" };
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
        // Auto-activate mic — Nexus should be ready to listen
        // the moment you open the page (like talking to a person).
        // On reconnect, re-enable if it was on.
        if (!micActive) {
            micEnabledByUser = true;
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

        case "response_chunk":
            // Progressive text streaming — append to a live message div.
            removeThinkingIndicator();
            streamingText += msg.text;
            if (!streamingDiv) {
                streamingDiv = document.createElement("div");
                streamingDiv.className = "message system streaming";
                streamingBody = document.createElement("div");
                streamingDiv.appendChild(streamingBody);
                responsePanel.appendChild(streamingDiv);
            }
            streamingBody.innerHTML = formatMarkdown(streamingText);
            responsePanel.scrollTop = responsePanel.scrollHeight;
            break;

        case "response":
            removeThinkingIndicator();
            // Stop any audio still playing from a previous response
            // before rendering the new one.
            stopAudio();
            // Replace streaming preview with the final rendered response.
            if (streamingDiv) {
                streamingDiv.remove();
                streamingDiv = null;
                streamingBody = null;
                streamingText = "";
            }
            renderResponse(msg);
            // Wait for server audio (Edge TTS / ElevenLabs).
            // If no audio arrives within 5s, just go idle (no robot voice).
            awaitingServerAudio = true;
            clearTimeout(serverAudioTimer);
            serverAudioTimer = setTimeout(() => {
                if (awaitingServerAudio) {
                    awaitingServerAudio = false;
                    setAvatarState("idle");
                }
            }, 5000);
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

        case "emotion":
            updateEmotionDisplay(msg.state);
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
    const raw = queryInput.value.trim();
    if (!raw || !ws || ws.readyState !== WebSocket.OPEN) return;

    // Stop any audio from the previous response before sending a new query.
    stopAudio();

    let query = capitalizeFirst(raw);

    // Check for mode commands in typed input too.
    const modeResult = extractVoiceMode(query);
    if (modeResult.modeChanged) {
        if (modeSelect) {
            modeSelect.value = modeResult.mode;
            onModeChange();
        }
        if (modeResult.query) {
            query = modeResult.query;
        } else {
            // Just a mode switch, no query.
            appendMessage("user", raw, "You");
            appendMessage("system",
                "Switched to " + modeResult.mode + " mode.",
                "NEXUS"
            );
            queryInput.value = "";
            return;
        }
    }

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
    // Server audio arrived — cancel browser TTS fallback timer.
    awaitingServerAudio = false;
    clearTimeout(serverAudioTimer);

    // Queue this chunk for sequential playback.
    audioQueue.push(blob);
    if (!audioPlaying) {
        _playNext();
    }
}

function _playNext() {
    if (audioQueue.length === 0) {
        audioPlaying = false;
        isSpeaking = false;
        currentAudio = null;
        setAvatarState("idle");
        setListeningActive(true);
        return;
    }
    audioPlaying = true;
    isSpeaking = true;
    setAvatarState("speaking");

    const blob = audioQueue.shift();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentAudio = audio;

    // Guard against double _playNext() calls — both onerror and
    // play().catch() can fire for the same element on failure.
    let advanced = false;
    function advance() {
        if (advanced) return;
        advanced = true;
        URL.revokeObjectURL(url);
        // Only advance if this audio is still the current one
        // (stopAudio may have reset the state already).
        if (currentAudio === audio) {
            _playNext();
        }
    }

    audio.onended = advance;
    audio.onerror = advance;

    audio.play().catch((err) => {
        console.error("Audio playback failed:", err);
        advance();
    });
}

function stopAudio() {
    // Flush entire queue and stop current playback.
    audioQueue.length = 0;
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    audioPlaying = false;
    isSpeaking = false;
}

// TTS is handled server-side (Edge TTS / ElevenLabs).
// The server sends audio as binary WebSocket frames → playAudio().

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
// Camera — emotion detection via server-side face analysis
// ---------------------------------------------------------------------------

function startCamera() {
    if (cameraActive) return;

    cameraVideo = document.getElementById("emotion-video");
    if (!cameraVideo) return;

    // Off-screen canvas for frame capture.
    emotionCanvas = document.createElement("canvas");
    emotionCanvas.width = 320;
    emotionCanvas.height = 240;
    emotionCtx = emotionCanvas.getContext("2d");

    navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
        audio: false,
    }).then((stream) => {
        cameraStream = stream;
        cameraVideo.srcObject = stream;
        cameraVideo.play();
        cameraActive = true;
        cameraVideo.classList.add("active");

        if (cameraBtn) {
            cameraBtn.classList.add("active");
            cameraBtn.title = "Camera active (click to disable)";
        }

        // Start periodic frame capture.
        emotionTimer = setInterval(captureAndSendFrame, EMOTION_CAPTURE_MS);
        console.log("Camera started for emotion detection");
    }).catch((err) => {
        console.warn("Camera not available:", err.message);
        if (cameraBtn) {
            cameraBtn.title = "Camera unavailable: " + err.message;
        }
    });
}

function stopCamera() {
    cameraActive = false;

    if (emotionTimer) {
        clearInterval(emotionTimer);
        emotionTimer = null;
    }
    if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
        cameraStream = null;
    }
    if (cameraVideo) {
        cameraVideo.srcObject = null;
        cameraVideo.classList.remove("active");
    }
    if (cameraBtn) {
        cameraBtn.classList.remove("active");
        cameraBtn.title = "Enable camera for emotion detection";
    }
}

function toggleCamera() {
    if (cameraActive) {
        stopCamera();
    } else {
        startCamera();
    }
}

function captureAndSendFrame() {
    if (!cameraVideo || !cameraStream || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (cameraVideo.readyState < 2) return; // video not ready

    emotionCtx.drawImage(cameraVideo, 0, 0, 320, 240);
    const dataUrl = emotionCanvas.toDataURL("image/jpeg", 0.5);
    const base64 = dataUrl.split(",")[1];

    ws.send(JSON.stringify({
        type: "video_frame",
        data: base64,
    }));
}

// ---------------------------------------------------------------------------
// Emotion display — show detected emotion near avatar
// ---------------------------------------------------------------------------

function updateEmotionDisplay(state) {
    currentEmotion = state;
    if (!emotionBadge) return;

    if (!state || state.confidence < 0.3) {
        emotionBadge.textContent = "";
        emotionBadge.className = "emotion-badge";
        return;
    }

    const emoji = _emotionEmoji(state.emotion);
    emotionBadge.textContent = emoji + " " + state.emotion;
    emotionBadge.className = "emotion-badge visible";

    // Colour the badge by valence.
    if (state.valence > 0.2) {
        emotionBadge.style.borderColor = "var(--evidence-green)";
        emotionBadge.style.color = "var(--evidence-green)";
    } else if (state.valence < -0.3) {
        emotionBadge.style.borderColor = "var(--speculation-red)";
        emotionBadge.style.color = "var(--speculation-red)";
    } else {
        emotionBadge.style.borderColor = "var(--accent-cyan)";
        emotionBadge.style.color = "var(--accent-cyan)";
    }
}

function _emotionEmoji(emotion) {
    const map = {
        happy: "\u{1F60A}",
        sad: "\u{1F614}",
        angry: "\u{1F620}",
        fearful: "\u{1F628}",
        disgusted: "\u{1F616}",
        surprised: "\u{1F632}",
        confused: "\u{1F914}",
        frustrated: "\u{1F624}",
        neutral: "\u{1F610}",
    };
    return map[emotion] || "\u{1F610}";
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
    cameraBtn = document.getElementById("camera-btn");
    emotionBadge = document.getElementById("emotion-badge");

    // Camera button.
    if (cameraBtn) cameraBtn.addEventListener("click", toggleCamera);

    // Check browser support for voice INPUT (mic).
    if (!SpeechRecognition) {
        appendMessage("system",
            "Voice input (microphone) requires Chrome or Edge. " +
            "You can type your questions below. " +
            "Voice output will still work in this browser.",
            "NEXUS"
        );
        // Disable mic button but keep it visible — show tooltip.
        if (micBtn) {
            micBtn.disabled = true;
            micBtn.title = "Voice input requires Chrome or Edge";
            micBtn.style.opacity = "0.4";
        }
        if (queryInput) queryInput.focus();
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
