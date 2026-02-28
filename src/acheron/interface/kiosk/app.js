/**
 * Nexus Kiosk — Conversational voice interface for Acheron.
 *
 * Designed for continuous, natural interaction like Arthur/Otto:
 *   - Always-on microphone with voice activity detection (VAD)
 *   - Automatic speech segmentation (speak → pause → process)
 *   - Say "stop" or "Nexus stop" to interrupt
 *   - Speaking over Nexus interrupts current response
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

// Audio state
let micStream = null;
let audioCtx = null;
let analyserNode = null;
let processorNode = null;
let micActive = false;

// VAD state
let vadState = "silence"; // "silence" | "speech" | "trailing"
let speechBuffer = [];    // collected Float32 chunks during speech
let silenceFrames = 0;
let speechFrames = 0;
const VAD_SPEECH_THRESHOLD = 0.008;  // RMS threshold to detect speech (sensitive)
const VAD_SPEECH_START_FRAMES = 3;   // consecutive frames above threshold to start
const VAD_SPEECH_END_FRAMES = 70;    // consecutive frames below threshold to end (~3s at 48kHz)
const VAD_FRAME_SIZE = 2048;         // samples per analysis frame

// Playback state
let currentAudio = null;
let isSpeaking = false; // Nexus is speaking
let sttAvailable = false; // server has STT ready
let micEnabledByUser = false; // user has clicked mic at least once

// DOM references (set in DOMContentLoaded)
let responsePanel, queryInput, micBtn, sendBtn, modeSelect, voiceSelect;
let statusDot, statusText, avatarCanvas, avatarLabel, avatarGlow;
let listeningIndicator;

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
        stopMic();
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
            setStatus("active", msg.message);
            // Flag that STT is ready — mic will start on first user click.
            if (msg.stt_available) {
                sttAvailable = true;
            }
            break;

        case "listening":
            setAvatarState("idle");
            setListeningActive(true);
            removeThinkingIndicator();
            break;

        case "pong":
            break;

        case "transcription":
            appendMessage("user", msg.text, "You");
            removeThinkingIndicator();
            appendThinkingIndicator();
            break;

        case "response":
            removeThinkingIndicator();
            renderResponse(msg);
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
// Microphone & VAD — continuous listening
// ---------------------------------------------------------------------------

async function startMic() {
    if (micActive) return;

    try {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        audioCtx = new (window.AudioContext || window.webkitAudioContext)();

        // Critical: resume AudioContext (browsers suspend it until user gesture).
        if (audioCtx.state === "suspended") {
            await audioCtx.resume();
        }

        const source = audioCtx.createMediaStreamSource(micStream);

        // Analyser for VAD (RMS energy detection).
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = VAD_FRAME_SIZE;
        source.connect(analyserNode);

        // ScriptProcessor to capture PCM and run VAD each frame.
        processorNode = audioCtx.createScriptProcessor(VAD_FRAME_SIZE, 1, 1);
        processorNode.onaudioprocess = onAudioProcess;
        source.connect(processorNode);
        processorNode.connect(audioCtx.destination);

        micActive = true;
        vadState = "silence";
        speechBuffer = [];
        silenceFrames = 0;
        speechFrames = 0;

        console.log(`Mic active — sample rate: ${audioCtx.sampleRate}Hz, ` +
            `frame: ${VAD_FRAME_SIZE} samples (${(VAD_FRAME_SIZE / audioCtx.sampleRate * 1000).toFixed(0)}ms), ` +
            `silence timeout: ${(VAD_SPEECH_END_FRAMES * VAD_FRAME_SIZE / audioCtx.sampleRate).toFixed(1)}s`);

        if (micBtn) {
            micBtn.classList.add("active");
            micBtn.title = "Microphone active (click to mute)";
        }
        setListeningActive(true);
        setStatus("active", "Listening — speak naturally");
    } catch (err) {
        console.error("Microphone access denied:", err);
        setStatus("error", "Microphone access denied");
    }
}

function stopMic() {
    micActive = false;
    vadState = "silence";
    speechBuffer = [];

    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }
    if (analyserNode) {
        analyserNode.disconnect();
        analyserNode = null;
    }
    if (audioCtx) {
        audioCtx.close().catch(() => {});
        audioCtx = null;
    }
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
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

/**
 * ScriptProcessor callback — runs VAD on each audio frame.
 *
 * VAD state machine:
 *   silence  →  (RMS > threshold for N frames)  →  speech
 *   speech   →  (RMS < threshold for M frames)   →  silence  (+ send segment)
 */
function onAudioProcess(e) {
    if (!micActive) return;

    const input = e.inputBuffer.getChannelData(0);
    const rms = computeRMS(input);

    if (vadState === "silence") {
        if (rms > VAD_SPEECH_THRESHOLD) {
            speechFrames++;
            if (speechFrames >= VAD_SPEECH_START_FRAMES) {
                vadState = "speech";
                silenceFrames = 0;
                speechBuffer = [];
                setListeningActive(false);
                setAvatarState("listening");
                setStatus("active", "Hearing you...");

                // If Nexus is speaking, interrupt.
                if (isSpeaking) {
                    stopAudio();
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: "interrupt" }));
                    }
                }
            }
        } else {
            speechFrames = 0;
        }
        // Buffer a few pre-speech frames so the start of the word isn't clipped.
        if (speechFrames > 0) {
            speechBuffer.push(new Float32Array(input));
        }
    } else if (vadState === "speech") {
        speechBuffer.push(new Float32Array(input));

        if (rms < VAD_SPEECH_THRESHOLD) {
            silenceFrames++;
            // Show countdown so user knows when it'll send.
            const framesLeft = VAD_SPEECH_END_FRAMES - silenceFrames;
            const secsLeft = (framesLeft * VAD_FRAME_SIZE / (audioCtx ? audioCtx.sampleRate : 48000)).toFixed(1);
            if (silenceFrames % 10 === 0 && framesLeft > 0) {
                setStatus("active", `Pause detected... processing in ${secsLeft}s`);
            }
            if (silenceFrames >= VAD_SPEECH_END_FRAMES) {
                // Speech ended — send the segment.
                vadState = "silence";
                speechFrames = 0;
                silenceFrames = 0;
                setStatus("active", "Processing your question...");
                sendSpeechSegment();
            }
        } else {
            silenceFrames = 0;
            setStatus("active", "Hearing you...");
        }
    }
}

function computeRMS(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
        sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
}

/**
 * Package the speech buffer as a WAV and send to server.
 */
function sendSpeechSegment() {
    if (!speechBuffer.length || !ws || ws.readyState !== WebSocket.OPEN) {
        speechBuffer = [];
        setListeningActive(true);
        return;
    }

    // Merge all float32 chunks.
    let totalLength = 0;
    for (const chunk of speechBuffer) totalLength += chunk.length;
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of speechBuffer) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }
    speechBuffer = [];

    // Resample to 16kHz for Whisper.
    const nativeSR = audioCtx ? audioCtx.sampleRate : 48000;
    const resampled = resample(merged, nativeSR, 16000);

    // Convert float32 to int16 PCM.
    const pcm = float32ToInt16(resampled);

    // Wrap as WAV.
    const wav = createWAV(pcm, 16000);

    // Send: first a JSON header, then the binary WAV.
    ws.send(JSON.stringify({ type: "speech_segment" }));
    ws.send(wav);

    appendThinkingIndicator();
    setAvatarState("thinking");
}

/**
 * Simple linear interpolation resampler.
 */
function resample(input, fromRate, toRate) {
    if (fromRate === toRate) return input;
    const ratio = fromRate / toRate;
    const outLength = Math.round(input.length / ratio);
    const output = new Float32Array(outLength);
    for (let i = 0; i < outLength; i++) {
        const srcIdx = i * ratio;
        const low = Math.floor(srcIdx);
        const high = Math.min(low + 1, input.length - 1);
        const frac = srcIdx - low;
        output[i] = input[low] * (1 - frac) + input[high] * frac;
    }
    return output;
}

function float32ToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
}

/**
 * Create a WAV file from int16 PCM data.
 */
function createWAV(pcm, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcm.length * (bitsPerSample / 8);
    const headerSize = 44;

    const buffer = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, "WAVE");

    // fmt sub-chunk
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);          // sub-chunk size
    view.setUint16(20, 1, true);           // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // data sub-chunk
    writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    // PCM data
    const pcmView = new Int16Array(buffer, headerSize);
    pcmView.set(pcm);

    return buffer;
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
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

    // Listening: subtle mic input visualization.
    if (avatarState === "listening" && analyserNode) {
        const data = new Uint8Array(analyserNode.fftSize);
        analyserNode.getByteTimeDomainData(data);
        ctx.beginPath();
        ctx.strokeStyle = "rgba(0, 229, 255, 0.4)";
        ctx.lineWidth = 1.5;
        const sliceWidth = (radius * 2) / data.length;
        let xPos = cx - radius;
        for (let i = 0; i < data.length; i++) {
            const v = data[i] / 128.0;
            const y = cy + radius * 0.8 + (v - 1) * 30;
            if (i === 0) ctx.moveTo(xPos, y);
            else ctx.lineTo(xPos, y);
            xPos += sliceWidth;
        }
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
