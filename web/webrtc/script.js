const socket = io();

// UI Elements
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const callBtn = document.getElementById('callBtn');
const cameraBtn = document.getElementById('cameraBtn');
const micBtn = document.getElementById('micBtn');
const statusBadge = document.getElementById('connectionStatus');
const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');

// State
let localStream;
let peerConnection;
let isCreated = false;

// --- TTS Sign-Language Preference (set on pre-call modal in landing.html) ---
let ttsSignLang = localStorage.getItem('ttsSignLang') || 'en'; // 'en', 'hi', 'ml'
const ttsAutoEnabled = localStorage.getItem('ttsAutoEnabled') === 'true';

// --- Accessibility Feature Flags ---
const accHearing = localStorage.getItem('acc_hearing') === 'true';
const accSpeech = localStorage.getItem('acc_speech') === 'true';


// Config
const iceServers = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
    ]
};

// --- Helper Functions ---

function showStatus(text, type = 'neutral') {
    statusBadge.innerText = text;
    statusBadge.className = 'status-badge'; // Reset
    if (type === 'success') statusBadge.classList.add('connected');
    console.log(`Status: ${text}`);
}

function showError(msg) {
    errorMessage.innerHTML = msg;
    errorModal.classList.add('show');
}

// --- Button Events ---

async function toggleCall() {
    // Prevent double clicking
    if (callBtn.disabled) return;

    // If not connected, start process
    if (!localStream) {
        callBtn.innerHTML = '<ion-icon name="ellipsis-horizontal-outline"></ion-icon> <span>Connecting...</span>';
        callBtn.classList.add('has-text');
        callBtn.disabled = true;

        await startLocalStream();

        if (localStream) {
            callBtn.innerHTML = '<ion-icon name="cellular-outline"></ion-icon> <span>Waiting...</span>';
            socket.emit('ready');
            isCreated = true;
        } else {
            callBtn.innerHTML = '<ion-icon name="call-outline"></ion-icon>';
            callBtn.classList.remove('has-text');
            callBtn.disabled = false;
        }
    } else {
        // Disconnect logic (End Call)
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }
        localVideo.srcObject = null;
        remoteVideo.srcObject = null;

        // Reset button to Join state (Green)
        callBtn.innerHTML = '<ion-icon name="call-outline"></ion-icon>';
        callBtn.className = 'control-btn start-call';
        callBtn.title = 'Join Call';

        // Reset Media Buttons
        updateMediaButton(cameraBtn, false, 'videocam', 'videocam-off');
        updateMediaButton(micBtn, false, 'mic', 'mic-off');

        // Turn off translation if it was running
        if (translationEnabled) {
            toggleTranslation();
        }

        isCreated = false;
        showStatus('Disconnected');
    }
}
window.toggleCall = toggleCall;

function toggleCamera() {
    if (localStream) {
        const videoTrack = localStream.getVideoTracks()[0];
        videoTrack.enabled = !videoTrack.enabled;
        updateMediaButton(cameraBtn, videoTrack.enabled, 'videocam', 'videocam-off');
    }
}
window.toggleCamera = toggleCamera;

function toggleMic() {
    if (localStream) {
        const audioTrack = localStream.getAudioTracks()[0];
        audioTrack.enabled = !audioTrack.enabled;
        updateMediaButton(micBtn, audioTrack.enabled, 'mic', 'mic-off');
    }
}
window.toggleMic = toggleMic;

function updateMediaButton(btn, isEnabled, iconOn, iconOff) {
    const icon = btn.querySelector('ion-icon');
    if (isEnabled) {
        btn.classList.add('active');
        btn.classList.remove('video-off');
        icon.setAttribute('name', iconOn);
    } else {
        btn.classList.remove('active');
        btn.classList.add('video-off');
        icon.setAttribute('name', iconOff);
    }
}

// --- Media & WebRTC ---

async function startLocalStream() {
    try {
        // Request HD format for video calling
        localStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: true
        });
        localVideo.srcObject = localStream;
        showStatus('Camera Ready - Joining...');

        // Sync button states so they visually show up as active (blue)
        updateMediaButton(cameraBtn, true, 'videocam', 'videocam-off');
        updateMediaButton(micBtn, true, 'mic', 'mic-off');

        // Auto-enable "Live Signs" if Hearing Impaired profile is active
        if (accHearing && !translationEnabled) {
            // Add a small delay so the video element has time to render
            setTimeout(() => {
                toggleTranslation();
            }, 500);
        }
    } catch (err) {
        console.error('Error accessing media devices:', err);
        let msg = `<b>Camera access failed.</b><br>${err.message}`;
        if (window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            msg += `<br><br><b>Reason:</b> Browsers block camera access on insecure (HTTP) connections.`;
        }
        showError(msg);
        return null;
    }
}

function createPeerConnection() {
    if (peerConnection) return; // Don't overwrite existing

    console.log("Creating RTCPeerConnection");
    peerConnection = new RTCPeerConnection(iceServers);

    if (localStream) {
        localStream.getTracks().forEach(track => {
            peerConnection.addTrack(track, localStream);
        });
    }

    peerConnection.ontrack = event => {
        console.log("Track received");
        remoteVideo.srcObject = event.streams[0];
        showStatus('Connected', 'success');

        // Switch button to End Call state (Red)
        callBtn.innerHTML = '<ion-icon name="call"></ion-icon>';
        callBtn.className = 'control-btn end-call active';
        callBtn.title = 'End Call';
        callBtn.disabled = false;
    };

    peerConnection.onicecandidate = event => {
        if (event.candidate) {
            socket.emit('message', { type: 'candidate', candidate: event.candidate });
        }
    };

    peerConnection.onconnectionstatechange = () => {
        console.log(`Connection state: ${peerConnection.connectionState}`);
        if (peerConnection.connectionState === 'connected') {
            showStatus('Connected', 'success');
        } else if (peerConnection.connectionState === 'disconnected') {
            showStatus('Disconnected');
        } else if (peerConnection.connectionState === 'failed') {
            showStatus('Failed', 'error');
        }
    };
}

// --- Socket Signaling ---

// Rule: If we receive 'ready', WE initiate the call.
socket.on('ready', async () => {
    if (isCreated) {
        // We are already in the room and ready. A new person just joined and is ready.
        console.log("Another peer is ready. Initiating call...");
        createPeerConnection();
        // Create Offer
        try {
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);
            socket.emit('message', { type: 'offer', sdp: offer });
        } catch (err) {
            console.error("Offer error", err);
        }
    }
});

socket.on('message', async (data) => {
    if (data.type === 'offer') {
        console.log("Received Offer");
        if (!isCreated) {
            // We haven't clicked 'Join' yet, but we are receiving a call.
            // We can't answer properly without local stream (usually).
            // But let's try to answer anyway to establish connection?
            // No, let's wait or show incoming. 
            // For this app: Expect users to click Join.
            console.log("Ignored offer because not ready.");
            return;
        }
        createPeerConnection();
        try {
            await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
            const answer = await peerConnection.createAnswer();
            await peerConnection.setLocalDescription(answer);
            socket.emit('message', { type: 'answer', sdp: answer });
        } catch (err) {
            console.error("Answer error", err);
        }
    }
    else if (data.type === 'answer') {
        console.log("Received Answer");
        if (peerConnection) {
            await peerConnection.setRemoteDescription(new RTCSessionDescription(data.sdp));
        }
    }
    else if (data.type === 'candidate') {
        if (peerConnection) {
            try {
                await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
            } catch (e) {
                console.error("Candidate error", e);
            }
        }
    }
});

// ================================================================
// PHASED SIGN DETECTION — 4-STATE MACHINE (BATCH MODE)
// ================================================================
// State 1: SETUP     (3s, one-time)  → "Setting up camera..." — NO frames captured
// State 2: LISTENING (4s, countdown) → Frames captured LOCALLY into array
// State 3: PROCESSING                → Batch sent to backend, wait for result
// State 4: SHOWING   (2s)            → Display result, then discard & loop to LISTENING
// ================================================================

let translationEnabled = false;

// State tracking
const PHASE = { IDLE: 'idle', SETUP: 'setup', LISTENING: 'listening', PROCESSING: 'processing', SHOWING: 'showing' };
let currentPhase = PHASE.IDLE;
let hasCompletedSetup = false;

// Batch result from backend
let batchResult = null;
let batchResultReceived = false;

// Variables to deduplicate chat log entries
let lastLogText = null;
let lastLogTime = 0;

// --- UI Helper ---
function updateCaptionUI(text, color, boxClass) {
    const masterBox = document.getElementById('masterCaptionBox');
    const masterText = document.getElementById('masterCaptionText');
    if (masterBox) masterBox.className = `call-caption-box ${boxClass || 'state-detecting'}`;
    if (masterText) {
        masterText.innerText = text;
        masterText.style.color = color || '#00e676';
    }
}

function updateStatusPill(text, className) {
    const el = document.getElementById('detectionStatus');
    if (el) {
        el.innerText = text;
        el.className = `status-pill ${className || 'detecting'}`;
    }
}

// --- Core Toggle ---
async function toggleTranslation() {
    translationEnabled = !translationEnabled;
    const btn = document.getElementById('transBtn');
    const masterBox = document.getElementById('masterCaptionBox');

    if (translationEnabled) {
        if (!localVideo.srcObject) {
            await startLocalStream();
            if (!localVideo.srcObject) {
                translationEnabled = false;
                return;
            }
        }

        btn.classList.add('active');
        masterBox.classList.remove('hidden');

        socket.emit('set_mode', 'general');
        socket.emit('clear_buffer');

        runPhasedCycle();
    } else {
        btn.classList.remove('active');
        masterBox.classList.add('hidden');
        const transEl = document.getElementById('masterCaptionTranslation');
        if (transEl) { transEl.style.display = 'none'; transEl.innerText = ''; }

        currentPhase = PHASE.IDLE;
        hasCompletedSetup = false;
        socket.emit('clear_buffer');
    }
}

// ================================================================
// MAIN CYCLE — captures frames locally, sends batch to backend
// ================================================================
async function runPhasedCycle() {
    if (currentPhase !== PHASE.IDLE && currentPhase !== PHASE.SHOWING) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 320;
    canvas.height = 240;

    while (translationEnabled) {

        // ═══════════════════════════════════════
        // PHASE 1: SETUP (3 seconds, one-time)
        // ═══════════════════════════════════════
        if (!hasCompletedSetup) {
            currentPhase = PHASE.SETUP;
            console.log('[Phase] SETUP — 3s camera warmup');
            updateCaptionUI('⏳ Setting up camera...', '#ffd54f', 'state-detecting');
            updateStatusPill('Setting Up', 'cooldown');

            socket.emit('clear_buffer');
            await sleep(3000);

            if (!translationEnabled) break;
            hasCompletedSetup = true;
        }

        // ═══════════════════════════════════════
        // PHASE 2: LISTENING (4 seconds — capture frames LOCALLY)
        // ═══════════════════════════════════════
        currentPhase = PHASE.LISTENING;
        let capturedFrames = [];
        socket.emit('clear_buffer');

        console.log('[Phase] LISTENING — 4s local frame capture');

        // Capture frames at ~15 FPS into local array
        // Training data was recorded at ~15 FPS effective rate
        // (cv2.VideoCapture + MediaPipe processing = ~66ms per frame)
        // 30 frames at 15 FPS = 2 seconds, matching the training data exactly
        let captureInterval = setInterval(() => {
            if (currentPhase !== PHASE.LISTENING || !translationEnabled) return;
            if (localVideo && localVideo.srcObject && localVideo.videoWidth > 0) {
                ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
                capturedFrames.push(canvas.toDataURL('image/jpeg', 0.7));
            }
        }, 66); // ~15 FPS to match training data

        // Countdown UI
        for (let remaining = 4; remaining >= 1; remaining--) {
            if (!translationEnabled) break;
            updateCaptionUI(`👂 Listening... ${remaining}s`, '#00e676', 'state-detecting');
            updateStatusPill(`Listening ${remaining}s`, 'detecting');
            await sleep(1000);
        }

        clearInterval(captureInterval);
        if (!translationEnabled) {
            capturedFrames = []; // Clean up memory
            break;
        }

        console.log(`[Phase] Captured ${capturedFrames.length} frames locally`);

        // ═══════════════════════════════════════
        // PHASE 3: PROCESSING — send batch to backend, wait for result
        // ═══════════════════════════════════════
        currentPhase = PHASE.PROCESSING;
        updateCaptionUI('🔍 Analyzing...', '#42a5f5', 'state-executing');
        updateStatusPill('Analyzing...', 'cooldown');

        // Send ALL captured frames — backend sliding window will find the best 30-frame segment
        console.log(`[Phase] Sending ${capturedFrames.length} frames to backend for sliding window analysis`);

        // Reset batch result and send
        batchResult = null;
        batchResultReceived = false;
        socket.emit('process_batch', { frames: capturedFrames });

        // Wait for batch_result event (timeout: 20s, or socket disconnect)
        const waitStart = Date.now();
        const BATCH_TIMEOUT_MS = 20000;
        while (!batchResultReceived && translationEnabled && (Date.now() - waitStart < BATCH_TIMEOUT_MS)) {
            await sleep(200);
        }

        // Handle timeout — show error instead of silently showing 'No sign'
        if (!batchResultReceived && translationEnabled) {
            console.warn('[Phase] Batch timed out — backend may be overloaded or disconnected');
            updateCaptionUI('⚠️ Analysis timed out. Retrying...', '#ff9800', 'state-executing');
            updateStatusPill('Timeout', 'cooldown');
            await sleep(2000);
            capturedFrames = [];
            batchResult = null;
            batchResultReceived = false;
            socket.emit('clear_buffer');
            continue; // Restart the cycle from LISTENING
        }

        if (!translationEnabled) break;

        // ═══════════════════════════════════════
        // PHASE 4: SHOWING — display result
        // ═══════════════════════════════════════
        currentPhase = PHASE.SHOWING;

        if (batchResult && batchResult.text && batchResult.conf >= 0.75) {
            console.log(`[Phase] SHOWING — "${batchResult.text}" (${(batchResult.conf * 100).toFixed(0)}%)`);
            await displayResult(batchResult.text, batchResult.conf);
        } else {
            console.log('[Phase] SHOWING — No sign detected');
            updateCaptionUI('❌ No sign detected', '#ef5350', 'state-executing');
            updateStatusPill('No sign', 'cooldown');
            await sleep(1500);
        }

        if (!translationEnabled) break;

        // Discard and restart
        capturedFrames = [];
        batchResult = null;
        batchResultReceived = false;
        socket.emit('clear_buffer');
    }

    currentPhase = PHASE.IDLE;
    console.log('[Phase] IDLE — cycle ended');
}

// ================================================================
// BATCH RESULT HANDLER — receives prediction from backend
// ================================================================
socket.on('batch_result', (data) => {
    console.log('[Batch Result]', data);
    batchResult = data;
    batchResultReceived = true;
});

// Keep prediction handler for remote caller display
socket.on('prediction', async (data) => {
    // Only show remote predictions (from the other person signing)
    if (data.sid !== socket.id && data.text) {
        console.log(`[Remote Prediction] ${data.text} (${(data.conf * 100).toFixed(0)}%)`);

        // Display in remote caption box
        const remoteCaptionBox = document.getElementById('remoteCaptionBox');
        const remoteCaptionText = document.getElementById('remoteCaptionText');
        if (remoteCaptionBox && remoteCaptionText) {
            remoteCaptionText.innerText = `${data.text} (${(data.conf * 100).toFixed(0)}%)`;
            remoteCaptionBox.classList.remove('hidden');

            // Translate if a non-English language is selected
            const lang = document.getElementById('langSelect').value;
            let displayText = data.text;
            if (lang !== 'en') {
                try {
                    const res = await fetch('/translate_sign', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: data.text, target_lang: lang })
                    });
                    if (res.ok) {
                        const tdata = await res.json();
                        displayText = tdata.translated || data.text;
                        remoteCaptionText.innerText = `${data.text} → ${displayText}`;
                    }
                } catch (e) {
                    console.warn('[Remote translate error]', e);
                }
            }

            // Add to chat log
            if (data.text !== lastLogText || (Date.now() - lastLogTime > 3000)) {
                const logMsg = (displayText !== data.text) 
                    ? `Partner signed: ${displayText}` 
                    : `Partner signed: ${data.text}`;
                addMessageToLog(logMsg, 'remote', data.text);
                lastLogText = data.text;
                lastLogTime = Date.now();
            }

            // Speak if TTS is enabled
            if (ttsEnabled) {
                speakText(displayText, lang || 'en');
            }

            // Auto-hide after 3 seconds
            clearTimeout(window._remoteCaptionTimeout);
            window._remoteCaptionTimeout = setTimeout(() => {
                remoteCaptionBox.classList.add('hidden');
            }, 3000);
        }
    }
});

// ================================================================
// ENGINE STATUS — simplified for phased approach
// ================================================================
socket.on('engine_status', (data) => {
    const statusEl = document.getElementById('detectionStatus');
    if (!statusEl) return;

    if (data.status === 'loading_model') {
        statusEl.innerText = 'Loading Model...';
        statusEl.className = 'status-pill cooldown';
        return;
    }
});

// ================================================================
// DISPLAY RESULT — shows the sign with translation/TTS/logging
// ================================================================
async function displayResult(text, conf) {
    const masterBox = document.getElementById('masterCaptionBox');
    const masterText = document.getElementById('masterCaptionText');
    const masterTranslation = document.getElementById('masterCaptionTranslation');
    const sid = (batchResult && batchResult.sid) ? batchResult.sid : socket.id;

    masterBox.className = 'call-caption-box state-executing';
    masterText.style.color = "#007bff";
    masterText.innerText = `✅ Sign: ${text} (${(conf * 100).toFixed(0)}%)`;
    updateStatusPill('Showing', 'cooldown');

    // Reset translation line
    if (masterTranslation) {
        masterTranslation.style.display = 'none';
        masterTranslation.innerText = '';
    }

    if (sid === socket.id) {
        // Local signer — log and translate if needed
        if (text !== lastLogText || (Date.now() - lastLogTime > 3000)) {
            addMessageToLog(`You signed: ${text}`, 'local');
            lastLogText = text;
            lastLogTime = Date.now();
        }

        // Translate for local signer too
        const lang = document.getElementById('langSelect').value;
        if (lang !== 'en') {
            try {
                const res = await fetch('/translate_sign', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, target_lang: lang })
                });
                if (res.ok) {
                    const tdata = await res.json();
                    const translated = tdata.translated || text;
                    if (masterTranslation) {
                        masterTranslation.innerText = translated;
                        masterTranslation.style.display = 'block';
                    }
                }
            } catch (e) {
                console.warn('[Local translate error]', e);
            }
        }
    } else {
        // Remote signer — log, translate if needed, speak
        if (text !== lastLogText || (Date.now() - lastLogTime > 3000)) {
            lastLogText = text;
            lastLogTime = Date.now();
        }

        if (ttsEnabled) {
            const lang = ttsSignLang;
            if (lang !== 'en') {
                try {
                    const res = await fetch('/translate_sign', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, target_lang: lang })
                    });
                    if (res.ok) {
                        const tdata = await res.json();
                        const translated = tdata.translated || text;
                        masterText.innerText = `✅ Sign: ${text} (${(conf * 100).toFixed(0)}%)`;
                        if (masterTranslation) {
                            masterTranslation.innerText = translated;
                            masterTranslation.style.display = 'block';
                        }
                        addMessageToLog(translated, 'remote', text);
                        speakText(translated, lang);
                    } else {
                        addMessageToLog(text, 'remote');
                        speakText(text, 'en');
                    }
                } catch (fetchErr) {
                    console.warn('[translate_sign fetch error]', fetchErr);
                    addMessageToLog(text, 'remote');
                    speakText(text, 'en');
                }
            } else {
                addMessageToLog(text, 'remote');
                speakText(text, 'en');
            }
        } else {
            const lang = document.getElementById('langSelect').value;
            if (lang !== 'en') {
                try {
                    const res = await fetch('/translate_sign', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, target_lang: lang })
                    });
                    if (res.ok) {
                        const tdata = await res.json();
                        const translated = tdata.translated || text;
                        masterText.innerText = `✅ Sign: ${text} (${(conf * 100).toFixed(0)}%)`;
                        if (masterTranslation) {
                            masterTranslation.innerText = translated;
                            masterTranslation.style.display = 'block';
                        }
                        addMessageToLog(translated, 'remote', text);
                    } else {
                        addMessageToLog(text, 'remote');
                    }
                } catch (e) {
                    addMessageToLog(text, 'remote');
                }
            } else {
                addMessageToLog(text, 'remote');
            }
        }
    }

    // Show result for 2 seconds
    await sleep(2000);
}

// ================================================================
// UTILITY
// ================================================================
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Receive Chat Message
socket.on('chat_message', (data) => {
    // Only display remote messages, local messages are handled on send
    if (data.sender !== socket.id) {
        addMessageToLog(data.text, 'remote', data.original);
        if (ttsEnabled) speakText(data.text, data.lang);
        playSignSequence(data.text);
    }
});

// Update window object for HTML access
window.toggleTranslation = toggleTranslation;

// --- Spelling Mode Logic ---
let spellingEnabled = false;

function toggleSpelling() {
    spellingEnabled = !spellingEnabled;
    const btn = document.getElementById('spellBtn');

    if (spellingEnabled) {
        btn.classList.add('active');
        socket.emit('set_mode', 'spelling');
    } else {
        btn.classList.remove('active');
        socket.emit('set_mode', 'general');
    }
}
// --- Layout Logic ---
function swapVideo() {
    const localWrapper = document.getElementById('localWrapper');
    const remoteWrapper = document.getElementById('remoteWrapper');

    if (localWrapper.classList.contains('floating')) {
        // Swap to make Local Main
        localWrapper.classList.remove('floating');
        localWrapper.classList.add('main');

        remoteWrapper.classList.remove('main');
        remoteWrapper.classList.add('floating');
    } else {
        // Swap back to Default (Remote Main)
        localWrapper.classList.remove('main');
        localWrapper.classList.add('floating');

        remoteWrapper.classList.remove('floating');
        remoteWrapper.classList.add('main');
    }
}
window.swapVideo = swapVideo;

// --- Sidebar Toggle ---
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const chatBtn = document.getElementById('chatBtn');
    if (sidebar) {
        sidebar.classList.toggle('collapsed');
        if (chatBtn) chatBtn.classList.toggle('active', !sidebar.classList.contains('collapsed'));
    }
}
window.toggleSidebar = toggleSidebar;

// --- Sidebar & Chat Logic ---
let ttsEnabled = false;
let recognition = null;

// Initialize Speech Recognition
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
        const text = event.results[0][0].transcript;
        document.getElementById('chatInput').value = text;
        toggleDictation(); // Stop after result
    };

    recognition.onerror = (event) => {
        console.error('Speech error:', event.error);
        toggleDictation();
    };
}

function toggleDictation() {
    const btn = document.getElementById('dictateBtn');
    if (btn.classList.contains('recording')) {
        recognition.stop();
        btn.classList.remove('recording');
    } else {
        if (recognition) {
            recognition.start();
            btn.classList.add('recording');
        } else {
            alert("Speech recognition not supported in this browser.");
        }
    }
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    const lang = document.getElementById('langSelect').value;

    if (text) {
        // Optimistically add message to local log before sending to socket
        addMessageToLog(text, 'local');
        socket.emit('chat_message', { text, target_lang: lang });
        input.value = '';
    }
}

function handleEncrypt(e) {
    if (e.key === 'Enter') sendMessage();
}

function addMessageToLog(text, type, original) {
    const log = document.getElementById('chatLog');
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-msg ${type}`;

    let content = text;
    if (original && original !== text) {
        content = `${original} <br><span class="msg-translation" style="font-size: 0.85em; opacity: 0.8; color: #38bdf8;">(${text})</span>`;
    }

    msgDiv.innerHTML = content;
    log.appendChild(msgDiv);
    log.scrollTop = log.scrollHeight;
}

// Apply saved TTS preference from pre-call modal
if (ttsAutoEnabled) {
    ttsEnabled = true;
    const ttsBtn = document.getElementById('ttsBtn');
    if (ttsBtn) ttsBtn.classList.add('active');
}

// === AUTO-INITIALIZATION (runs immediately since script loads after DOM) ===

// Set language dropdown
const _langSel = document.getElementById('langSelect');
if (_langSel && ttsSignLang) {
    _langSel.value = ttsSignLang;
}

// Auto-open Smart Chat sidebar if Speech Impaired profile is active
if (accSpeech) {
    toggleSidebar();
}

// Auto-join the call immediately
console.log('[SignVision] Auto-joining call...');
toggleCall();

// --- Text-to-Speech Toggle ---
function toggleTTS() {
    ttsEnabled = !ttsEnabled;
    const btn = document.getElementById('ttsBtn');
    if (btn) btn.classList.toggle('active', ttsEnabled);
    // Also sync ttsSignLang with the sidebar language selector if TTS is enabled
    if (ttsEnabled) {
        const sel = document.getElementById('langSelect');
        if (sel && sel.value) ttsSignLang = sel.value;
    }
}
window.toggleTTS = toggleTTS;


function speakText(text, lang = 'en') {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    // Lang mapping — extended with Malayalam
    const langMap = { 'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR', 'hi': 'hi-IN', 'de': 'de-DE', 'ml': 'ml-IN' };
    utterance.lang = langMap[lang] || 'en-US';
    window.speechSynthesis.speak(utterance);
}

function changeLanguage() {
    // Just updates the selection for next message
    const lang = document.getElementById('langSelect').value;
    console.log("Language changed to:", lang);
}

function changeModel() {
    const arch = document.getElementById('modelSelect').value;
    console.log("Model architecture changed to:", arch);
    socket.emit('set_architecture', arch);
}

// --- Text-to-Sign Player ---
// Simple mock mapping for demo - normally would fetch from DB
async function playSignSequence(text) {
    const container = document.getElementById('signPlayerContainer');
    const video = document.getElementById('signVideo');
    const words = text.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(' ');

    container.classList.add('show');

    for (const word of words) {
        // Try to play video if exists
        // We use a simplified check: request the file, if ok play, else skip
        const src = `/videos/${word}.mp4`; // Assuming direct mapping for now

        try {
            // Check availability (optional, or just try play)
            video.src = src;
            try {
                await video.play();
                // Wait for end
                await new Promise(r => {
                    video.onended = r;
                    video.onerror = r; // Skip if error
                });
            } catch (e) {
                console.log(`No video for ${word}`);
            }
        } catch (e) {
            // ignore
        }
    }

    // Hide after done
    setTimeout(() => {
        container.classList.remove('show');
        video.src = '';
    }, 2000);
}

window.toggleSpelling = toggleSpelling;
window.toggleDictation = toggleDictation;
window.sendMessage = sendMessage;
window.handleEncrypt = handleEncrypt;
window.changeLanguage = changeLanguage;
window.changeModel = changeModel;

// toggleSidebar is already defined above (line ~640)

// Initial check for speech support
if (!('webkitSpeechRecognition' in window)) {
    console.warn("Speech Recognition not supported in this browser.");
    document.getElementById('dictateBtn').style.display = 'none';
}

// Enhance Status UI
// statusBadge is already defined globally
const originalConnect = socket.io.engine.on('open', () => { }); // preserve

socket.on('connect', () => {
    statusBadge.innerText = "Connected";
    statusBadge.classList.add('connected');

    const arch = document.getElementById('modelSelect').value;
    socket.emit('set_architecture', arch);
});

socket.on('disconnect', () => {
    statusBadge.innerText = "Disconnected";
    statusBadge.classList.remove('connected');
});


// --- Text-to-Sign Search ---
async function searchAndPlaySign() {
    const input = document.getElementById('signSearchInput');
    const word = input.value.trim();
    if (!word) return;

    const container = document.getElementById('signPlayerContainer');
    const video = document.getElementById('signVideo');
    // Ensure label exists or fallback
    let label = container.querySelector('.player-label');
    if (!label) {
        // Create if missing (though it is in index.html)
        label = document.createElement('div');
        label.className = 'player-label';
        container.appendChild(label);
    }

    label.innerText = `Searching '${word}'...`;
    container.classList.remove('hidden');
    container.classList.add('show');

    try {
        const response = await fetch(`/search_sign?word=${encodeURIComponent(word)}`);

        if (response.ok) {
            const data = await response.json();
            if (data.video_url) {
                label.innerText = `Sign for: ${data.word}`;
                video.src = data.video_url;
                video.play().catch(e => console.error("Autoplay error:", e));

                // Share with the other person
                socket.emit('play_sign_video', { word: data.word, url: data.video_url });
            }
        } else {
            label.innerText = `Sign '${word}' not found`;
            setTimeout(() => {
                container.classList.add('hidden');
                container.classList.remove('show');
            }, 3000);
        }
    } catch (e) {
        console.error(e);
        label.innerText = "Error searching.";
    }
}

window.searchAndPlaySign = searchAndPlaySign;

// Handle receiving a shared sign video
socket.on('play_sign_video', (data) => {
    const container = document.getElementById('signPlayerContainer');
    const video = document.getElementById('signVideo');
    let label = container.querySelector('.player-label');

    if (!label) {
        label = document.createElement('div');
        label.className = 'player-label';
        container.appendChild(label);
    }

    label.innerText = `Remote shared sign: ${data.word}`;
    video.src = data.url;

    container.classList.remove('hidden');
    container.classList.add('show');

    video.play().catch(e => console.error("Autoplay error from remote:", e));

    // Auto hide after a few seconds or when ended
    video.onended = () => {
        setTimeout(() => {
            container.classList.add('hidden');
            container.classList.remove('show');
        }, 1000);
    };
});

// Bind Enter key
const searchInput = document.getElementById('signSearchInput');
if (searchInput) {
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchAndPlaySign();
    });
}
