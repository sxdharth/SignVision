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

callBtn.addEventListener('click', async () => {
    // Prevent double clicking
    if (callBtn.disabled) return;

    // If not connected, start process
    if (!localStream) {
        callBtn.innerText = 'Connecting...';
        callBtn.disabled = true;

        await startLocalStream();

        if (localStream) {
            callBtn.innerText = 'Waiting...';
            // Announce we are ready
            // The logic: 
            // 1. We join.
            // 2. If someone else is ALREADY there, they will receive our 'ready' and THEY will call US.
            // 3. Wait, if we are the second one, we emit 'ready'. 
            //    The FIRST person receives 'ready'.
            //    So the FIRST person (who is already waiting) should call the SECOND person (us).
            socket.emit('ready');
            isCreated = true;
        } else {
            callBtn.innerText = 'Join Call';
            callBtn.disabled = false;
        }
    }
});

cameraBtn.addEventListener('click', () => {
    if (localStream) {
        const videoTrack = localStream.getVideoTracks()[0];
        videoTrack.enabled = !videoTrack.enabled;
        updateMediaButton(cameraBtn, videoTrack.enabled, 'videocam', 'videocam-off');
    }
});

micBtn.addEventListener('click', () => {
    if (localStream) {
        const audioTrack = localStream.getAudioTracks()[0];
        audioTrack.enabled = !audioTrack.enabled;
        updateMediaButton(micBtn, audioTrack.enabled, 'mic', 'mic-off');
    }
});

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
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localVideo.srcObject = localStream;
        showStatus('Camera Ready - Joining...');
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
        callBtn.innerText = 'Connected';
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

// --- Live Translation Logic ---
let translationEnabled = false;
let transInterval = null;

function toggleTranslation() {
    translationEnabled = !translationEnabled;
    const btn = document.getElementById('transBtn');
    const localBox = document.getElementById('localCaptionBox');
    const remoteBox = document.getElementById('remoteCaptionBox');

    if (translationEnabled) {
        btn.classList.add('active');
        // Do not immediately show the boxes until there is a prediction
        startTranslation();
    } else {
        btn.classList.remove('active');
        localBox.classList.add('hidden');
        remoteBox.classList.add('hidden');
        stopTranslation();
    }
}

function startTranslation() {
    // Canvas for grabbing frames
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    transInterval = setInterval(() => {
        if (!remoteVideo.srcObject) return;

        // Set canvas size to match video
        if (remoteVideo.videoWidth > 0 && remoteVideo.videoHeight > 0) {
            canvas.width = remoteVideo.videoWidth / 4; // Scale down for speed
            canvas.height = remoteVideo.videoHeight / 4;

            // Draw frame
            context.drawImage(remoteVideo, 0, 0, canvas.width, canvas.height);

            // Compress to JPEG
            const dataUrl = canvas.toDataURL('image/jpeg', 0.5);

            // Send to server
            socket.emit('process_frame', dataUrl);
        }
    }, 200); // 5 FPS is enough for gesture context, maybe increase to 100ms (10 FPS) if fast
}

function stopTranslation() {
    if (transInterval) {
        clearInterval(transInterval);
        transInterval = null;
    }
}

// Receive prediction
socket.on('prediction', (data) => {
    const { sid, text, conf } = data;
    const isLocal = sid === socket.id;

    let targetBox, targetText;

    if (isLocal) {
        targetBox = document.getElementById('localCaptionBox');
        targetText = document.getElementById('localCaptionText');

        // Add to Chat Log as "System/Sign" if local
        addMessageToLog(`You signed: ${text}`, 'local');
    } else {
        targetBox = document.getElementById('remoteCaptionBox');
        targetText = document.getElementById('remoteCaptionText');

        // Add to Chat Log as Remote Sign
        addMessageToLog(`Remote signed: ${text}`, 'remote');

        // TTS: Speak the predicted sign for remote only!
        if (ttsEnabled) {
            speakText(text);
        }
    }

    if (targetBox && targetText) {
        targetText.innerText = `Sign: ${text} (${(conf * 100).toFixed(0)}%)`;
        targetBox.classList.remove('hidden');

        // Clear timeout to hide for the specific box
        const timeoutKey = isLocal ? 'localCaptionTimeout' : 'remoteCaptionTimeout';
        if (window[timeoutKey]) clearTimeout(window[timeoutKey]);

        window[timeoutKey] = setTimeout(() => {
            targetBox.classList.add('hidden');
        }, 3000);
    }
});

// Receive Engine Status
socket.on('engine_status', (data) => {
    const statusEl = document.getElementById('detectionStatus');
    if (!statusEl) return;

    if (data.status === 'detecting') {
        statusEl.innerText = 'Detecting';
        statusEl.className = 'status-pill detecting';
    } else {
        statusEl.innerText = 'Waiting...';
        statusEl.className = 'status-pill cooldown';
    }
});

// Receive Chat Message
socket.on('chat_message', (data) => {
    const isLocal = data.sender === socket.id;
    addMessageToLog(data.text, isLocal ? 'local' : 'remote', data.original);

    // TTS: Read aloud if remote message or translated
    if (!isLocal && ttsEnabled) {
        speakText(data.text, data.lang);
    }

    // Text-to-Sign: If remote message, try to play signs
    if (!isLocal) {
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
        content = `${original}<span class="msg-translation">${text}</span>`;
    }

    msgDiv.innerHTML = content;
    log.appendChild(msgDiv);
    log.scrollTop = log.scrollHeight;
}

// --- Text-to-Speech ---
function toggleTTS() {
    ttsEnabled = !ttsEnabled;
    const btn = document.getElementById('ttsBtn');
    btn.classList.toggle('active', ttsEnabled);
}

function speakText(text, lang = 'en') {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    // Simple Lang mapping
    const langMap = { 'en': 'en-US', 'es': 'es-ES', 'fr': 'fr-FR', 'hi': 'hi-IN', 'de': 'de-DE' };
    utterance.lang = langMap[lang] || 'en-US';
    window.speechSynthesis.speak(utterance);
}

function changeLanguage() {
    // Just updates the selection for next message
    const lang = document.getElementById('langSelect').value;
    console.log("Language changed to:", lang);
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

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const btn = document.getElementById('chatBtn');
    sidebar.classList.toggle('collapsed');
    btn.classList.toggle('active', !sidebar.classList.contains('collapsed'));
}
window.toggleSidebar = toggleSidebar;

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

// Bind Enter key
const searchInput = document.getElementById('signSearchInput');
if (searchInput) {
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchAndPlaySign();
    });
}
