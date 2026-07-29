import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from aiohttp import web
import socketio
import asyncio

# Create a Socket.IO server with high tolerance for CPU-blocking inference
sio = socketio.AsyncServer(
    async_mode='aiohttp', 
    cors_allowed_origins='*',
    ping_timeout=60, # Wait 60s for a ping instead of 5s
    ping_interval=25,
    max_http_buffer_size=10_000_000 # 10MB buffer for queued frames
)
app = web.Application()
sio.attach(app)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
app.router.add_static('/static', CURRENT_DIR)
# Serve WLASL videos for the Text-to-Sign feature
# Assuming videos are in d:/SignVision_S8_V2/Data/videos (or similar)
# We need to find where the raw mp4 files are.
# Based on wlasl_processor, they might be in Data/videos or downloaded on demand.
# For now, let's map a generic video folder if it exists.
VIDEO_DIR = os.path.join(CURRENT_DIR, '../../asl_dataset_video/videos')
if os.path.exists(VIDEO_DIR):
    app.router.add_static('/videos', VIDEO_DIR)

from deep_translator import GoogleTranslator
import aiohttp
from bs4 import BeautifulSoup

async def fetch_video_url(word):
    """
    Scrapes aslbloom.com to find the sign language video for a given word.
    """
    url = f"https://www.aslbloom.com/signs/{word.lower()}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 404:
                    return None
                html = await response.text()
        except Exception as e:
            print(f"Error fetching page {url}: {e}")
            return None
            
    soup = BeautifulSoup(html, 'html.parser')
    
    # Strategy 1: Look for the specific video class we found in analysis
    # Selector: .layout2_video.w-embed video
    # We look for all video tags and filter.
    videos = soup.find_all('video')
    
    for video in videos:
        # Check src attribute
        src = video.get('src')
        if src and "mp4" in src and "amazonaws" in src:
            return src
            
        # Check source children
        sources = video.find_all('source')
        for source in sources:
            src = source.get('src')
            if src and "mp4" in src and "amazonaws" in src:
                return src

    return None

async def search_sign(request):
    try:
        word = request.query.get('word')
        if not word:
            return web.json_response({'error': 'Missing word parameter'}, status=400)
            
        print(f"Searching for sign: {word}")
        video_url = await fetch_video_url(word)
        
        if video_url:
            return web.json_response({'word': word, 'video_url': video_url})
        else:
            return web.json_response({'error': 'Sign not found'}, status=404)
            
    except Exception as e:
        print(f"Search error: {e}")
        return web.json_response({'error': str(e)}, status=500)

async def home(request):
    with open(os.path.join(CURRENT_DIR, 'home.html'), 'r') as f:
        return web.Response(text=f.read(), content_type='text/html')

async def landing(request):
    with open(os.path.join(CURRENT_DIR, 'landing.html'), 'r') as f:
        return web.Response(text=f.read(), content_type='text/html')

async def call(request):
    with open(os.path.join(CURRENT_DIR, 'index.html'), 'r') as f:
        return web.Response(text=f.read(), content_type='text/html')

async def smart_home(request):
    with open(os.path.join(CURRENT_DIR, '../smart_home.html'), 'r', encoding='utf-8') as f:
        # Add cache busting headers so the new UI actually loads
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        return web.Response(text=f.read(), content_type='text/html', headers=headers)


async def translate_sign(request):
    """Translate a recognized sign word to the user's preferred language."""
    try:
        data = await request.json()
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'en')

        if not text:
            return web.json_response({'error': 'Missing text'}, status=400)

        # If target is English, no translation needed
        if target_lang == 'en':
            return web.json_response({'original': text, 'translated': text, 'lang': 'en'})

        # Map frontend lang codes to deep_translator codes
        lang_map = {'hi': 'hi', 'ml': 'ml', 'en': 'en'}
        dl_lang = lang_map.get(target_lang, 'en')

        translated = GoogleTranslator(source='en', target=dl_lang).translate(text)
        return web.json_response({'original': text, 'translated': translated, 'lang': target_lang})
    except Exception as e:
        print(f"translate_sign error: {e}")
        return web.json_response({'error': str(e)}, status=500)

app.router.add_get('/', home)
app.router.add_get('/landing', landing)
app.router.add_get('/call', call)
app.router.add_get('/search_sign', search_sign)
app.router.add_post('/translate_sign', translate_sign)
app.router.add_get('/smart_home.html', smart_home)
app.router.add_get('/smart_iot.html', smart_home) # New route to bypass cache


# Room management
# For this simple demo, we'll just use a default 'room'
ROOM = 'main_room'



@sio.event
async def ready(sid):
    print(f"Client {sid} is ready")
    await sio.emit('ready', skip_sid=sid, room=ROOM)

@sio.event
async def chat_message(sid, data):
    # data: { 'text': 'Hello', 'target_lang': 'es' }
    print(f"Chat from {sid}: {data}")
    
    room = ROOM
    # Translation Logic
    text = data.get('text', '')
    if not text:
        return
        
    target_lang = data.get('target_lang', 'en')
    
    translated_text = text
    if target_lang != 'en':
        try:
            # Map frontend lang codes to deep_translator codes
            lang_map = {'hi': 'hi', 'ml': 'ml', 'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de'}
            dl_lang = lang_map.get(target_lang, 'en')
            if dl_lang != 'en':
                translated_text = GoogleTranslator(source='auto', target=dl_lang).translate(text)
        except Exception as e:
            print(f"Chat Translation failed: {e}")
            translated_text = text # Fallback to original
        
    await sio.emit('chat_message', {
        'sender': sid,
        'original': text,
        'text': translated_text,
        'lang': target_lang
    }, skip_sid=sid, room=room)

@sio.event
async def play_sign_video(sid, data):
    # Relays a video URL to the other person in the room so they see the sign
    await sio.emit('play_sign_video', data, skip_sid=sid, room=ROOM)

@sio.event
async def message(sid, data):
    # Relay signaling messages (offer, answer, candidate)
    # print(f"Relaying {data.get('type')} from {sid}") # Reduce noise
    await sio.emit('message', data, skip_sid=sid, room=ROOM)

# --- Real-time Detection Logic ---
import sys
import base64
import numpy as np
import cv2

# Ensure src is in path
sys.path.append(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(os.path.join(CURRENT_DIR, '../../src'))

# Dictionary to store per-client engines
# {sid: InferenceEngine}
engines = {}
pending_modes = {}
pending_architectures = {}

try:
    from src.inference_engine import InferenceEngine
    print("Inference Engine Class Loaded.")
except Exception as e:
    print(f"Error loading InferenceEngine: {e}")
    InferenceEngine = None

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    await sio.enter_room(sid, ROOM)
    
    # Initialize engine for this client
    if InferenceEngine:
        try:
            engines[sid] = InferenceEngine()
            print(f"Engine initialized for {sid}")
            
            if sid in pending_architectures:
                engines[sid].set_architecture(pending_architectures.pop(sid))
                print(f"Applied pending architecture for {sid}")
                
            if sid in pending_modes:
                engines[sid].set_mode(pending_modes.pop(sid))
                print(f"Applied pending mode for {sid}")
        except Exception as e:
            print(f"Failed to init engine for {sid}: {e}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    await sio.leave_room(sid, ROOM)
    
    # Cleanup engine
    if sid in engines:
        engines[sid].close()
        del engines[sid]
        print(f"Engine cleaned up for {sid}")

@sio.event
async def set_mode(sid, mode):
    if sid in engines:
        engines[sid].set_mode(mode)
        print(f"Set mode for {sid} to {mode}")
    else:
        pending_modes[sid] = mode
        print(f"Pending mode for {sid} set to {mode}")

@sio.event
async def set_architecture(sid, arch):
    print(f"Requested architecture switch to {arch} for {sid}")
    
    # Notify frontend to pause frames and show 'Loading Model...'
    await sio.emit('engine_status', {'status': 'loading_model'}, room=sid)
    await asyncio.sleep(0.1) # Yield to event loop to send the message
    
    if sid in engines:
        # This is a blocking sync call, but necessary to hot-swap Keras graphs
        engines[sid].set_architecture(arch)
        status = 'cooldown' if engines[sid].is_cooldown_active else 'detecting'
        await sio.emit('engine_status', {'status': status}, room=sid)
        print(f"Successfully hot-swapped {sid} to {engines[sid].architecture}")
    else:
        pending_architectures[sid] = arch
        print(f"Pending arch for {sid} set to {arch}")

@sio.event
async def clear_buffer(sid):
    if sid in engines:
        engines[sid].clear_sequence()
        # print(f"Buffer cleared for {sid}") # Optional debug

@sio.event
async def process_frame(sid, data):
    engine = engines.get(sid)
    if engine is None:
        return

    try:
        # Decode Base64 image
        img_data = data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # Predict
        prediction, confidence = engine.predict(frame)
        
        # Emit status event (optimization: could be sent only on change, but simple polling is fine for now)
        status = 'cooldown' if engine.is_cooldown_active else 'detecting'
        await sio.emit('engine_status', {'status': status}, room=sid)

        # Emit prediction to exactly the Sender first to guarantee arrival
        if prediction:
            await sio.emit('prediction', {'sid': sid, 'text': prediction, 'conf': confidence}, room=sid)
            # Broadcast to remote caller, skipping sender
            await sio.emit('prediction', {'sid': sid, 'text': prediction, 'conf': confidence}, room=ROOM, skip_sid=sid)

    except Exception as e:
        print(f"Frame processing error: {e}")

@sio.event
async def process_batch(sid, data):
    """
    Receives a batch of base64-encoded frames captured on the frontend,
    decodes them all, and runs predict_batch for a single prediction.
    """
    engine = engines.get(sid)
    if engine is None:
        await sio.emit('batch_result', {'text': None, 'conf': 0.0}, room=sid)
        return

    try:
        frames_b64 = data.get('frames', [])
        print(f"Received batch of {len(frames_b64)} frames from {sid}")

        if not frames_b64:
            await sio.emit('batch_result', {'text': None, 'conf': 0.0}, room=sid)
            return

        # Decode all frames
        frames = []
        for frame_data in frames_b64:
            try:
                img_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
                img_bytes = base64.b64decode(img_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                continue  # Skip corrupted frames

        print(f"Decoded {len(frames)} valid frames from batch")

        if not frames:
            await sio.emit('batch_result', {'text': None, 'conf': 0.0}, room=sid)
            return

        # Run batch prediction
        prediction, confidence = engine.predict_batch(frames)

        result = {
            'sid': sid,
            'text': prediction,
            'conf': float(confidence)
        }
        print(f"Batch result: {result}")

        await sio.emit('batch_result', result, room=sid)

        # Also broadcast to remote caller if there's a prediction
        if prediction:
            await sio.emit('prediction', result, room=ROOM, skip_sid=sid)

    except Exception as e:
        print(f"Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        await sio.emit('batch_result', {'text': None, 'conf': 0.0}, room=sid)

if __name__ == '__main__':
    print("Starting WebRTC Signaling Server on http://0.0.0.0:8080")
    web.run_app(app, port=8080)
