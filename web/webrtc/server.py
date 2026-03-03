import os
from aiohttp import web
import socketio

# Create a Socket.IO server
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
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


app.router.add_get('/', landing)
app.router.add_get('/call', call)
app.router.add_get('/search_sign', search_sign)
app.router.add_get('/smart_home.html', smart_home)
app.router.add_get('/smart_iot.html', smart_home) # New route to bypass cache


# Room management
# For this simple demo, we'll just use a default 'room'
ROOM = 'main_room'



@sio.event
async def ready(sid):
    # Broadcast 'ready' to others
    # The client who receives 'ready' will initiate the offer if they are already there
    print(f"Client {sid} is ready")
    await sio.emit('ready', skip_sid=sid, room=ROOM)

    await sio.emit('ready', skip_sid=sid, room=ROOM)

@sio.event
async def chat_message(sid, data):
    # data: { 'text': 'Hello', 'lang': 'es' }
    print(f"Chat from {sid}: {data}")
    
    room = ROOM
    # Translation Logic
    text = data.get('text')
    target_lang = data.get('target_lang', 'en')
    
    translated_text = text
    try:
        if target_lang != 'en': # Assuming source is english for now, or auto
            translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation failed: {e}")
        
    await sio.emit('chat_message', {
        'sender': sid,
        'original': text,
        'text': translated_text,
        'lang': target_lang
    }, room=room)

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

        if prediction:
            await sio.emit('prediction', {'sid': sid, 'text': prediction, 'conf': confidence}, room=ROOM)

    except Exception as e:
        print(f"Frame processing error: {e}")

if __name__ == '__main__':
    print("Starting WebRTC Signaling Server on http://0.0.0.0:8080")
    web.run_app(app, port=8080)
