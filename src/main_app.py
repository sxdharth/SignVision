import cv2
import pyttsx3
from deep_translator import GoogleTranslator
from inference_engine import InferenceEngine
import threading
import time

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    engine = InferenceEngine()
    cap = cv2.VideoCapture(0)
    
    translator = GoogleTranslator(source='auto', target='es') # Spanish by default
    
    tts_enabled = False
    translation_enabled = False
    last_prediction = ""
    current_sentence = []
    
    print("Starting Main App...")
    print("Controls:")
    print("  'q': Quit")
    print("  't': Toggle TTS")
    print("  'l': Toggle Translation (to Spanish)")
    print("  's': Toggle Spelling Mode")
    print("  'c': Clear sentence")
    
    spelling_mode = False

    while True:
        # Update engine mode
        engine.set_mode('spelling' if spelling_mode else 'general')
        ret, frame = cap.read()
        if not ret:
            break
            
        prediction, confidence = engine.predict(frame)
        
        display_frame = frame.copy()
        
        # UI Overlay
        cv2.rectangle(display_frame, (0, 0), (640, 40), (245, 117, 16), -1)
        
        mode_str = "SPELL" if spelling_mode else "GEN"
        status_text = f"Mode: {mode_str} | TTS: {'ON' if tts_enabled else 'OFF'} | Trans: {'ON' if translation_enabled else 'OFF'}"
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection Status
        if engine.is_cooldown_active:
            cv2.putText(display_frame, "Status: WAIT", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange
        else:
            cv2.putText(display_frame, "Status: DETECT", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green
        
        if prediction:
            res_text = f"{prediction} ({confidence:.2f})"
            cv2.putText(display_frame, res_text, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if prediction != last_prediction:
                current_sentence.append(prediction)
                last_prediction = prediction
                
                if tts_enabled:
                    threading.Thread(target=speak, args=(prediction,)).start()
        
        sentence_str = " ".join(current_sentence)
        if len(sentence_str) > 50:
            current_sentence = current_sentence[-5:] # Keep last 5 words
            sentence_str = " ".join(current_sentence)
            
        cv2.rectangle(display_frame, (0, 440), (640, 480), (0, 0, 0), -1)
        cv2.putText(display_frame, sentence_str, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if translation_enabled and sentence_str:
            try:
                translated = translator.translate(sentence_str)
                cv2.rectangle(display_frame, (0, 400), (640, 440), (0, 0, 0), -1)
                cv2.putText(display_frame, translated, (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            except Exception as e:
                pass

        cv2.imshow('Sign Language Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            tts_enabled = not tts_enabled
        elif key == ord('l'):
            translation_enabled = not translation_enabled
        elif key == ord('s'):
            spelling_mode = not spelling_mode
        elif key == ord('c'):
            current_sentence = []
            last_prediction = ""

    cap.release()
    cv2.destroyAllWindows()
    engine.close()

if __name__ == "__main__":
    main()
