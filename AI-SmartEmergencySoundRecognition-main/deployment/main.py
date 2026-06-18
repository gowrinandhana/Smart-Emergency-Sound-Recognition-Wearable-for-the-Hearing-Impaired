# -*- coding: utf-8 -*-
import time
import os
import threading
import numpy as np
from collections import deque

# --- HARDWARE LIBRARIES ---
try:
    from gpiozero import OutputDevice
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import sh1106 
except ImportError:
    print("\n? ERROR: Hardware libraries not found.")
    print("Please run: pip install gpiozero luma.oled\n")
    exit()

# ===========================
# ZONE 1: HARDWARE SETUP
# ===========================
print("\n--- INITIALIZING HARDWARE ---")

# Setup Motor
vibration_motor = None
try:
    vibration_motor = OutputDevice(17)
    print("? Motor connected on GPIO 17")
    # STARTUP BUZZ TEST
    vibration_motor.on()
    time.sleep(0.2)
    vibration_motor.off()
except Exception as e:
    print(f"?? Motor Setup Failed: {e}")

# Setup Display
display = None
try:
    serial = i2c(port=1, address=0x3C)
    display = sh1106(serial) 
    print("? OLED Display connected (sh1106)")
    
    with canvas(display) as draw:
        draw.text((10, 20), "STARTING UP...", fill="white")
        draw.text((10, 35), "Loading Model...", fill="white")
except Exception as e:
    print(f"?? Display Setup Failed: {e}")

# ===========================
# ZONE 2: AI & AUDIO CONFIG
# ===========================
print("\n--- LOADING AI MODELS ---")
import sounddevice as sd
import joblib
from scipy.signal import butter, lfilter
import tflite_runtime.interpreter as tflite

BASE_DIR = "/home/raspberrypi/sound_project/" 
SVM_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.npy")
YAMNET_PATH = os.path.join(BASE_DIR, "yamnet.tflite")

if not os.path.exists(SVM_PATH): 
    print(f"? Error: Could not find {SVM_PATH}")
    exit()

svm_model = joblib.load(SVM_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)
interpreter = tflite.Interpreter(model_path=YAMNET_PATH)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
embeddings_output_index = interpreter.get_output_details()[1]['index'] 

print("? AI Models Loaded!")

if display:
    with canvas(display) as draw:
        draw.text((10, 20), "System Ready", fill="white")
        draw.text((10, 35), "Listening...", fill="white")

MIC_SAMPLE_RATE = 16000 
CHUNK_SIZE = 1024        
BUFFER_DURATION = 1.0    
SILENCE_THRESHOLD = 0.003  
FILTER_CUTOFF = 150        

THRESHOLDS = {
    'train': 1.00,     
    'rain': 1.00,           
    'thunderstorm': 1.00,
    'gunshot': 1.00,
    'door_knock': 0.65,
    'crying_baby': 0.35,    
    'fire_smoke_alarm': 0.50,
    'dog': 0.50,
    'siren': 0.50,
    'glass_breaking': 0.50,
    'DEFAULT': 0.50
}

# ===========================
# ZONE 3: HELPER FUNCTIONS
# ===========================
def trigger_hardware_alert(alert_text):
    """Runs the vibration and screen update in the background."""
    def alert_task():
        if display:
            try:
                with canvas(display) as draw:
                    draw.rectangle(display.bounding_box, outline="white", fill="black")
                    draw.text((5, 15), "?? ALERT! ??", fill="white")
                    draw.text((5, 35), alert_text, fill="white")
            except:
                pass 
        
        if vibration_motor:
            for _ in range(3):
                vibration_motor.on()
                time.sleep(0.3)
                vibration_motor.off()
                time.sleep(0.2)
        
        time.sleep(2.0)
        if display:
            try:
                with canvas(display) as draw:
                    draw.text((10, 25), "Listening...", fill="white")
            except:
                pass

    threading.Thread(target=alert_task, daemon=True).start()

def apply_high_pass_filter(audio_data, cutoff=FILTER_CUTOFF, fs=MIC_SAMPLE_RATE):
    nyquist = 0.5 * fs
    b, a = butter(5, cutoff / nyquist, btype='high', analog=False)
    return lfilter(b, a, audio_data)

def extract_hybrid_embedding(audio_data):
    waveform = audio_data.astype(np.float32)
    interpreter.resize_tensor_input(input_index, [len(waveform)])
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, waveform)
    interpreter.invoke()
    embeddings = interpreter.get_tensor(embeddings_output_index)
    return np.concatenate([np.mean(embeddings, axis=0), np.max(embeddings, axis=0)])

# ===========================
# ZONE 4: MAIN LOOP
# ===========================
audio_buffer = deque(maxlen=int(MIC_SAMPLE_RATE / CHUNK_SIZE * BUFFER_DURATION))

def audio_callback(indata, frames, time, status):
    audio_buffer.append(indata.copy())
    volume = np.max(np.abs(indata))
    if volume > 0.001:
        bar = '#' * int(volume * 50) 
        print(f"\rVolume: |{bar.ljust(20)}| {volume:.4f}", end="", flush=True)

def main():
    print(f"\n?? LISTENING CONTINUOUSLY...")
    print("-" * 50)

    stream = sd.InputStream(device=2, channels=1, samplerate=MIC_SAMPLE_RATE, 
                            callback=audio_callback, blocksize=CHUNK_SIZE)
    
    last_detection_time = 0
    COOLDOWN_SECONDS = 4.0 
    
    # NEW VARIABLES FOR OVERRIDE LOGIC
    last_detected_class = None
    last_detected_confidence = 0.0

    with stream:
        while True:
            if len(audio_buffer) == audio_buffer.maxlen:
                full_clip = np.concatenate(audio_buffer, axis=0).flatten()
                clean_clip = apply_high_pass_filter(full_clip)
                vol = np.max(np.abs(clean_clip))
                
                if vol < SILENCE_THRESHOLD:
                    time.sleep(0.1) 
                    continue

                try:
                    embedding = extract_hybrid_embedding(clean_clip).reshape(1, -1)
                    probs = svm_model.predict_proba(embedding)[0]
                    pred_idx = np.argmax(probs)
                    
                    detected_class = classes[pred_idx]
                    confidence = probs[pred_idx]
                    
                    print(f"\nDebug: Heard '{detected_class}' (Conf: {confidence:.2f}) | Vol: {vol:.3f}")
                    
                    required_conf = THRESHOLDS.get(detected_class, THRESHOLDS['DEFAULT'])
                    current_time = time.time()
                    
                    if confidence >= required_conf:
                        
                        time_since_last_alert = current_time - last_detection_time
                        cooldown_active = time_since_last_alert < COOLDOWN_SECONDS
                        
                        # OVERRIDE LOGIC: Allow update if confidence is much higher than the last alert
                        is_higher_confidence = (confidence > (last_detected_confidence + 0.15))
                        is_different_class = (detected_class != last_detected_class)
                        
                        trigger_alert = False
                        
                        if not cooldown_active:
                            trigger_alert = True
                            print(f"?? ALERT: {detected_class.upper()} DETECTED!")
                        elif cooldown_active and is_different_class and is_higher_confidence:
                            trigger_alert = True
                            print(f"? OVERRIDE ALERT: {detected_class.upper()} (Conf: {confidence:.2f}) overrode {last_detected_class.upper()} (Conf: {last_detected_confidence:.2f})")
                        else:
                            print(f"   (Skipping alert: Cooldown active)")

                        if trigger_alert:
                            trigger_hardware_alert(detected_class.upper())
                            last_detection_time = current_time 
                            last_detected_class = detected_class
                            last_detected_confidence = confidence
                            
                except Exception as e:
                    print(f"\nError: {e}")
                    
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")
        if display: display.clear()
        if vibration_motor: vibration_motor.off()
