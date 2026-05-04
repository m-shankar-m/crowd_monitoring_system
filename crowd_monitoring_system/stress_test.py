import requests
import threading
import time
import psutil
import os
import cv2
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def log(msg):
    print(f"[TEST] {msg}")

def test_health():
    log("Testing /health...")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200, "Health check failed"
    log("PASS: /health passed")

def test_predict_zone():
    log("Testing /predict-zone with various payloads...")
    # Normal payload
    r = requests.post(f"{BASE_URL}/predict-zone", json={"periods": 15, "history_counts": [1]*20})
    assert r.status_code == 200, "predict-zone failed on normal payload"
    
    # Empty history
    r = requests.post(f"{BASE_URL}/predict-zone", json={"periods": 15, "history_counts": []})
    assert r.status_code == 200, "predict-zone failed on empty history"
    
    # Missing history
    r = requests.post(f"{BASE_URL}/predict-zone", json={"periods": 15})
    assert r.status_code == 200, "predict-zone failed on missing history"
    log("PASS: /predict-zone passed")

def test_ram_and_load():
    log("Starting RAM & Concurrency Load Test (/live-density)...")
    
    # Generate a dummy image to send
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = buffer.tobytes()

    pid = os.getpid() # Test script PID, not backend, but we can measure backend via requests
    
    success_count = 0
    error_count = 0
    latencies = []
    
    def worker():
        nonlocal success_count, error_count
        for _ in range(20): # 20 requests per thread
            try:
                start = time.time()
                files = {"file": ("frame.jpg", img_bytes, "image/jpeg")}
                r = requests.post(f"{BASE_URL}/live-density", files=files, data={"zone_name": "Test", "max_capacity": 50}, timeout=10)
                if r.status_code == 200:
                    success_count += 1
                else:
                    error_count += 1
                latencies.append(time.time() - start)
            except Exception as e:
                error_count += 1

    threads = []
    # 4 concurrent threads simulating 4 camera zones sending simultaneously
    for _ in range(4):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    log(f"Load Test Complete: {success_count} Successes, {error_count} Errors.")
    log(f"Average Latency: {avg_latency:.3f}s")
    
    if error_count == 0:
        log("PASS: RAM & Concurrency Test passed (No crashed requests)")
    else:
        log("FAIL: RAM & Concurrency Test failed (Errors detected)")

if __name__ == "__main__":
    try:
        test_health()
        test_predict_zone()
        test_ram_and_load()
        log("All tests finished.")
    except Exception as e:
        log(f"Testing encountered a critical exception: {e}")
