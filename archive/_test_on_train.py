"""
Direct test of on_train via requests
"""
import os, sys
os.chdir(r"C:\Users\XJH\DeepPredict")
sys.stdout.reconfigure(encoding='utf-8')

import requests, json

API = "http://localhost:7861"

# First upload and call on_file_upload to set up state
with open(r"C:\Users\XJH\DeepResearch\test_data\demo\iris_header.csv", "rb") as f:
    r = requests.post(f"{API}/gradio_api/upload", files={"files": f})
fp = json.loads(r.text)[0]

filedata = {"path": fp, "url": None, "size": None, "orig_name": "iris_header.csv",
            "mime_type": "text/csv", "is_stream": False, "meta": {"_type": "gradio.FileData"}}

r2 = requests.post(f"{API}/gradio_api/call/on_file_upload", json={"data": [filedata]}, timeout=30)
event_id = json.loads(r2.text)["event_id"]

# Wait for on_file_upload to complete
import time
time.sleep(3)

# Now try calling on_train directly
train_inputs = [
    None,           # feature_col
    "sepal_length", # target_col  
    "regression",   # predict_mode
    "GradientBoosting",
    "",
    "10",
    "\u5206\u949f",
    "",
]

print("Calling on_train...")
r4 = requests.post(f"{API}/gradio_api/call/on_train",
                  json={"data": train_inputs}, timeout=120)
print(f"HTTP: {r4.status_code}")
print(f"Body: {r4.text[:200]}")

if r4.status_code == 200:
    event_id2 = json.loads(r4.text)["event_id"]
    
    import threading, queue
    q2 = queue.Queue()
    def get2():
        try:
            with requests.get(f"{API}/gradio_api/call/on_train/{event_id2}",
                             stream=True, timeout=120) as r5:
                for line in r5.iter_lines():
                    if line:
                        q2.put(line.decode())
        except Exception as e:
            q2.put(f"Error: {e}")
    
    t2 = threading.Thread(target=get2)
    t2.start()
    t2.join(timeout=90)
    
    lines = []
    while not q2.empty():
        lines.append(q2.get_nowait())
    
    print(f"\nGot {len(lines)} SSE lines:")
    for line in lines:
        print(f"  {line[:200]}")

print("\nDone")
