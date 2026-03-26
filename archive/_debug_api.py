"""
Debug: Check why on_train fails via API
"""
import os
os.chdir(r"C:\Users\XJH\DeepPredict")

import requests, json, threading, queue

API = "http://localhost:7861"

# Step 1: Upload
with open(r"C:\Users\XJH\DeepResearch\test_data\demo\iris_header.csv", "rb") as f:
    r = requests.post(f"{API}/gradio_api/upload", files={"files": f})
fp = json.loads(r.text)[0]
print("Uploaded:", fp)

# Step 2: Call on_file_upload via API
filedata = {
    "path": fp, "url": None, "size": None, "orig_name": "iris_header.csv",
    "mime_type": "text/csv", "is_stream": False, "meta": {"_type": "gradio.FileData"}
}
r2 = requests.post(f"{API}/gradio_api/call/on_file_upload",
                   json={"data": [filedata]}, timeout=30)
event_id = json.loads(r2.text)["event_id"]
print("on_file_upload event:", event_id)

# Get result
q = queue.Queue()
def get1():
    try:
        with requests.get(f"{API}/gradio_api/call/on_file_upload/{event_id}",
                         stream=True, timeout=15) as r3:
            for line in r3.iter_lines():
                if line:
                    q.put(("upload", line.decode()))
    except Exception as e:
        q.put(("error", str(e)))

t = threading.Thread(target=get1)
t.start()
t.join(timeout=12)

while not q.empty():
    tag, data = q.get_nowait()
    if tag == "upload" and data.startswith("data:"):
        result = json.loads(data[5:])
        print(f"\non_file_upload returned {len(result)} values")
        for i, v in enumerate(result):
            tname = type(v).__name__
            print(f"  [{i}] {tname}: {str(v)[:80]}")

# Step 3: Call on_train
print("\n=== Calling on_train ===")
train_inputs = [
    None,           # feature_col
    "sepal_length", # target_col
    "regression",   # predict_mode
    "GradientBoosting",  # model_select
    "",             # requirement
    "10",           # n_future_val
    "分钟",          # n_future_unit
    "",             # chart_requirement
]

r4 = requests.post(f"{API}/gradio_api/call/on_train",
                   json={"data": train_inputs}, timeout=120)
print(f"on_train HTTP: {r4.status_code}, body: {r4.text[:200]}")

if r4.status_code == 200:
    event_id2 = json.loads(r4.text)["event_id"]
    print(f"on_train event: {event_id2}")
    
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
    
    while not q2.empty():
        print("SSE:", q2.get_nowait()[:150])

print("\nDone")
