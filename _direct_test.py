"""
Direct import test of on_train function
"""
import os, sys
os.chdir(r"C:\Users\XJH\DeepPredict")
sys.stdout.reconfigure(encoding='utf-8')

# Import the actual function
from deeppredict_web import (
    data_loader, predictor, lstm_pred, data_decoupler,
    on_file_upload, on_train, extract_file_path
)
import gradio as gr

print("Imports OK")

# Load data
csv_path = r"C:\Users\XJH\DeepResearch\test_data\demo\iris_header.csv"
print(f"Loading: {csv_path}")

# Simulate on_file_upload
class FakeFile:
    def __init__(self, path):
        self.path = path
        self.orig_name = os.path.basename(path)

file_obj = FakeFile(csv_path)
result = on_file_upload(file_obj)
print(f"on_file_upload returned {len(result)} values")
print(f"data_loader.df shape: {data_loader.df.shape if data_loader.df is not None else None}")
print(f"data_loader.numeric_cols: {data_loader.numeric_cols if data_loader else None}")

# Now call on_train
print("\nCalling on_train...")
try:
    train_result = on_train(
        feature_col=None,
        target_col="sepal_length",
        predict_mode="regression",
        model_select="GradientBoosting",
        requirement="",
        n_future_val="10",
        n_future_unit="分钟",
        chart_requirement=""
    )
    print(f"on_train returned {len(train_result)} values")
    print(f"ZIP path: {train_result[-1]}")
    
    # Check outputs dir
    import shutil
    from pathlib import Path
    outputs = Path("outputs")
    if outputs.exists():
        zips = list(outputs.glob("*.zip"))
        print(f"\nZIP files found: {len(zips)}")
        if zips:
            latest = sorted(zips, key=lambda p: p.stat().st_mtime)[-1]
            print(f"Latest: {latest.name}")
            
            # Extract and list
            import zipfile
            extract_dir = Path("outputs/extracted")
            shutil.rmtree(extract_dir, ignore_errors=True)
            with zipfile.ZipFile(latest, "r") as zf:
                zf.extractall(extract_dir)
                print("\nZIP contents:")
                for name in sorted(zf.namelist()):
                    print(f"  {name}")
    else:
        print("outputs/ does not exist!")
        
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\nDone")
