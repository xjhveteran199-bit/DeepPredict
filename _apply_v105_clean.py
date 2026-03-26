"""
v1.05 clean patch: Only add BA chart + training history to v1.04
No other changes - preserves the working file upload functionality
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\XJH\DeepPredict\deeppredict_web.py', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
patches = []

# ============================================================
# PATCH 1: text.usetex=False after matplotlib.use('Agg')
# ============================================================
for i, l in enumerate(lines):
    if "matplotlib.use('Agg')" in l:
        # Check if text.usetex already added
        if i+1 < len(lines) and "text.usetex" not in lines[i]:
            # Add text.usetex after this line
            lines.insert(i+1, "mpl.rcParams['text.usetex'] = False  # 禁用 LaTeX，纯 matplotlib 渲染\n")
            patches.append("1. text.usetex=False")
            print("Patch 1: text.usetex=False")
        break

# ============================================================
# PATCH 2: Bland-Altman function before class GradientPlot
# ============================================================
for i, l in enumerate(lines):
    if 'class GradientPlot' in l:
        BA_FUNC = '''

def plot_bland_altman(y_true, y_pred, title="Bland-Altman"):
    """Bland-Altman \u4e00\u81f4\u6027\u5206\u6790\u56fe"""
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()
    mean_arr = (y_pred_arr + y_true_arr) / 2
    diff_arr = y_pred_arr - y_true_arr
    mean_diff = np.mean(diff_arr)
    std_diff = np.std(diff_arr, ddof=1)
    loa_lo = mean_diff - 1.96 * std_diff
    loa_hi = mean_diff + 1.96 * std_diff
    fig = Figure(figsize=(7, 5), dpi=150)
    ax = fig.add_subplot(111)
    ax.scatter(mean_arr, diff_arr, alpha=0.6, s=20, color="#4DBBD5")
    ax.axhline(mean_diff, color="#E64B35", lw=2, label="Mean={:.4f}".format(mean_diff))
    ax.axhline(loa_lo, color="#8491B4", lw=1.5, linestyle="--", label="95% LoA=[{:.4f}, {:.4f}]".format(loa_lo, loa_hi))
    ax.axhline(loa_hi, color="#8491B4", lw=1.5, linestyle="--")
    ax.axhline(0, color="gray", lw=1, linestyle=":", alpha=0.7)
    ax.set_xlabel("(Predicted + Actual) / 2", fontsize=11)
    ax.set_ylabel("Predicted - Actual", fontsize=11)
    ax.set_title(title + " - Bland-Altman Analysis", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

'''
        lines.insert(i, BA_FUNC)
        patches.append("2. Bland-Altman function")
        print("Patch 2: Bland-Altman function")
        break

# ============================================================
# PATCH 3: train_history tracking in sklearn training
# Find "self.model.fit(X_train, y_train)" and replace with loop
# ============================================================
for i, l in enumerate(lines):
    if 'self.model.fit(X_train, y_train)' in l and 'for ep' not in lines[i-1]:
        indent = ' ' * (len(l) - len(l.lstrip()))
        inner_indent = indent + '    '
        # Replace single fit with loop that tracks history
        old_block = (
            indent + 'self.model.fit(X_train, y_train)\n' +
            lines[i+1] +  # train_pred = self.model.predict(X_train)
            lines[i+2]     # if self.task_type == ... self.train_history = ...
        )
        new_block = (
            indent + 'n_estimators = params.get("n_estimators", 50) if hasattr(self.model, "n_estimators") else 1\n' +
            indent + 'max_show = min(10, n_estimators) if n_estimators > 1 else 1\n' +
            indent + 'self.train_history = {"epoch": list(range(1, max_show + 1))}\n' +
            inner_indent + 'if self.task_type == "classification":\n' +
            inner_indent + '    from sklearn.metrics import accuracy_score\n' +
            inner_indent + '    if max_show == 1:\n' +
            inner_indent + '        self.model.fit(X_train, y_train)\n' +
            inner_indent + '        train_pred = self.model.predict(X_train)\n' +
            inner_indent + '        self.train_history["accuracy"] = [round(accuracy_score(y_train, train_pred), 6)]\n' +
            inner_indent + '    else:\n' +
            inner_indent + '        self.train_history["accuracy"] = []\n' +
            inner_indent + '        for ep in range(max_show):\n' +
            inner_indent + '            self.model.fit(X_train, y_train)\n' +
            inner_indent + '            train_pred = self.model.predict(X_train)\n' +
            inner_indent + '            self.train_history["accuracy"].append(round(accuracy_score(y_train, train_pred), 6))\n' +
            inner_indent + 'else:\n' +
            inner_indent + '    from sklearn.metrics import mean_squared_error\n' +
            inner_indent + '    if max_show == 1:\n' +
            inner_indent + '        self.model.fit(X_train, y_train)\n' +
            inner_indent + '        train_pred = self.model.predict(X_train)\n' +
            inner_indent + '        self.train_history["loss"] = [round(np.sqrt(mean_squared_error(y_train, train_pred)), 6)]\n' +
            inner_indent + '    else:\n' +
            inner_indent + '        self.train_history["loss"] = []\n' +
            inner_indent + '        for ep in range(max_show):\n' +
            inner_indent + '            self.model.fit(X_train, y_train)\n' +
            inner_indent + '            train_pred = self.model.predict(X_train)\n' +
            inner_indent + '            self.train_history["loss"].append(round(np.sqrt(mean_squared_error(y_train, train_pred)), 6))\n'
        )
        if lines[i+1].strip().startswith('train_pred') and lines[i+2].strip().startswith('if self.task_type'):
            lines[i:i+3] = [new_block]
            patches.append("3. train_history tracking")
            print("Patch 3: train_history tracking")
        break

# ============================================================
# PATCH 4: BA chart generation + training history in ZIP
# Find the ZIP creation block and add BA/history files
# ============================================================
for i, l in enumerate(lines):
    if 'zf.write(output_dir / "metrics.json"' in l:
        # Insert BA chart generation before ZIP block, and add files to ZIP
        # Find the line before ZIP block
        # We need to insert BA generation code before zip_name = ...
        # And add BA/history files to the ZIP
        indent = ' ' * (len(l) - len(l.lstrip()))
        inner_indent = indent + '    '

        # BA generation code - insert before zip_name
        ba_gen = (
            indent + '# Bland-Altman + Training History chart generation\n'
            + indent + 'ba_png_path = None\n'
            + indent + 'ba_csv_path = None\n'
            + indent + 'hist_png_path = None\n'
            + indent + 'hist_csv_path = None\n'
            + indent + 'if predictor and predictor.is_fitted and predictor.task_type == "regression":\n'
            + inner_indent + 'try:\n'
            + inner_indent + '    split_idx = min(int(len(X_df) * 0.8), len(X_df) - 1)\n'
            + inner_indent + '    X_test_ba = X_df.iloc[split_idx:].select_dtypes(include=[np.number])\n'
            + inner_indent + '    y_test_ba = np.asarray(y_series.iloc[split_idx:]).flatten()\n'
            + inner_indent + '    if len(y_test_ba) > 0:\n'
            + inner_indent + '        X_scaled_ba = predictor.scaler.transform(X_test_ba.fillna(X_test_ba.median()))\n'
            + inner_indent + '        y_pred_ba = predictor.model.predict(X_scaled_ba)\n'
            + inner_indent + '        ba_fig = plot_bland_altman(y_test_ba, y_pred_ba, title=model_name)\n'
            + inner_indent + '        ba_png_path = str(output_dir / "bland_altman.png")\n'
            + inner_indent + '        ba_fig.savefig(ba_png_path, dpi=150, bbox_inches="tight")\n'
            + inner_indent + '        plt.close(ba_fig)\n'
            + inner_indent + '        ba_df = pd.DataFrame({"mean": list((y_pred_ba + y_test_ba) / 2), "diff": list(y_pred_ba - y_test_ba)})\n'
            + inner_indent + '        ba_csv_path = str(output_dir / "bland_altman_data.csv")\n'
            + inner_indent + '        ba_df.to_csv(ba_csv_path, index=False, encoding="utf-8-sig")\n'
            + inner_indent + 'except Exception as e:\n'
            + inner_indent + '    print("BA chart error: {}".format(e))\n'
            + indent + 'if predictor and predictor.is_fitted and predictor.train_history:\n'
            + inner_indent + '    th = predictor.train_history\n'
            + inner_indent + '    n = len(th.get("loss", th.get("accuracy", [])))\n'
            + inner_indent + '    if n > 0:\n'
            + inner_indent + '        try:\n'
            + inner_indent + '            hist_fig = Figure(figsize=(7, 4), dpi=150)\n'
            + inner_indent + '            ax_h = hist_fig.add_subplot(111)\n'
            + inner_indent + '            if "loss" in th:\n'
            + inner_indent + '                ax_h.plot(th["epoch"], th["loss"], color="#E64B35", lw=2, marker="o", markersize=4)\n'
            + inner_indent + '                ax_h.set_ylabel("Loss (RMSE)", fontsize=10)\n'
            + inner_indent + '                ax_h.set_title("Training Loss per Epoch (" + model_name + ")", fontsize=11)\n'
            + inner_indent + '            else:\n'
            + inner_indent + '                ax_h.plot(th["epoch"], th["accuracy"], color="#4DBBD5", lw=2, marker="o", markersize=4)\n'
            + inner_indent + '                ax_h.set_ylabel("Accuracy", fontsize=10)\n'
            + inner_indent + '                ax_h.set_title("Training Accuracy per Epoch (" + model_name + ")", fontsize=11)\n'
            + inner_indent + '            ax_h.set_xlabel("Epoch", fontsize=10)\n'
            + inner_indent + '            ax_h.grid(True, alpha=0.3)\n'
            + inner_indent + '            hist_fig.tight_layout()\n'
            + inner_indent + '            hist_png_path = str(output_dir / "training_history.png")\n'
            + inner_indent + '            hist_fig.savefig(hist_png_path, dpi=150, bbox_inches="tight")\n'
            + inner_indent + '            plt.close(hist_fig)\n'
            + inner_indent + '            pd.DataFrame(th).to_csv(str(output_dir / "training_history.csv"), index=False, encoding="utf-8-sig")\n'
            + inner_indent + '            hist_csv_path = str(output_dir / "training_history.csv")\n'
            + inner_indent + '        except Exception as e:\n'
            + inner_indent + '            print("History chart error: {}".format(e))\n'
        )
        # Find the zip_name line and insert before it
        for j in range(i-5, i):
            if 'zip_name = f"DeepPredict' in lines[j]:
                lines.insert(j, ba_gen)
                patches.append("4. BA + history generation")
                print("Patch 4: BA + history generation")
                break
        break

# Now add BA/history to ZIP writes
for i, l in enumerate(lines):
    if 'zf.write(output_dir / "metrics.json"' in l:
        new_line = (
            l.rstrip() + '\n'
            + '            if ba_csv_path:\n'
            + '                zf.write(ba_csv_path, arcname="bland_altman_data.csv")\n'
            + '            if hist_csv_path:\n'
            + '                zf.write(hist_csv_path, arcname="training_history.csv")\n'
        )
        lines[i] = new_line
        patches.append("5. BA/history in ZIP")
        print("Patch 5: BA/history in ZIP")
        break

new_content = ''.join(lines)
with open(r'C:\Users\XJH\DeepPredict\deeppredict_web.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"\nNew lines: {len(lines)}")

try:
    compile(new_content, 'dp.py', 'exec')
    print("Syntax: OK")
except SyntaxError as e:
    print(f"ERROR at line {e.lineno}: {e.msg}")
    ls = new_content.split('\n')
    for i in range(max(0, e.lineno-3), min(len(ls), e.lineno+2)):
        print(f"  {i+1}: {repr(ls[i][:100])}")

print(f"\nPatches applied: {patches}")
