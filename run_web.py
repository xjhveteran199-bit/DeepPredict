"""
DeepPredict Web - 启动脚本

同时运行：
  - FastAPI（首页 + API + Stripe Webhook）: http://localhost:8000
  - Gradio（预测工具 UI）: http://localhost:7860

Stripe Webhook 本地开发测试：
  stripe listen --forward-to localhost:8000/api/webhooks/stripe
"""

import os
import sys
import threading
import uvicorn
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))


def run_api():
    from web.backend.main import app
    print("=" * 55)
    print("  DeepPredict Web 服务")
    print("  访问首页: http://localhost:8000")
    print("  预测工具: http://localhost:8000/app")
    print("  API 文档: http://localhost:8000/docs")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


def run_gradio():
    from deeppredict_web import demo
    print("  Gradio UI:  http://localhost:7860")
    print("=" * 55)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


def main():
    # 检查 .env
    env_file = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_file):
        print("⚠️  未找到 .env 文件，Stripe 功能不可用")
        print("   复制 .env.example 为 .env 并填入 Stripe 密钥\n")

    missing = [v for v in ["STRIPE_SECRET_KEY", "STRIPE_PUBLISHABLE_KEY"]
               if not os.environ.get(v)]
    if missing:
        print(f"⚠️  缺少环境变量: {', '.join(missing)}（Stripe 不可用）\n")

    print("启动 DeepPredict Web 服务...\n")
    t = threading.Thread(target=run_api, daemon=True)
    t.start()

    import time; time.sleep(2)
    run_gradio()


if __name__ == "__main__":
    main()
