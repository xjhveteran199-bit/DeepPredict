"""
DeepPredict Web - FastAPI 主入口
- /             → 首页（HTML）
- /app          → 预测工具（Gradio）
- /api/*        → REST API
- /webhooks/*   → Stripe Webhook
"""

import os
import sys
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.routing import APIRoute
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# 路径（相对于运行目录，而非当前文件位置）
BASE_DIR = os.path.abspath(os.getcwd())  # 运行目录
sys.path.insert(0, BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

from web.backend.models.database import SQLModel, engine, User
from web.backend.integrations.stripe_integration import (
    get_publishable_key, create_checkout_session,
    list_products, verify_webhook_signature,
)
from web.backend.routes.credits import (
    create_user, get_user_by_api_key, get_user_by_email,
    add_credits, get_user_transactions, get_user_stats,
)


# ====== 数据库初始化 ======

@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield


# ====== 主应用 ======

app = FastAPI(
    title="DeepPredict",
    description="零门槛深度学习预测工具",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件和模板
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "web", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "web", "templates"))


# ====== 依赖项 ======

def get_current_user(request: Request):
    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise HTTPException(status_code=401, detail="缺少 X-API-Key")
    user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return user


# ====== 前端页面路由 ======

@app.get("/")
async def serve_landing():
    return FileResponse(os.path.join(BASE_DIR, "web", "templates", "index.html"))


@app.get("/app")
async def serve_app():
    gradio_url = os.environ.get("GRADIO_URL", "http://localhost:7860")
    return RedirectResponse(url=gradio_url)


@app.get("/success")
async def payment_success():
    return templates.TemplateResponse("success.html", {"request": {}})


@app.get("/pricing")
async def serve_pricing():
    return FileResponse(os.path.join(BASE_DIR, "web", "templates", "pricing.html"))


# ====== API 路由 ======

@app.get("/api")
async def api_root():
    return {"ok": True, "version": "1.0.0", "docs": "/docs"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/products")
async def get_products():
    return list_products()


@app.get("/api/config")
async def get_config():
    return {
        "stripePublishableKey": get_publishable_key(),
        "freeCredits": 100,
    }


# --- Auth ---

@app.post("/api/auth/register")
async def register(email: str, name: str = ""):
    try:
        user = create_user(email, name)
        return {"ok": True, "user": {
            "email": user.email,
            "name": user.name,
            "credits": user.credits,
            "api_key": user.api_key,
        }}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def login(email: str):
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {
        "email": user.email,
        "name": user.name,
        "credits": user.credits,
        "api_key": user.api_key,
    }


# --- User ---

@app.get("/api/user/me")
async def get_me(user=Depends(get_current_user)):
    return {
        "email": user.email,
        "name": user.name,
        "credits": user.credits,
        "api_key": user.api_key,
        "created_at": user.created_at.isoformat(),
    }


@app.get("/api/user/transactions")
async def get_transactions(user=Depends(get_current_user)):
    return get_user_transactions(user.id)


@app.get("/api/user/stats")
async def get_stats(user=Depends(get_current_user)):
    return get_user_stats(user.id)


# --- Payment ---

@app.post("/api/checkout")
async def checkout(
    product_key: str,
    success_url: str = "https://deeppredict.ai/success",
    cancel_url: str = "https://deeppredict.ai/",
    user=Depends(get_current_user)
):
    try:
        url = create_checkout_session(
            user_api_key=user.api_key,
            product_key=product_key,
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"checkout_url": url}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Stripe Webhook ---

@app.post("/api/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = verify_webhook_signature(payload, sig)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook 验证失败: {e}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_api_key = session["metadata"].get("user_api_key", "")
        credits = int(session["metadata"].get("credits", "0"))
        product_key = session["metadata"].get("product_key", "")
        payment_id = session.get("payment_intent", "")

        if user_api_key and credits:
            user = get_user_by_api_key(user_api_key)
            if user:
                add_credits(
                    user.id, credits,
                    tx_type="purchase",
                    description=f"购买 {product_key}",
                    stripe_payment_id=payment_id,
                )
                if not user.stripe_customer_id and session.get("customer"):
                    from web.backend.models.database import Session as DbSession
                    with DbSession() as db:
                        db_user = db.query(User).filter(User.id == user.id).first()
                        if db_user:
                            db_user.stripe_customer_id = session["customer"]
                            db.commit()

    return {"received": True}


# --- Prediction API (示例) ---

@app.post("/api/predict")
async def predict(
    model: str,
    target_col: str,
    n_future: int = 30,
    user=Depends(get_current_user)
):
    CREDITS_COST = {"PatchTST": 20, "LSTM": 15, "GradientBoosting": 5}
    cost = CREDITS_COST.get(model, 10)
    if user.credits < cost:
        raise HTTPException(status_code=402, detail=f"积分不足：需要 {cost}，当前 {user.credits}")
    return {
        "ok": True,
        "credits_used": cost,
        "credits_remaining": user.credits - cost,
        "message": "预测功能接入中，请通过网页版使用完整功能"
    }


# ====== 错误处理 ======

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) if os.environ.get("DEBUG") else "Internal server error"}
    )
