"""
DeepPredict Web - Stripe 集成
使用 Stripe Checkout（托管页面，最简单，无需 PCI 合规）
"""

import os
import stripe
from typing import Optional

# ====== 配置 ======
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")

# ====== 产品/价格配置 ======
# 在 Stripe Dashboard 创建产品后填入 Price ID，或通过 API 创建
PRODUCTS = {
    "credits_100": {
        "name": "100 积分",
        "credits": 100,
        "price_usd": 1.0,
        "stripe_price_id": os.environ.get("STRIPE_PRICE_100", ""),  # price_xxx
        "description": "永久有效，适合个人学习"
    },
    "credits_500": {
        "name": "500 积分",
        "credits": 500,
        "price_usd": 4.0,
        "stripe_price_id": os.environ.get("STRIPE_PRICE_500", ""),
        "description": "9折优惠，适合研究用途",
        "badge": "最受欢迎"
    },
    "credits_1000": {
        "name": "1000 积分",
        "credits": 1000,
        "price_usd": 7.0,
        "stripe_price_id": os.environ.get("STRIPE_PRICE_1000", ""),
        "description": "7折大幅优惠，适合频繁使用者",
        "badge": "超值"
    },
}

FREE_CREDITS_SIGNUP = 100  # 注册送积分


def get_publishable_key() -> str:
    return STRIPE_PUBLISHABLE_KEY


def create_checkout_session(
    user_api_key: str,
    product_key: str,
    success_url: str,
    cancel_url: str
) -> str:
    """
    创建 Stripe Checkout 会话，返回 session_id（重定向到 Stripe 托管页）
    """
    product = PRODUCTS.get(product_key)
    if not product:
        raise ValueError(f"Unknown product: {product_key}")

    if not product["stripe_price_id"]:
        raise ValueError(f"Product {product_key} not configured: missing stripe_price_id")

    # 检查是否已有 Stripe customer，没有则创建
    from web.backend.models.database import User, Session
    with Session() as session:
        user = session.query(User).filter(User.api_key == user_api_key).first()
        if not user:
            raise ValueError("User not found")

        extra_kwargs = {}
        if user.stripe_customer_id:
            extra_kwargs["customer"] = user.stripe_customer_id
        else:
            extra_kwargs["customer_email"] = user.email

        checkout = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],
            line_items=[{
                "price": product["stripe_price_id"],
                "quantity": 1,
            }],
            metadata={
                "user_api_key": user_api_key,
                "product_key": product_key,
                "credits": str(product["credits"]),
            },
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            **extra_kwargs
        )
        return checkout.url


def verify_webhook_signature(payload: bytes, sig: str) -> dict:
    """验证 Stripe webhook 签名并返回事件"""
    return stripe.Webhook.construct_object(
        payload, sig, STRIPE_WEBHOOK_SECRET
    )


def create_stripe_customer(email: str, name: str = "") -> str:
    """创建 Stripe Customer"""
    customer = stripe.Customer.create(
        email=email,
        name=name,
    )
    return customer.id


def list_products() -> dict:
    """返回可购买的产品列表（不含 stripe_price_id 敏感字段）"""
    return {
        key: {
            "name": p["name"],
            "credits": p["credits"],
            "price_usd": p["price_usd"],
            "description": p["description"],
            "badge": p.get("badge", ""),
            "configured": bool(p["stripe_price_id"]),
        }
        for key, p in PRODUCTS.items()
    }
