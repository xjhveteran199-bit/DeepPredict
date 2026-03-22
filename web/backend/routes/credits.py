"""
DeepPredict Web - 积分系统核心逻辑
"""

import secrets
from datetime import datetime
from .database import User, CreditTransaction, Session
from .integrations.stripe_integration import FREE_CREDITS_SIGNUP


def generate_api_key() -> str:
    """生成随机 API Key"""
    return f"dp_{secrets.token_urlsafe(32)}"


def create_user(email: str, name: str = "") -> User:
    """创建新用户（注册）"""
    with Session() as session:
        # 检查是否已存在
        existing = session.query(User).filter(User.email == email).first()
        if existing:
            return existing

        api_key = generate_api_key()
        user = User(
            email=email,
            name=name,
            api_key=api_key,
            credits=FREE_CREDITS_SIGNUP,  # 注册送积分
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        # 记录积分变动
        tx = CreditTransaction(
            user_id=user.id,
            amount=FREE_CREDITS_SIGNUP,
            balance_after=FREE_CREDITS_SIGNUP,
            type="signup",
            description="注册赠送"
        )
        session.add(tx)
        session.commit()
        return user


def get_user_by_api_key(api_key: str) -> User | None:
    with Session() as session:
        return session.query(User).filter(User.api_key == api_key, User.is_active == True).first()


def get_user_by_email(email: str) -> User | None:
    with Session() as session:
        return session.query(User).filter(User.email == email, User.is_active == True).first()


def add_credits(user_id: int, amount: int, tx_type: str, description: str, stripe_payment_id: str = "") -> int:
    """
    增加用户积分，返回变动后余额
    """
    with Session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        user.credits += amount
        user.updated_at = datetime.utcnow()
        session.commit()

        # 记录
        tx = CreditTransaction(
            user_id=user_id,
            amount=amount,
            balance_after=user.credits,
            type=tx_type,
            description=description,
            stripe_payment_id=stripe_payment_id
        )
        session.add(tx)
        session.commit()
        session.refresh(user)
        return user.credits


def deduct_credits(user_id: int, amount: int, description: str) -> int:
    """
    扣除用户积分，返回变动后余额
    """
    with Session() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")
        if user.credits < amount:
            raise ValueError(f"积分不足：需要 {amount}，当前 {user.credits}")

        user.credits -= amount
        user.updated_at = datetime.utcnow()
        session.commit()

        tx = CreditTransaction(
            user_id=user_id,
            amount=-amount,
            balance_after=user.credits,
            type="prediction",
            description=description
        )
        session.add(tx)
        session.commit()
        session.refresh(user)
        return user.credits


def get_user_transactions(user_id: int, limit: int = 50):
    """获取用户的积分变动记录"""
    with Session() as session:
        txs = session.query(CreditTransaction).filter(
            CreditTransaction.user_id == user_id
        ).order_by(CreditTransaction.created_at.desc()).limit(limit).all()
        return [
            {
                "id": t.id,
                "amount": t.amount,
                "balance_after": t.balance_after,
                "type": t.type,
                "description": t.description,
                "created_at": t.created_at.isoformat()
            }
            for t in txs
        ]


def get_user_stats(user_id: int) -> dict:
    """获取用户统计数据"""
    with Session() as session:
        from .database import Prediction
        total_predictions = session.query(Prediction).filter(
            Prediction.user_id == user_id
        ).count()
        successful_predictions = session.query(Prediction).filter(
            Prediction.user_id == user_id,
            Prediction.status == "success"
        ).count()
        total_credits_used = session.query(CreditTransaction).filter(
            CreditTransaction.user_id == user_id,
            CreditTransaction.amount < 0
        ).count()
        return {
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
        }
