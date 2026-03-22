"""
DeepPredict Web - 数据库模型
SQLite + SQLModel（轻量，无需独立数据库）
"""

import os
from sqlmodel import SQLModel, Field, Relationship, Session as SQLModelSession, create_engine
from datetime import datetime
from typing import Optional

# 数据库引擎（SQLite，文件存储）
DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///./deeppredict.db")
connect_args = {"check_same_thread": False} if "sqlite" in DB_PATH else {}
engine = create_engine(DB_PATH, echo=False, connect_args=connect_args)


def get_session():
    with SQLModelSession(engine) as session:
        yield session


class Session:
    """简易上下文管理器 session"""
    def __enter__(self):
        self.session = SQLModelSession(engine)
        return self.session
    def __exit__(self, *args):
        self.session.close()


class User(SQLModel, table=True):
    """用户表"""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str = Field(default="")
    api_key: str = Field(unique=True, index=True)  # 用户API Key
    credits: int = Field(default=100)  # 积分余额
    stripe_customer_id: Optional[str] = Field(default=None)  # Stripe 客户ID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

    # 关联
    transactions: list["CreditTransaction"] = Relationship(back_populates="user")
    predictions: list["Prediction"] = Relationship(back_populates="user")


class CreditTransaction(SQLModel, table=True):
    """积分变动记录"""
    __tablename__ = "credit_transactions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    amount: int  # 正数=获得，负数=消耗
    balance_after: int  # 变动后余额
    type: str = Field(default="")  # purchase | prediction | refund | bonus | signup
    description: str = Field(default="")
    stripe_payment_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship(back_populates="transactions")


class Prediction(SQLModel, table=True):
    """预测记录"""
    __tablename__ = "predictions"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    model_name: str  # PatchTST / LSTM / GradientBoosting
    target_col: str  # 目标列
    data_rows: int  # 数据行数
    credits_used: int = Field(default=10)  # 本次消耗积分
    status: str = Field(default="success")  # success / failed
    error_msg: Optional[str] = Field(default=None)
    result_summary: Optional[str] = Field(default=None)  # R² / RMSE 等
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship(back_populates="predictions")


class StripePrice(SQLModel, table=True):
    """Stripe 价格配置"""
    __tablename__ = "stripe_prices"

    id: Optional[int] = Field(default=None, primary_key=True)
    stripe_price_id: str = Field(unique=True)  # Stripe 后台的 Price ID
    name: str  # 如 "100 积分"
    credits: int  # 对应积分数量
    price_usd: float  # 价格（美元）
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
