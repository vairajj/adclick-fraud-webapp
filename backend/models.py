from sqlalchemy import Column, DateTime, Float, Integer, String
from datetime import datetime

from database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    click_time = Column(DateTime, default=datetime.now)

    device_type = Column(String(50))
    browser = Column(String(50))
    operating_system = Column(String(50))
    ad_position = Column(String(50))
    device_ip_reputation = Column(String(50))

    click_duration = Column(Float)
    scroll_depth = Column(Float)
    mouse_movement = Column(Float)
    keystrokes_detected = Column(Integer)

    click_frequency = Column(Integer)
    time_since_last_click = Column(Float)

    VPN_usage = Column(Integer)
    proxy_usage = Column(Integer)

    probability = Column(Float)
    label = Column(String(50))
    risk = Column(String(20))


class C2IDTLPrediction(Base):
    __tablename__ = "c2idtl_predictions"

    id = Column(Integer, primary_key=True, index=True)
    click_time = Column(DateTime, nullable=False)

    ip = Column(Integer, nullable=False)
    app = Column(Integer, nullable=False)
    device = Column(Integer, nullable=False)
    os = Column(Integer, nullable=False)
    channel = Column(Integer, nullable=False)

    probability = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)

