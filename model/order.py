from sqlalchemy import Column, Integer, ForeignKey
from .base import Base

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    food_id = Column(Integer, ForeignKey('foods.id'))
    user_id = Column(Integer, ForeignKey('users.id'))