import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from model import Base, User, Food, Order

def init_db():
    # Create engine
    engine = create_engine('sqlite:///example.db')
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Add example data
    users = [
        User(name="Hans Müller", email="hans.mueller@example.com"),
        User(name="Anna Schmidt", email="anna.schmidt@example.com")
    ]
    session.add_all(users)
    session.commit()

    foods = [
        Food(name="Pizza Margherita", price=12.5),
        Food(name="Spaghetti Carbonara", price=15.0),
        Food(name="Lasagne", price=14.0),
    ]
    session.add_all(foods)
    session.commit()

    orders = [
        Order(food_id=1, user_id=1),
        Order(food_id=2, user_id=1),
        Order(food_id=3, user_id=2),
    ]
    session.add_all(orders)
    session.commit()

    session.close()
    print("Database created.")


if __name__ == "__main__":
    if not os.path.exists("example.db"):
        init_db()
    else:
        print("Database alredy exists!")