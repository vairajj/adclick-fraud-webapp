from database import DATABASE_PATH
from sqlalchemy.orm import Session
import models
from database import engine

print("Database path:", DATABASE_PATH)

with Session(engine) as db:
    count = db.query(models.Prediction).count()
    print("Total stored predictions:", count)
