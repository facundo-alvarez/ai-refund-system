from validation import ProductValidator
from db import DatabaseManager


result = ProductValidator.validate("asd", "asd")
db = DatabaseManager("src/database.db")
db.setup_database()
print(result)