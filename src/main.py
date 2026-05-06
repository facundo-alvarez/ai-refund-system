from validation import ProductValidator
from db import DatabaseManager


result = ProductValidator.validate("asd", "asd")
db = DatabaseManager("shop.db")
db.setup_database()
print(result)