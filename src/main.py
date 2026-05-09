import helpers
from db import DatabaseManager


result = helpers.validate("asd", "asd")
db = DatabaseManager("src/database.db")
db.setup_database()
print(result)