from db import DatabaseManager


db = DatabaseManager("src/database.db")
db.setup_database()