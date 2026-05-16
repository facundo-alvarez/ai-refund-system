import sqlite3

class DatabaseManager:
    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON;")

    def setup_database(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS statuses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS returns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL UNIQUE,
                category_id INTEGER,
                status_id INTEGER DEFAULT 1,
                predicted_category_id INTEGER,
                confidence  REAL,
                image_path TEXT,
                FOREIGN KEY (category_id) REFERENCES categories (id),
                FOREIGN KEY (status_id) REFERENCES statuses (id),
                FOREIGN KEY (predicted_category_id) REFERENCES categories (id)
            )
        ''')
        
        self._seed_data()
        self.connection.commit()

    def _seed_data(self):
        """Populates lookup tables if empty."""
        categories = [
            ('Dress',), ('Hat',), ('Longsleeve',), ('Outwear',), 
            ('Pants',), ('Shirt',), ('Shoes',), ('Shorts',), 
            ('Skirt',), ('T-Shirt',)
        ]
        statuses = [('New',), ('Processed',), ('Flagged',)]
        
        self.cursor.executemany('INSERT OR IGNORE INTO categories (name) VALUES (?)', categories)
        self.cursor.executemany('INSERT OR IGNORE INTO statuses (name) VALUES (?)', statuses)

    def close(self):
        self.connection.close()


if __name__ == "__main__":
    db = DatabaseManager("database.db")
    db.setup_database()