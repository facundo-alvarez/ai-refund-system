import os
import sqlite3

class DatabaseInitializer:
    """
    Handles creation and initialization of the SQLite database schema
    and seed data required by the application.

    Attributes:
        connection (sqlite3.Connection): Active database connection.
        cursor (sqlite3.Cursor): Cursor used for executing SQL queries.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize database connection and enable foreign key support.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON;")

    def setup_database(self):
        """
        Create tables if they do not already exist and populate
        initial seed data.

        Tables created:
            - categories: Clothes categories
            - statuses: Processing status values
            - returns: Main table storing return records and image metadata
        """

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
        
        self.__seed_data()
        self.connection.commit()

    def __seed_data(self):
        """
        Insert initial values into the tables if they are empty.

        Populates:
            - categories table with clothing types
            - statuses table with workflow states
        """

        categories = [
            ('Dress',), ('Hat',), ('Longsleeve',), ('Outwear',), 
            ('Pants',), ('Shirt',), ('Shoes',), ('Shorts',), 
            ('Skirt',), ('T-Shirt',)
        ]
        statuses = [('New',), ('Processed',), ('Flagged',)]
        
        self.cursor.executemany('INSERT OR IGNORE INTO categories (name) VALUES (?)', categories)
        self.cursor.executemany('INSERT OR IGNORE INTO statuses (name) VALUES (?)', statuses)

    def close(self):
        """
        Close the database connection and release resources.
        """
        self.connection.close()


if __name__ == "__main__":
    db = DatabaseInitializer("instance/database.db")
    db.setup_database()
    db.close()