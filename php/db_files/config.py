import mysql.connector
import os

# Optionally, load credentials from environment variables
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', 'hightech2024')
DB_NAME = os.getenv('DB_NAME', 'cardiology')


def get_db_connection():
    """
    Returns a MySQL connection using mysql-connector-python.
    """
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )