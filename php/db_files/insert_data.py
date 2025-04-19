import pandas as pd
import mysql.connector
import logging
import os
from php.graphs.config import get_db_connection

# Configure logging
logging.basicConfig(
    filename='db_insert.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Path to the CSV file (relative to project/python/graphs/)
CSV_FILE_PATH = './dataset/heart.csv'

def create_table_if_not_exists(conn):
    """Creates the heart_disease_stats table if it doesn't exist."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS heart_disease_stats (
                age INT,
                sex INT,
                cp INT,
                trestbps INT,
                chol INT,
                fbs INT,
                restecg INT,
                thalach INT,
                exang INT,
                oldpeak FLOAT,
                slope INT,
                ca INT,
                thal INT,
                target INT
            )
        """)
        conn.commit()
        logging.info("Checked/created heart_disease_stats table")
        cursor.close()
    except mysql.connector.Error as e:
        logging.error(f"Error creating table: {e}")
        raise

def insert_data(df):
    """Inserts the DataFrame into the heart_disease_stats table."""
    try:
        with get_db_connection() as conn:
            # Create table if it doesn't exist
            create_table_if_not_exists(conn)
            
            cursor = conn.cursor()
            # Clear existing data (optional, comment out to append)
            cursor.execute("TRUNCATE TABLE heart_disease_stats")
            
            # Insert rows
            insert_query = """
                INSERT INTO heart_disease_stats
                (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            rows_inserted = 0
            for _, row in df.iterrows():
                cursor.execute(insert_query, (
                    int(row['age']),
                    int(row['sex']),
                    int(row['cp']),
                    int(row['trestbps']),
                    int(row['chol']),
                    int(row['fbs']),
                    int(row['restecg']),
                    int(row['thalach']),
                    int(row['exang']),
                    float(row['oldpeak']),
                    int(row['slope']),
                    int(row['ca']),
                    int(row['thal']),
                    int(row['target'])
                ))
                rows_inserted += 1
            
            conn.commit()
            logging.info(f"Inserted {rows_inserted} rows into heart_disease_stats")
            cursor.close()
    except mysql.connector.Error as e:
        logging.error(f"Error inserting data: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def main():
    """Main function to read CSV file and insert into database."""
    try:
        # Check if CSV file exists
        if not os.path.exists(CSV_FILE_PATH):
            logging.error(f"CSV file not found: {CSV_FILE_PATH}")
            raise FileNotFoundError(f"CSV file not found: {CSV_FILE_PATH}")
        
        # Read CSV file into DataFrame
        df = pd.read_csv(CSV_FILE_PATH)
        logging.info(f"Read {len(df)} rows from {CSV_FILE_PATH}")
        
        # Insert data into database
        insert_data(df)
        print(f"Successfully inserted {len(df)} rows into heart_disease_stats")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Main function error: {e}")
        raise

if __name__ == "__main__":
    main()
