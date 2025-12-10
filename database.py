import sqlite3
from datetime import datetime

DB_NAME = "anpr.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            date_time TEXT NOT NULL,
            camera_location TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_plate(plate_number, camera_location="Gate 1"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO vehicle_log (plate_number, date_time, camera_location)
        VALUES (?, ?, ?)
    """, (plate_number, dt, camera_location))
    conn.commit()
    conn.close()

def get_all_logs(limit=10):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, plate_number, date_time, camera_location
        FROM vehicle_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_log(row_id):
    """Delete a single log entry by id."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM vehicle_log WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

