import sqlite3
from datetime import datetime, date
import os
import cv2
import urllib.request

class DatabaseManager:
    def __init__(self, db_path="attendance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                student_id TEXT UNIQUE NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                date DATE NOT NULL,
                time_in TEXT,
                time_out TEXT,
                status TEXT DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_student(self, name, student_id, email=None):
        """Add a new student to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO students (name, student_id, email)
                VALUES (?, ?, ?)
            ''', (name, student_id, email))
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error adding student: {e}")
            return False
    
    def mark_attendance(self, student_id):
        """Mark attendance for a student"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current date and time as strings
            current_date = date.today().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')
            
            # Check if student exists
            cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
            student = cursor.fetchone()
            
            if not student:
                print(f"Student with ID {student_id} not found")
                conn.close()
                return False
            
            # Check if attendance already exists for today
            cursor.execute('''
                SELECT id, time_in, time_out FROM attendance 
                WHERE student_id = ? AND date = ?
            ''', (student_id, current_date))
            
            existing_record = cursor.fetchone()
            
            if existing_record:
                record_id, time_in, time_out = existing_record
                if time_in and not time_out:
                    # Mark time out
                    cursor.execute('''
                        UPDATE attendance 
                        SET time_out = ? 
                        WHERE id = ?
                    ''', (current_time, record_id))
                    print(f"Marked time out for student {student_id} at {current_time}")
                else:
                    # Already has complete record for today
                    print(f"Student {student_id} already has complete attendance for today")
                    conn.close()
                    return False
            else:
                # Create new attendance record
                cursor.execute('''
                    INSERT INTO attendance (student_id, date, time_in, status)
                    VALUES (?, ?, ?, ?)
                ''', (student_id, current_date, current_time, 'Present'))
                print(f"Marked time in for student {student_id} at {current_time}")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def get_attendance_report(self, target_date=None):
        """Get attendance report for a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if target_date is None:
                target_date = date.today()
            
            # Convert date to string if it's a date object
            if isinstance(target_date, date):
                date_str = target_date.strftime('%Y-%m-%d')
            else:
                date_str = str(target_date)
            
            cursor.execute('''
                SELECT s.name, s.student_id, a.time_in, a.time_out, a.status
                FROM students s
                LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
                ORDER BY s.name
            ''', (date_str,))
            
            results = cursor.fetchall()
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting attendance report: {e}")
            return []
    
    def get_all_students(self):
        """Get all students from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT name, student_id, email FROM students ORDER BY name')
            results = cursor.fetchall()
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error getting students: {e}")
            return []

    @staticmethod
    def check_cascade_files():
        """Check if required cascade files exist and download if missing"""
        cascade_files = {
            'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
            'haarcascade_mouth.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml'  # Use smile cascade as alternative
        }
        
        opencv_data_path = cv2.data.haarcascades
        missing_files = []
        
        for cascade_file, url in cascade_files.items():
            file_path = os.path.join(opencv_data_path, cascade_file)
            if not os.path.exists(file_path):
                try:
                    print(f"Downloading {cascade_file}...")
                    urllib.request.urlretrieve(url, file_path)
                    print(f"Successfully downloaded {cascade_file}")
                except Exception as e:
                    print(f"Failed to download {cascade_file}: {e}")
                    missing_files.append(cascade_file)
        
        if missing_files:
            print(f"Missing cascade files: {missing_files}")
            print("Please ensure OpenCV is properly installed with: pip install opencv-python")
            return False
        return True
    
    @staticmethod
    def get_available_cascades():
        """Get list of available cascade files"""
        opencv_data_path = cv2.data.haarcascades
        available_cascades = {}
        
        cascade_mapping = {
            'face': 'haarcascade_frontalface_default.xml',
            'eye': 'haarcascade_eye.xml',
            'mouth': 'haarcascade_smile.xml'  # Use smile cascade instead of mouth
        }
        
        for name, filename in cascade_mapping.items():
            file_path = os.path.join(opencv_data_path, filename)
            if os.path.exists(file_path):
                available_cascades[name] = file_path
            else:
                print(f"Warning: {filename} not found at {file_path}")
        
        return available_cascades
    
    def drop_all_tables(self):
        """Drop all tables from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Drop each table
            for table in tables:
                if table[0] != 'sqlite_sequence':  # Skip system table
                    cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
                    print(f"Dropped table: {table[0]}")
            
            conn.commit()
            print("All tables dropped successfully")
            
        except Exception as e:
            print(f"Error dropping tables: {e}")
        finally:
            conn.close()
    
    def reset_database(self):
        """Reset the entire database"""
        self.drop_all_tables()
        self.init_database()
        print("Database reset complete")