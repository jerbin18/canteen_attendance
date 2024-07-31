import pandas as pd
import numpy as np
import cv2
import face_recognition
import tkinter as tk
from tkinter import ttk
import datetime
import sqlite3
import pytz

menus = {
    "breakfast": ["Pancakes - $5", "Coffee - $2", "Omelette - $4"],
    "lunch": ["Burger - $5", "Pizza - $8", "Salad - $4"],
    "snacks": ["Chips - $2", "Cookies - $3", "Juice - $2"]
}

def initialize_database():
    """Initialize the SQLite database and create the table."""
    conn = sqlite3.connect('canteen.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS selections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            dish TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def load_face_encodings_from_csv(file_path):
    """Load face encodings and labels from a CSV file."""
    df = pd.read_csv(file_path)
    encodings = df.iloc[:, 1:].values  # All columns except the first one (label)
    labels = df['label'].values  # The first column
    return encodings, labels

def get_user_name(encoding):
    """Identify the user based on face encoding."""
    matches = face_recognition.compare_faces(known_encodings, encoding)
    distances = face_recognition.face_distance(known_encodings, encoding)
    
    if matches:
        best_match_index = np.argmin(distances)
        if matches[best_match_index]:
            return known_labels[best_match_index]
    return "Unknown"

def get_time_based_menu():
    """Get the menu based on the current time."""
    current_hour = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).hour
    if 6 <= current_hour < 11:
        return menus["breakfast"]
    elif 11 <= current_hour < 16:
        return menus["lunch"]
    else:
        return menus["snacks"]

def save_to_database(user_name, dish):
    """Save the user's dish selection to the SQLite database."""
    conn = sqlite3.connect('canteen.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO selections (user_name, dish) VALUES (?, ?)
    ''', (user_name, dish))
    
    conn.commit()
    conn.close()

def display_menu(user_name):
    """Create a GUI window to display the menu for a recognized user and handle dish selection."""
    menu_items = get_time_based_menu()
    
    # Get the current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

    # Create GUI window
    menu_window = tk.Tk()
    menu_window.title(f"Menu for {user_name}")

    label = ttk.Label(menu_window, text=f"Menu for {user_name}", font=("Helvetica", 16))
    label.pack(pady=10)

    time_label = ttk.Label(menu_window, text=f"Current Time: {current_time}", font=("Helvetica", 12))
    time_label.pack(pady=5)

    selected_dish = tk.StringVar(menu_window)
    selected_dish.set(menu_items[0])  # Default value

    dish_menu = ttk.OptionMenu(menu_window, selected_dish, *menu_items)
    dish_menu.pack(pady=10)

    def confirm_selection():
        """Handle dish selection, print it, and save it to the database."""
        selected = selected_dish.get()
        print(f"Selected dish: {selected}")
        save_to_database(user_name, selected)
        menu_window.destroy()

    confirm_button = ttk.Button(menu_window, text="Confirm", command=confirm_selection)
    confirm_button.pack(pady=10)

    menu_window.mainloop()

def main():
    # Initialize the database
    initialize_database()

    # Load known face encodings and labels from CSV
    try:
        global known_encodings, known_labels
        known_encodings, known_labels = load_face_encodings_from_csv('data/features_all.csv')

        # Convert encodings to numpy arrays
        known_encodings = np.array(known_encodings)
        
        print(f"Known Encodings Shape: {known_encodings.shape}")
        print(f"Known Labels: {known_labels}")

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    cap = cv2.VideoCapture(0)  # Open the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        rgb_frame = frame[:, :, ::-1]

        # Find all face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            user_name = get_user_name(face_encoding)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, user_name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if user_name != "Unknown":
                # Display the menu for the recognized user
                display_menu(user_name)

        # Display the resulting frame
        cv2.imshow('Face Recognition Canteen Management System', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
