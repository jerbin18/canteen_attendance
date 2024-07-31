from flask import Flask, render_template, request, Response
import sqlite3
from datetime import datetime
import csv
import io
import pytz  # For time zone conversion

app = Flask(__name__)

# Define timezone
IST = pytz.timezone('Asia/Kolkata')

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('canteen.db')  # Ensure this matches the face recognition DB
    cursor = conn.cursor()

    cursor.execute("SELECT user_name, dish, timestamp FROM selections WHERE DATE(timestamp) = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    # Convert UTC to IST for display
    attendance_data_ist = []
    for record in attendance_data:
        user_name, dish, timestamp_utc = record
        # Convert UTC timestamp to datetime object
        timestamp_utc = datetime.strptime(timestamp_utc, '%Y-%m-%d %H:%M:%S')
        timestamp_utc = pytz.utc.localize(timestamp_utc)
        # Convert to IST
        timestamp_ist = timestamp_utc.astimezone(IST).strftime('%Y-%m-%d %H:%M:%S')
        attendance_data_ist.append((user_name, dish, timestamp_ist))

    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data_ist)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('canteen.db')  # Ensure this matches the face recognition DB
    cursor = conn.cursor()

    cursor.execute("SELECT user_name, dish, timestamp FROM selections WHERE DATE(timestamp) = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return "No canteen data available for the selected date."

    # Convert UTC to IST for CSV download
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['User Name', 'Dish', 'Timestamp'])  # Write header
    
    for record in attendance_data:
        user_name, dish, timestamp_utc = record
        # Convert UTC timestamp to datetime object
        timestamp_utc = datetime.strptime(timestamp_utc, '%Y-%m-%d %H:%M:%S')
        timestamp_utc = pytz.utc.localize(timestamp_utc)
        # Convert to IST
        timestamp_ist = timestamp_utc.astimezone(IST).strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([user_name, dish, timestamp_ist])

    # Set up response to serve the CSV file
    output.seek(0)
    return Response(output, mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=attendance.csv'})

if __name__ == '__main__':
    app.run(debug=True)
