import random
import csv
from datetime import datetime, timedelta

# List of names and a mapping to IDs
names = ["Messi", "Ronaldo", "Shaad", "Sunil", "Sparsh", "Aryan", "Zaid"]
name_to_id = {name: i + 1 for i, name in enumerate(names)}

# Function to generate random time around 9 AM
def generate_random_time_around_9am():
    base_time = datetime.strptime("09:00:00", "%H:%M:%S")
    random_offset = timedelta(minutes=random.randint(-30, 30), seconds=random.randint(0, 59))
    random_time = base_time + random_offset
    return random_time.strftime("%H:%M-%S")

# Generate attendance records
attendance_records = []

# Set start date to 7 days ago
start_date = datetime.now() - timedelta(days=7)

attendance_probability = 0.8  # 80% chance of a student being present

for day in range(60):  # Generate records for the past 7 days
    current_date = start_date + timedelta(days=day)
    date_str = current_date.strftime("%d-%m-%Y")
    daily_attendance = set()  # Track who has already been marked for the day
    for name in names:
        if random.random() < attendance_probability:
            id_ = name_to_id[name]
            time = generate_random_time_around_9am()
            attendance_records.append([name, id_, time, date_str])
            daily_attendance.add(name)

# Write to CSV file
with open('Attendance/Attendance_.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["NAME", "ID", "TIME", "DATE"])
    writer.writerows(attendance_records)

print("Attendance records generated and saved to 'attendance_records.csv'")
