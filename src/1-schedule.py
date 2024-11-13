import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from unidecode import unidecode

# Selected buildings, floors and rooms. Empty list to explore all.
selected_buildings = [] # e.g. ['Torre Norte', 'Pavilhão Central]
selected_floors = []
selected_rooms = []

# Start and end date to create the schedule.
start_date = datetime.strptime("01/09/2021", "%d/%m/%Y")
end_date = datetime.strptime("01/07/2024", "%d/%m/%Y")

# Base path of the csv files.
base_path = './IST_Schedule/Alameda'

def sanitize_name(name):
    name = unidecode(name)
    name = re.sub(r'[^\w\s-]', '', name)
    return name

def create_general_csv(floor_rooms, floor_path, building_name, floor_name):
    building_name = sanitize_name(building_name).strip().replace(' ', '_')
    floor_name = sanitize_name(floor_name).strip().replace(' ', '_')

    if building_name != floor_name:
        csv_name = '_' + building_name + '_' + floor_name + '_info.csv'
    else:
        csv_name = '_' + building_name + '_info.csv'
    
    csv_path = os.path.join(floor_path, csv_name).replace('\\', '/')

    general_info = []
    for room in floor_rooms:
        general_info.append({
            "Space_Type": room['type'],
            "Space_Name": sanitize_name(room['name']).strip(),
            "Normal_Cap": room['capacity']['normal'],
            "Exam_Cap": room['capacity']['exam'],
            "Description": sanitize_name(room['description']).strip(),
            "Space_id": room['id']
        })
    
    df = pd.DataFrame(general_info)
    df.to_csv(csv_path, index=False)

def fetch_course_data(course_id):
    course_url = f"https://fenix.tecnico.ulisboa.pt/api/fenix/v1/courses/{course_id}"

    response = requests.get(course_url + '/schedule')
    if response.status_code == 200:
        course_schedule = response.json()
        if not course_schedule or 'shifts' not in course_schedule:
            print(f"Course ID {course_id} has no schedule data available.")
            return None, None
    else:
        print(f"Failed to retrieve course schedule data: {response.status_code} for course ID {course_id}")
        return None, None

    response = requests.get(course_url + '/groups')
    if response.status_code == 200:
        course_groups = response.json()
    else:
        print(f"Failed to retrieve course group data: {response.status_code} for course ID {course_id}")
        return course_schedule, None

    return course_schedule, course_groups

def create_room_schedule_csv(room_data, room_path, start_date, end_date):
    room_name = sanitize_name(room_data['name']).strip().replace(' ', '_')
    csv_filename = os.path.join(room_path, f"{room_name}.csv").replace('\\', '/')

    events = room_data['events']

    if events:
        room_schedule = []
        for event in events:
            event_start_time = datetime.strptime(event['start'], "%H:%M")
            event_end_time = datetime.strptime(event['end'], "%H:%M")
            event_date = datetime.strptime(event['day'], "%d/%m/%Y")

            event_start = datetime.combine(event_date.date(), event_start_time.time())
            event_end = datetime.combine(event_date.date(), event_end_time.time())

            # Filter by date range and event type
            if (start_date <= event_date <= end_date) and event['type'] == 'LESSON':
                course_schedule, course_groups = fetch_course_data(event['course']['id'])

                if course_schedule is None:
                    print(f"No valid schedule for course ID {event['course']['id']}. Skipping event.")
                    continue  # Skip this event if no schedule is available

                shifts = course_schedule['shifts']
                found = False

                for shift in shifts:
                    lessons = shift['lessons']
                    for lesson in lessons:
                        if lesson['room']:
                            if lesson['room']['id'] == room_data['id']:
                                lesson_start = datetime.strptime(lesson['start'], "%Y-%m-%d %H:%M:%S")
                                if lesson_start == event_start:
                                    shift_name = shift['name']
                                    enrolled_students = shift['occupation']['current']
                                    max_occupation = shift['occupation']['max']
                                    found = True
                                    break
                    if found:
                        break

                if not found:
                    shift_name = None
                    enrolled_students = None
                    max_occupation = None

                if course_groups and event['info'] == 'L':
                    for group in course_groups:
                        if 'lab' in unidecode(group['name'].lower()):
                            ideal_capacity = course_groups[0]['idealCapacity']
                            break
                        else:
                            ideal_capacity = None
                else:
                    ideal_capacity = None

                for hour in range(event_start.hour, event_end.hour + 1):
                    if hour == event_start.hour:
                        hour_ratio = 1 - (event_start.minute / 60)
                    elif hour == event_end.hour:
                        hour_ratio = (event_end.minute / 60)
                    else:
                        hour_ratio = 1

                    if hour != event_end.hour or event_end.minute > 0:
                        room_schedule.append({
                            "Year": event_date.year,
                            "Month": event_date.month,
                            "Day": event_date.day,
                            "Week_Num": event_date.isocalendar()[1],
                            "Week_Day": event_date.isocalendar()[2],
                            "Hour": hour,
                            "Hour_Ratio": hour_ratio,
                            "Event_Type": event['type'],
                            "Course_Acronym": event['course']['acronym'],
                            'Course_Name': unidecode(event['course']['name']),
                            "Academic_Term": unidecode(event['course']['academicTerm']),
                            "Lesson_Type": event['info'],
                            "Shift": shift_name,
                            "Enrolled_Students": enrolled_students,
                            "Max_Occupation": max_occupation,
                            "Lab_Group_Elements": ideal_capacity,
                            "Course_id": event['course']['id']
                        })

        if room_schedule:
            df = pd.DataFrame(room_schedule)
            df = df.sort_values(by=['Year', 'Week_Num', 'Week_Day', 'Hour'])
            df.to_csv(csv_filename, index=False)

def fetch_room_data(room_id, start_date, end_date):
    room_url = f"https://fenix.tecnico.ulisboa.pt/api/fenix/v1/spaces/{room_id}"
    response = requests.get(room_url)
    room_data = response.json()

    if response.status_code == 200:
        start_date = datetime.date(start_date)
        end_date = datetime.date(end_date)
        
        start_monday = start_date - timedelta(days=start_date.weekday())
        current_monday = start_monday

        while current_monday <= end_date:
            day = current_monday.day
            month = current_monday.month
            year = current_monday.year
            date = f"{day:02}/{month:02}/{year}"
            
            current_monday += timedelta(days=7)
            
            room_events_url = f"{room_url}?day={date}"
            response = requests.get(room_events_url)
            
            if response.status_code == 200:
                room_data_aux = response.json()
                room_data['events'] += room_data_aux['events']
            else:
                print(f"Failed to retrieve room data: {response.status_code} for room ID {room_id}")
                return None
        
        return room_data
    else:
        print(f"Failed to retrieve room data: {response.status_code} for room ID {room_id}")
        return None

def fetch_and_create_building_data(building_id, building_name, start_date, end_date):
    building_url = f"https://fenix.tecnico.ulisboa.pt/api/fenix/v1/spaces/{building_id}"
    response = requests.get(building_url)
    
    if response.status_code == 200:
        building_data = response.json()
        building_path = sanitize_name(building_name).strip().replace(' ', '_')
        building_path = os.path.join(base_path, building_path).replace('\\', '/')
        os.makedirs(building_path, exist_ok=True)

        if 'containedSpaces' in building_data:
            for floor in building_data['containedSpaces']:
                if not selected_floors or floor['name'] in selected_floors:
                    process_space(floor, building_path, building_name, start_date, end_date)  # Recursive function
    else:
        print(f"Failed to retrieve building data: {response.status_code} for building ID {building_id}")

def process_space(space, parent_path, building_name, start_date, end_date):
    space_name = space['name']
    space_path = os.path.join(parent_path, sanitize_name(space_name).strip().replace(' ', '_')).replace('\\', '/')
    os.makedirs(space_path, exist_ok=True)

    space_url = f"https://fenix.tecnico.ulisboa.pt/api/fenix/v1/spaces/{space['id']}"
    response = requests.get(space_url)

    if response.status_code == 200:
        space_data = response.json()
        
        if 'containedSpaces' in space_data:
            floor_rooms = []
            room_flag = False
            for subspace in space_data['containedSpaces']:
                if is_room(subspace):
                    room_flag = True
                    if subspace['name']:
                        print(f"Fetching data from building {building_name}, topLevelSpace {space_name} ({space['id']}), room {subspace['name']} ({subspace['id']}) ...")
                        if (not selected_rooms or subspace['name'] in selected_rooms) and subspace['name']:
                            room_data = fetch_room_data(subspace['id'], start_date, end_date)
                            if room_data:    
                                floor_rooms.append(room_data)
                                create_room_schedule_csv(room_data, space_path, start_date, end_date)
                else:
                    room_flag = False
                    process_space(subspace, space_path, building_name, start_date, end_date)

            if room_flag:
                create_general_csv(floor_rooms, space_path, building_name, space_name)
    else:
        print(f"Failed to retrieve space data: {response.status_code} for space ID {space['id']}")

def is_room(space):
    space_url = f"https://fenix.tecnico.ulisboa.pt/api/fenix/v1/spaces/{space['id']}"
    response = requests.get(space_url)
    if response.status_code == 200:
        space_data = response.json()
        return 'events' in space_data
    else:
        print(f"Failed to retrieve space data: {response.status_code} for space ID {space['id']}")
        return False

if __name__ == "__main__":
    url = "https://fenix.tecnico.ulisboa.pt/api/fenix/v1/spaces/2448131360897"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if not selected_buildings:
            for building in data['containedSpaces']:
                fetch_and_create_building_data(building['id'], building['name'], start_date, end_date)
        else:
            # sanitize selected building names and lowercase them
            selected_buildings = [sanitize_name(building).lower() for building in selected_buildings]
            for building in data['containedSpaces']:
                if sanitize_name(building['name']).lower() in selected_buildings:
                    fetch_and_create_building_data(building['id'], building['name'], start_date, end_date)
    else:
        print(f"Failed to retrieve data: {response.status_code}")
    
    print('\nDone!')