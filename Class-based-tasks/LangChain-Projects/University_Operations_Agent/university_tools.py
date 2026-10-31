
from langchain.tools import tool


# ---------------------------------
# Sample University Database
# ---------------------------------

STUDENTS = {
    "001": {
        "name": "Ali",
        "department": "Computer Science",
        "semester": 6,
        "marks": {
            "AI": 85,
            "Python": 90,
            "Database": 78
        },
        "fee_paid": True
    },
    "002": {
        "name": "Sara",
        "department": "Software Engineering",
        "semester": 4,
        "marks": {
            "AI": 88,
            "Python": 82,
            "Database": 91
        },
        "fee_paid": False
    }
}

SCHEDULES = {
    "AI-301": {
        "course": "Artificial Intelligence",
        "day": "Monday",
        "time": "10:00 AM",
        "room": "Lab 2"
    },
    "CS-201": {
        "course": "Programming Fundamentals",
        "day": "Wednesday",
        "time": "12:00 PM",
        "room": "Room 15"
    }
}


# ---------------------------------
# Tool 1: Get Student Profile
# ---------------------------------

@tool
def get_student_profile(student_id: str) -> str:
    """Get a student's name, department, and semester using their student ID."""

    student = STUDENTS.get(student_id)

    if student is None:
        return f"ERROR: Student ID {student_id} was not found."

    return (
        f"Student ID: {student_id}\n"
        f"Name: {student['name']}\n"
        f"Department: {student['department']}\n"
        f"Semester: {student['semester']}"
    )


# ---------------------------------
# Tool 2: Get Student Marks
# ---------------------------------


@tool
def get_student_marks(student_id: str, course: str = "") -> str:
    """Get student marks by ID and course. If course is missing, first verify the student ID."""

    student = STUDENTS.get(student_id)

    # Check student ID first
    if student is None:
        return f"ERROR: Student ID {student_id} was not found."

    # If course is missing, show available courses
    if not course.strip():
        available_courses = ", ".join(student["marks"].keys())
        return (
            f"Student {student_id} exists. "
            f"Available courses: {available_courses}. "
            "Please specify a course."
        )

    marks = student["marks"].get(course.upper())

    if marks is None:
        return f"ERROR: Marks for course {course} are not available."

    return (
        f"Student: {student['name']}\n"
        f"Course: {course.upper()}\n"
        f"Marks: {marks}"
    )


# ---------------------------------
# Tool 3: Get Course Schedule
# ---------------------------------

@tool
def get_course_schedule(course_code: str) -> str:
    """Get the day, time, and room of a university course using its course code."""

    schedule = SCHEDULES.get(course_code.upper())

    if schedule is None:
        return f"ERROR: Schedule for course {course_code} is not available."

    return (
        f"Course: {schedule['course']}\n"
        f"Day: {schedule['day']}\n"
        f"Time: {schedule['time']}\n"
        f"Room: {schedule['room']}"
    )


# ---------------------------------
# Tool 4: Calculate Percentage
# ---------------------------------

@tool
def calculate_percentage(obtained: float, total: float) -> str:
    """Calculate percentage from obtained marks and total marks."""

    if total <= 0:
        return "ERROR: Total marks must be greater than zero."

    if obtained < 0 or obtained > total:
        return "ERROR: Obtained marks must be between zero and total marks."

    percentage = (obtained / total) * 100

    return f"Percentage: {percentage:.2f}%"


# ---------------------------------
# Tool 5: Get Fee Status
# ---------------------------------

@tool
def get_fee_status(student_id: str) -> str:
    """Check whether a student has paid their university fee."""

    student = STUDENTS.get(student_id)

    if student is None:
        return f"ERROR: Student ID {student_id} was not found."

    if student["fee_paid"]:
        return f"{student['name']} has paid the university fee."

    return f"{student['name']} has not paid the university fee."
