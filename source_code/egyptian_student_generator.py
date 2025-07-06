import numpy as np
import pandas as pd
import random
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from university_curriculum_modeling import CurriculumGraph, create_sample_curriculum
import json

@dataclass
class Student:
    """Represents a university student with academic history and preferences"""
    student_id: str
    name: str
    current_term: int
    gpa: float
    completed_courses: Set[str]
    course_grades: Dict[str, float]  # Course -> Grade (0.0-4.0)
    failed_courses: Set[str]  # Courses that need retaking
    interests: List[str]  # Primary interest areas
    interest_weights: Dict[str, float]  # How much they prefer each area
    max_courses_per_term: int  # Personal course load limit
    graduation_goal_terms: int  # Target graduation timeline
    
    def __post_init__(self):
        if isinstance(self.completed_courses, list):
            self.completed_courses = set(self.completed_courses)
        if isinstance(self.failed_courses, list):
            self.failed_courses = set(self.failed_courses)

class StudentSimulator:
    """Generates and manages simulated student data"""
    
    def __init__(self, curriculum: CurriculumGraph):
        self.curriculum = curriculum
        self.students = []
        
        # Egyptian first names (mix of male and female names)
        self.first_names = [
            # Traditional Egyptian male names
            'Ahmed', 'Mohamed', 'Mahmoud', 'Omar', 'Ali', 'Hassan', 'Youssef', 'Khaled',
            'Amr', 'Tamer', 'Karim', 'Sherif', 'Hesham', 'Waleed', 'Tarek', 'Mostafa',
            'Adel', 'Sayed', 'Hany', 'Emad', 'Ashraf', 'Magdy', 'Ramy', 'Wael',
            
            # Traditional Egyptian female names  
            'Fatma', 'Aisha', 'Nour', 'Maryam', 'Sara', 'Rana', 'Yasmin', 'Dina',
            'Heba', 'Eman', 'Noha', 'Reem', 'Layla', 'Salma', 'Zeinab', 'Amira',
            'Nadine', 'Mai', 'Rania', 'Ghada', 'Hala', 'Nihal', 'Shahd', 'Nada',
            
            # Modern Egyptian names
            'Seif', 'Ziad', 'Nasser', 'Farid', 'Marwan', 'Samir', 'Fady', 'Kareem',
            'Jana', 'Malak', 'Lara', 'Nayra', 'Farida', 'Habiba', 'Rowan', 'Carmen'
        ]
        
        # Egyptian family names (surnames)
        self.last_names = [
            'El-Sayed', 'Mohamed', 'Ahmed', 'Ali', 'Hassan', 'Hussein', 'Mahmoud', 'Ibrahim',
            'Abdel-Rahman', 'Abdel-Aziz', 'Abdel-Hamid', 'Abdel-Fattah', 'Farouk', 'Mansour',
            'El-Shamy', 'El-Bendary', 'El-Masry', 'El-Naggar', 'El-Saghir', 'El-Dakrory',
            'Ramadan', 'Shahin', 'Gaber', 'Saleh', 'Youssef', 'Kamal', 'Ismail', 'Omar',
            'El-Hawary', 'El-Gazzar', 'El-Zaher', 'El-Khouly', 'El-Morsy', 'El-Sharkawy',
            'Abdo', 'Nasser', 'Farag', 'Rizk', 'Sabry', 'Zaki', 'Rashad', 'Shaker',
            'El-Tantawy', 'El-Mansoury', 'El-Ashmony', 'El-Desouky', 'El-Wardany', 'El-Sadat'
        ]
    
    def generate_student_name(self, student_id: int) -> str:
        """Generate a random but consistent student name"""
        random.seed(student_id)  # Consistent names for same ID
        first = random.choice(self.first_names)
        last = random.choice(self.last_names)
        return f"{first} {last}"
    
    def generate_interests(self, student_id: int) -> Tuple[List[str], Dict[str, float]]:
        """Generate student interest areas and weights"""
        random.seed(student_id + 1000)  # Different seed for interests
        
        all_areas = list(self.curriculum.interest_areas.keys())
        
        # Each student has 1-3 primary interests
        num_interests = random.randint(1, 3)
        primary_interests = random.sample(all_areas, num_interests)
        
        # Generate weights for all areas
        weights = {}
        for area in all_areas:
            if area in primary_interests:
                weights[area] = random.uniform(0.7, 1.0)  # High interest
            else:
                weights[area] = random.uniform(0.1, 0.4)  # Low interest
        
        return primary_interests, weights
    
    def simulate_academic_progression(self, student_id: int, target_term: int) -> Tuple[Set[str], Dict[str, float], Set[str], float]:
        """Simulate a student's academic progression through target_term"""
        random.seed(student_id + 2000)  # Different seed for progression
        
        completed_courses = set()
        course_grades = {}
        failed_courses = set()
        cumulative_gpa = 0.0
        total_credits = 0
        
        # Get student's interests to bias course selection
        interests, interest_weights = self.generate_interests(student_id)
        
        # Start with foundational courses
        foundational_courses = ['CS101', 'MATH101']
        
        for term in range(1, target_term + 1):
            # Determine course load for this term
            if term <= 2:
                max_courses = min(3, random.randint(2, 4))  # Start lighter
            else:
                max_courses = random.randint(3, 5)
            
            # Get eligible courses
            eligible = self.curriculum.get_eligible_courses(completed_courses)
            
            # Add failed courses back to eligible (retake policy)
            eligible.update(failed_courses)
            
            # Prioritize courses by interest
            course_preferences = []
            for course in eligible:
                if course in self.curriculum.course_info:
                    interest_area = self.curriculum.course_info[course]['interest_area']
                    preference_score = interest_weights.get(interest_area, 0.2)
                    
                    # Boost preference for foundational courses early on
                    if term <= 3 and course in foundational_courses:
                        preference_score += 0.3
                    
                    course_preferences.append((course, preference_score))
            
            # Sort by preference and select courses
            course_preferences.sort(key=lambda x: x[1], reverse=True)
            selected_courses = [course for course, _ in course_preferences[:max_courses]]
            
            # Simulate grades for selected courses
            for course in selected_courses:
                if course in self.curriculum.course_info:
                    difficulty = self.curriculum.course_info[course]['difficulty']
                    interest_area = self.curriculum.course_info[course]['interest_area']
                    
                    # Base performance influenced by difficulty and interest
                    base_performance = 3.0  # B average base
                    difficulty_penalty = (difficulty - 5.0) * 0.1
                    interest_boost = interest_weights.get(interest_area, 0.2) * 1.5
                    
                    # Add some randomness
                    random_factor = random.gauss(0, 0.5)
                    
                    grade = base_performance - difficulty_penalty + interest_boost + random_factor
                    grade = max(0.0, min(4.0, grade))  # Clamp to 0-4 range
                    
                    course_grades[course] = grade
                    
                    # Determine pass/fail (< 2.0 is fail)
                    if grade >= 2.0:
                        completed_courses.add(course)
                        failed_courses.discard(course)  # Remove from failed if retaken
                    else:
                        failed_courses.add(course)
                    
                    # Update GPA calculation
                    credits = self.curriculum.course_info[course]['credits']
                    total_credits += credits
                    cumulative_gpa = ((cumulative_gpa * (total_credits - credits)) + (grade * credits)) / total_credits
        
        return completed_courses, course_grades, failed_courses, cumulative_gpa
    
    def generate_students(self, num_students: int = 100) -> List[Student]:
        """Generate the specified number of students with realistic academic histories"""
        students = []
        
        for i in range(num_students):
            student_id = f"STU{i+1:03d}"
            name = self.generate_student_name(i)
            
            # Vary current term (1-8, representing freshman to senior year)
            current_term = random.randint(1, 8)
            
            # Generate interests
            interests, interest_weights = self.generate_interests(i)
            
            # Simulate academic progression
            completed_courses, course_grades, failed_courses, gpa = self.simulate_academic_progression(i, current_term)
            
            # Personal constraints
            max_courses_per_term = random.randint(3, 5)
            graduation_goal_terms = random.randint(max(current_term + 1, 6), 10)
            
            student = Student(
                student_id=student_id,
                name=name,
                current_term=current_term,
                gpa=round(gpa, 2),
                completed_courses=completed_courses,
                course_grades=course_grades,
                failed_courses=failed_courses,
                interests=interests,
                interest_weights=interest_weights,
                max_courses_per_term=max_courses_per_term,
                graduation_goal_terms=graduation_goal_terms
            )
            
            students.append(student)
        
        self.students = students
        return students
    
    def save_students(self, filepath: str):
        """Save student data to JSON file"""
        student_data = []
        for student in self.students:
            data = asdict(student)
            # Convert sets to lists for JSON serialization
            data['completed_courses'] = list(data['completed_courses'])
            data['failed_courses'] = list(data['failed_courses'])
            student_data.append(data)
        
        with open(filepath, 'w') as f:
            json.dump(student_data, f, indent=2)
    
    def load_students(self, filepath: str):
        """Load student data from JSON file"""
        with open(filepath, 'r') as f:
            student_data = json.load(f)
        
        students = []
        for data in student_data:
            # Convert lists back to sets
            data['completed_courses'] = set(data['completed_courses'])
            data['failed_courses'] = set(data['failed_courses'])
            students.append(Student(**data))
        
        self.students = students
        return students
    
    def get_student_statistics(self) -> Dict:
        """Generate statistics about the student population"""
        if not self.students:
            return {}
        
        stats = {
            'total_students': len(self.students),
            'average_gpa': np.mean([s.gpa for s in self.students]),
            'gpa_std': np.std([s.gpa for s in self.students]),
            'average_completed_courses': np.mean([len(s.completed_courses) for s in self.students]),
            'average_current_term': np.mean([s.current_term for s in self.students]),
            'students_with_failed_courses': len([s for s in self.students if s.failed_courses]),
            'interest_distribution': {},
            'term_distribution': {}
        }
        
        # Interest area distribution
        interest_counts = {}
        for student in self.students:
            for interest in student.interests:
                interest_counts[interest] = interest_counts.get(interest, 0) + 1
        stats['interest_distribution'] = interest_counts
        
        # Term distribution
        term_counts = {}
        for student in self.students:
            term = student.current_term
            term_counts[term] = term_counts.get(term, 0) + 1
        stats['term_distribution'] = term_counts
        
        return stats
    
    def get_students_by_interest(self, interest_area: str) -> List[Student]:
        """Get students with specific interest area"""
        return [s for s in self.students if interest_area in s.interests]
    
    def get_students_needing_course(self, course_id: str) -> List[Student]:
        """Get students who need to take a specific course"""
        return [s for s in self.students if course_id not in s.completed_courses 
                and course_id in self.curriculum.get_eligible_courses(s.completed_courses)]

def analyze_student_data(students: List[Student], curriculum: CurriculumGraph):
    """Perform comprehensive analysis of student data"""
    print("=== STUDENT POPULATION ANALYSIS ===\n")
    
    # Basic statistics
    gpas = [s.gpa for s in students]
    completed_counts = [len(s.completed_courses) for s in students]
    
    print(f"Total Students: {len(students)}")
    print(f"Average GPA: {np.mean(gpas):.2f} (±{np.std(gpas):.2f})")
    print(f"GPA Range: {min(gpas):.2f} - {max(gpas):.2f}")
    print(f"Average Completed Courses: {np.mean(completed_counts):.1f}")
    
    # Interest distribution
    interest_counts = {}
    for student in students:
        for interest in student.interests:
            interest_counts[interest] = interest_counts.get(interest, 0) + 1
    
    print(f"\n=== INTEREST DISTRIBUTION ===")
    for interest, count in sorted(interest_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(students)) * 100
        print(f"{interest}: {count} students ({percentage:.1f}%)")
    
    # Academic progress distribution
    term_counts = {}
    for student in students:
        term_counts[student.current_term] = term_counts.get(student.current_term, 0) + 1
    
    print(f"\n=== ACADEMIC PROGRESS ===")
    for term in sorted(term_counts.keys()):
        count = term_counts[term]
        percentage = (count / len(students)) * 100
        year = "Freshman" if term <= 2 else "Sophomore" if term <= 4 else "Junior" if term <= 6 else "Senior"
        print(f"Term {term} ({year}): {count} students ({percentage:.1f}%)")
    
    # Failed courses analysis
    students_with_failures = [s for s in students if s.failed_courses]
    print(f"\n=== ACADEMIC CHALLENGES ===")
    print(f"Students with failed courses: {len(students_with_failures)} ({len(students_with_failures)/len(students)*100:.1f}%)")
    
    if students_with_failures:
        all_failed_courses = []
        for student in students_with_failures:
            all_failed_courses.extend(list(student.failed_courses))
        
        from collections import Counter
        failure_counts = Counter(all_failed_courses)
        print("Most commonly failed courses:")
        for course, count in failure_counts.most_common(5):
            course_name = curriculum.course_info.get(course, {}).get('name', 'Unknown')
            print(f"  {course} ({course_name}): {count} failures")

if __name__ == "__main__":
    # Create curriculum and generate students
    curriculum = create_sample_curriculum()
    simulator = StudentSimulator(curriculum)
    
    print("Generating 100 students...")
    students = simulator.generate_students(100)
    
    # Save student data
    simulator.save_students('student_data.json')
    print("Student data saved to 'student_data.json'")
    
    # Analyze the generated data
    analyze_student_data(students, curriculum)
    
    # Example: Show details for first few students
    print(f"\n=== SAMPLE STUDENT PROFILES ===")
    for i, student in enumerate(students[:3]):
        print(f"\nStudent {i+1}: {student.name} ({student.student_id})")
        print(f"  Current Term: {student.current_term}")
        print(f"  GPA: {student.gpa}")
        print(f"  Interests: {', '.join(student.interests)}")
        print(f"  Completed Courses: {len(student.completed_courses)} courses")
        print(f"  Failed Courses: {len(student.failed_courses)} courses")
        
        # Show some completed courses
        if student.completed_courses:
            sample_courses = list(student.completed_courses)[:5]
            print(f"  Sample Completed: {', '.join(sample_courses)}") 