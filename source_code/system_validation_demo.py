#!/usr/bin/env python3
"""
AI Academic Advisor - Quick Demo
Professional demonstration of the academic advisor system
48 Hours Challenge Implementation
"""

import json
from datetime import datetime
import sys
from typing import List, Dict, Any

# Import our modules
from university_curriculum_modeling import create_sample_curriculum
from egyptian_student_generator import StudentSimulator

def display_header():
    """Display professional header for the demo"""
    print("AI ACADEMIC ADVISOR - QUICK DEMONSTRATION")
    print("=" * 55)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def demonstrate_curriculum_creation():
    """Create and display curriculum information"""
    print("STEP 1: Creating Curriculum Graph")
    print("-" * 35)
    
    curriculum = create_sample_curriculum()
    num_courses = len(list(curriculum.graph.nodes()))
    num_prerequisites = len(list(curriculum.graph.edges()))
    
    print(f"Created curriculum with {num_courses} courses")
    print(f"Added {num_prerequisites} prerequisite relationships")
    print()
    
    # Display sample courses
    print("Sample Courses:")
    sample_courses = list(curriculum.graph.nodes())[:5]
    for course_id in sample_courses:
        course_info = curriculum.course_info[course_id]
        print(f"  - {course_id}: {course_info['name']} ({course_info['interest_area']})")
    print()
    
    return curriculum, sample_courses

def demonstrate_student_generation(curriculum):
    """Generate and analyze student population"""
    print("STEP 2: Generating Student Population")
    print("-" * 38)
    
    student_sim = StudentSimulator(curriculum)
    students = student_sim.generate_students(100)
    
    print(f"Generated {len(students)} students")
    
    # Calculate statistics
    total_gpa = sum(s.gpa for s in students)
    avg_gpa = total_gpa / len(students)
    avg_courses = sum(len(s.completed_courses) for s in students) / len(students)
    failed_students = sum(1 for s in students if len(s.failed_courses) > 0)
    
    print()
    print("Student Population Statistics:")
    print(f"  - Average GPA: {avg_gpa:.2f}")
    print(f"  - Average Completed Courses: {avg_courses:.1f}")
    print(f"  - Students with Course Failures: {failed_students}")
    print()
    
    return students

def demonstrate_recommendations(curriculum, students: List):
    """Show course recommendations for sample students"""
    print("STEP 3: Course Recommendation Examples")
    print("-" * 40)
    
    sample_students = students[:3]
    recommendation_results = []
    
    for i, student in enumerate(sample_students, 1):
        print(f"Student {i}: {student.student_id}")
        print(f"  Current Term: {student.current_term}")
        print(f"  GPA: {student.gpa:.2f}")
        print(f"  Interests: {', '.join(student.interest_weights.keys())}")
        print(f"  Completed Courses: {len(student.completed_courses)}")
        
        # Get eligible courses
        eligible_courses = curriculum.get_eligible_courses(student.completed_courses)
        print(f"  Eligible Courses: {len(eligible_courses)}")
        
        # Generate interest-based recommendations
        recommendations = generate_basic_recommendations(student, eligible_courses, curriculum)
        
        print("  Recommendations:")
        if recommendations:
            for course_id, course_info in recommendations:
                print(f"    - {course_id}: {course_info['name']} ({course_info['interest_area']})")
        else:
            top_interest = max(student.interest_weights.keys(), 
                             key=lambda x: student.interest_weights[x])
            print(f"    - No courses available for {top_interest} interest")
        
        recommendation_results.append({
            'student_id': student.student_id,
            'recommendations': len(recommendations),
            'eligible_courses': len(eligible_courses)
        })
        print()
    
    return recommendation_results

def generate_basic_recommendations(student, eligible_courses, curriculum):
    """Generate basic interest-aligned recommendations"""
    if not eligible_courses:
        return []
    
    # Find student's top interest
    top_interest = max(student.interest_weights.keys(), 
                     key=lambda x: student.interest_weights[x])
    
    # Find matching courses
    recommendations = []
    for course_id in eligible_courses:
        course_info = curriculum.course_info[course_id]
        if course_info['interest_area'] == top_interest:
            recommendations.append((course_id, course_info))
    
    return recommendations[:3]  # Return top 3 recommendations

def validate_system(curriculum, students):
    """Perform system validation checks"""
    print("STEP 4: System Validation")
    print("-" * 27)
    
    # Test prerequisite validation
    test_completed = {'CS101', 'MATH101'}
    test_eligible = curriculum.get_eligible_courses(test_completed)
    prerequisite_check = 'CS102' in test_eligible
    
    # Test course load limits
    course_load_check = all(1 <= student.max_courses_per_term <= 5 for student in students)
    
    # Test interest alignment
    interest_check = all(len(student.interest_weights) > 0 for student in students)
    
    print("Validation Results:")
    print(f"  - Prerequisite validation: {'PASS' if prerequisite_check else 'FAIL'}")
    print(f"  - Course load limits: {'PASS' if course_load_check else 'FAIL'}")
    print(f"  - Interest alignment: {'PASS' if interest_check else 'FAIL'}")
    print()
    
    return {
        'prerequisite_validation': prerequisite_check,
        'course_load_limits': course_load_check,
        'interest_alignment': interest_check
    }

def save_demo_results(curriculum, students, sample_courses, recommendations, validation):
    """Save demonstration results to JSON file"""
    print("STEP 5: Saving Demo Results")
    print("-" * 30)
    
    demo_results = {
        'timestamp': datetime.now().isoformat(),
        'curriculum_stats': {
            'total_courses': len(list(curriculum.graph.nodes())),
            'total_prerequisites': len(list(curriculum.graph.edges())),
            'sample_courses': [{'id': cid, 'name': curriculum.course_info[cid]['name']} 
                             for cid in sample_courses]
        },
        'student_stats': {
            'total_students': len(students),
            'average_gpa': sum(s.gpa for s in students) / len(students),
            'average_completed_courses': sum(len(s.completed_courses) for s in students) / len(students),
            'students_with_failures': sum(1 for s in students if len(s.failed_courses) > 0)
        },
        'recommendation_summary': recommendations,
        'validation_results': validation
    }
    
    with open('../generated_datasets/validation_results.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print("Demo results saved to 'generated_datasets/validation_results.json'")
    print()

def display_completion_message():
    """Display completion message and next steps"""
    print("QUICK DEMONSTRATION COMPLETE")
    print("=" * 32)
    print("NEXT STEPS:")
    print("  1. Review 'setup_and_usage_instructions/Installation_and_Usage_Guide.md' for detailed system information")
    print("  2. Run 'python source_code/complete_system_training.py' for full AI training")
    print("  3. Check 'project_deliverables/' for complete deliverables")
    print()

def main():
    """Main demonstration function"""
    try:
        display_header()
        
        curriculum, sample_courses = demonstrate_curriculum_creation()
        students = demonstrate_student_generation(curriculum)
        recommendations = demonstrate_recommendations(curriculum, students)
        validation = validate_system(curriculum, students)
        save_demo_results(curriculum, students, sample_courses, recommendations, validation)
        
        display_completion_message()
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 