#!/usr/bin/env python3
"""
AI Academic Advisor - Main Training and Demonstration Script
Complete system demonstration for the 48 Hours Challenge

This script demonstrates the complete AI academic advisor system:
1. Creates curriculum graph with prerequisite relationships
2. Generates 100 simulated students with diverse academic histories
3. Trains RL-based recommendation system using Deep Q-Network
4. Generates sample recommendations for demonstration
5. Produces comprehensive analysis and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple, Optional
import json
import time
import sys
from datetime import datetime

# Import our modules
from university_curriculum_modeling import CurriculumGraph, create_sample_curriculum
from egyptian_student_generator import StudentSimulator, Student, analyze_student_data
from deep_learning_advisor import CourseRecommendationAgent

def initialize_system():
    """Initialize the complete AI academic advisor system"""
    print("INITIALIZING AI ACADEMIC ADVISOR SYSTEM")
    print("=" * 60)
    
    # Step 1: Create curriculum graph
    print("Creating curriculum graph...")
    curriculum = create_sample_curriculum()
    curriculum.save_graph('generated_datasets/university_course_structure.json')
    print(f"  Created curriculum with {len(list(curriculum.graph.nodes()))} courses")
    print(f"  Added {len(list(curriculum.graph.edges()))} prerequisite relationships")
    
    # Step 2: Generate student population
    print("\nGenerating student population...")
    simulator = StudentSimulator(curriculum)
    students = simulator.generate_students(100)
    simulator.save_students('generated_datasets/egyptian_students_dataset.json')
    print(f"  Generated {len(students)} students with diverse academic histories")
    
    # Step 3: Analyze student population
    print("\nAnalyzing student population...")
    stats = simulator.get_student_statistics()
    print(f"  Average GPA: {stats['average_gpa']:.2f}")
    print(f"  Average completed courses: {stats['average_completed_courses']:.1f}")
    print(f"  Students with failures: {stats['students_with_failed_courses']}")
    
    return curriculum, students, simulator

def train_recommendation_system(curriculum: CurriculumGraph, students: List[Student]):
    """Train the RL-based course recommendation system"""
    print("\nTRAINING AI RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Initialize agent
    agent = CourseRecommendationAgent(curriculum, students, learning_rate=0.001, epsilon=0.3)
    
    # Train the system
    start_time = time.time()
    print("Training Deep Q-Network...")
    agent.train(num_episodes=500, update_target_freq=50)
    training_time = time.time() - start_time
    
    print(f"  Training completed in {training_time:.1f} seconds")
    print(f"  Final epsilon (exploration rate): {agent.epsilon:.3f}")
    
    # Save trained model
    try:
        agent.save_model('generated_datasets/trained_dqn_model.pth')
        print("  Model saved to 'trained_recommendation_model.pth'")
    except Exception as e:
        print(f"  Warning: Could not save model - {e}")
    
    return agent

def demonstrate_recommendations(agent: CourseRecommendationAgent, students: List[Student], 
                              curriculum: CurriculumGraph, num_demo_students: int = 5):
    """Demonstrate course recommendations for sample students"""
    print(f"\nGENERATING COURSE RECOMMENDATIONS")
    print("=" * 60)
    
    demo_results = []
    
    for i in range(num_demo_students):
        student = students[i]
        print(f"\nStudent {i+1}: {student.name} ({student.student_id})")
        print(f"  Current Term: {student.current_term}")
        print(f"  GPA: {student.gpa:.2f}")
        print(f"  Primary Interests: {', '.join(student.interests)}")
        print(f"  Completed Courses: {len(student.completed_courses)}")
        print(f"  Failed Courses: {len(student.failed_courses)}")
        
        # Get recommendations
        recommended_courses, confidence = agent.recommend_courses(student)
        
        print(f"\n  AI RECOMMENDATIONS:")
        print(f"  Confidence Score: {confidence:.2f}")
        
        if recommended_courses:
            print(f"  Recommended Courses ({len(recommended_courses)}):")
            for course in sorted(recommended_courses):
                if course in curriculum.course_info:
                    info = curriculum.course_info[course]
                    print(f"    - {course}: {info['name']} ({info['interest_area']})")
        else:
            print("    No courses recommended (may have completed program)")
        
        # Validate recommendations
        eligible_courses = curriculum.get_eligible_courses(student.completed_courses)
        valid_recommendations = recommended_courses.intersection(eligible_courses)
        
        print(f"  Valid recommendations: {len(valid_recommendations)}/{len(recommended_courses)}")
        print(f"  Respects course load limit: {len(recommended_courses) <= student.max_courses_per_term}")
        
        # Store results for analysis
        demo_results.append({
            'student_id': student.student_id,
            'student_name': student.name,
            'current_term': student.current_term,
            'gpa': student.gpa,
            'interests': student.interests,
            'completed_courses': len(student.completed_courses),
            'failed_courses': len(student.failed_courses),
            'recommended_courses': list(recommended_courses),
            'confidence_score': confidence,
            'valid_recommendations': len(valid_recommendations),
            'total_recommendations': len(recommended_courses)
        })
        
        print("-" * 40)
    
    return demo_results

def analyze_recommendation_quality(demo_results: List[Dict], curriculum: CurriculumGraph):
    """Analyze the quality and characteristics of recommendations"""
    print(f"\nRECOMMENDATION QUALITY ANALYSIS")
    print("=" * 60)
    
    # Basic metrics
    total_recommendations = sum(r['total_recommendations'] for r in demo_results)
    valid_recommendations = sum(r['valid_recommendations'] for r in demo_results)
    average_confidence = np.mean([r['confidence_score'] for r in demo_results])
    
    print(f"Overall Metrics:")
    print(f"  Total recommendations made: {total_recommendations}")
    print(f"  Valid recommendations: {valid_recommendations} ({valid_recommendations/max(total_recommendations,1)*100:.1f}%)")
    print(f"  Average confidence score: {average_confidence:.2f}")
    
    # Interest alignment analysis
    interest_alignment = {}
    for result in demo_results:
        student_interests = set(result['interests'])
        for course in result['recommended_courses']:
            if course in curriculum.course_info:
                course_area = curriculum.course_info[course]['interest_area']
                if course_area in student_interests:
                    interest_alignment[result['student_id']] = interest_alignment.get(result['student_id'], 0) + 1
    
    aligned_students = len([s for s in interest_alignment.values() if s > 0])
    print(f"  Students with interest-aligned recommendations: {aligned_students}/{len(demo_results)}")
    
    # Course difficulty distribution
    recommended_difficulties = []
    for result in demo_results:
        for course in result['recommended_courses']:
            if course in curriculum.course_info:
                recommended_difficulties.append(curriculum.course_info[course]['difficulty'])
    
    if recommended_difficulties:
        avg_difficulty = np.mean(recommended_difficulties)
        print(f"  Average difficulty of recommended courses: {avg_difficulty:.1f}/10")
    
    return {
        'total_recommendations': total_recommendations,
        'valid_recommendations': valid_recommendations,
        'average_confidence': average_confidence,
        'interest_aligned_students': aligned_students,
        'average_difficulty': np.mean(recommended_difficulties) if recommended_difficulties else 0
    }

def create_visualizations(curriculum: CurriculumGraph, students: List[Student], 
                         demo_results: List[Dict], agent: CourseRecommendationAgent):
    """Create visualizations for the system"""
    print(f"\nCREATING VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Curriculum Graph Visualization
    print("Generating curriculum graph visualization...")
    try:
        curriculum.visualize_graph(save_path='curriculum_graph.png')
        print("  Saved: curriculum_graph.png")
    except Exception as e:
        print(f"  Warning: Could not create graph visualization - {e}")
    
    # 2. Student Distribution Analysis
    plt.figure(figsize=(15, 10))
    
    # GPA distribution
    plt.subplot(2, 3, 1)
    gpas = [s.gpa for s in students]
    plt.hist(gpas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Student GPA Distribution')
    plt.xlabel('GPA')
    plt.ylabel('Number of Students')
    
    # Term distribution
    plt.subplot(2, 3, 2)
    terms = [s.current_term for s in students]
    plt.hist(terms, bins=range(1, 10), alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Student Term Distribution')
    plt.xlabel('Current Term')
    plt.ylabel('Number of Students')
    
    # Completed courses distribution
    plt.subplot(2, 3, 3)
    completed_counts = [len(s.completed_courses) for s in students]
    plt.hist(completed_counts, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Completed Courses Distribution')
    plt.xlabel('Number of Completed Courses')
    plt.ylabel('Number of Students')
    
    # Interest area popularity
    plt.subplot(2, 3, 4)
    interest_counts = {}
    for student in students:
        for interest in student.interests:
            interest_counts[interest] = interest_counts.get(interest, 0) + 1
    
    areas = list(interest_counts.keys())
    counts = list(interest_counts.values())
    plt.bar(areas, counts, alpha=0.7, color='gold')
    plt.title('Interest Area Popularity')
    plt.xlabel('Interest Area')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=45)
    
    # Training progress (if available)
    plt.subplot(2, 3, 5)
    if hasattr(agent, 'training_rewards') and agent.training_rewards:
        # Moving average of rewards
        window_size = 50
        if len(agent.training_rewards) >= window_size:
            moving_avg = np.convolve(agent.training_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, color='purple')
            plt.title('Training Progress (Moving Average)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
        else:
            plt.text(0.5, 0.5, 'Insufficient training data', ha='center', va='center')
            plt.title('Training Progress')
    else:
        plt.text(0.5, 0.5, 'No training data available', ha='center', va='center')
        plt.title('Training Progress')
    
    # Recommendation confidence scores
    plt.subplot(2, 3, 6)
    confidence_scores = [r['confidence_score'] for r in demo_results]
    plt.bar(range(len(confidence_scores)), confidence_scores, alpha=0.7, color='orange')
    plt.title('Recommendation Confidence Scores')
    plt.xlabel('Student Index')
    plt.ylabel('Confidence Score')
    
    plt.tight_layout()
    plt.savefig('system_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: system_analysis.png")

def generate_system_report(curriculum: CurriculumGraph, students: List[Student], 
                          demo_results: List[Dict], analysis_results: Dict):
    """Generate a comprehensive system report"""
    print(f"\nGENERATING SYSTEM REPORT")
    print("=" * 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_overview': {
            'total_courses': len(list(curriculum.graph.nodes())),
            'prerequisite_relationships': len(list(curriculum.graph.edges())),
            'total_students': len(students),
            'interest_areas': list(curriculum.interest_areas.keys())
        },
        'student_population': {
            'average_gpa': np.mean([s.gpa for s in students]),
            'gpa_std': np.std([s.gpa for s in students]),
            'average_completed_courses': np.mean([len(s.completed_courses) for s in students]),
            'students_with_failures': len([s for s in students if s.failed_courses])
        },
        'recommendation_system': {
            'total_recommendations': analysis_results['total_recommendations'],
            'valid_recommendations': analysis_results['valid_recommendations'],
            'validity_rate': analysis_results['valid_recommendations'] / max(analysis_results['total_recommendations'], 1),
            'average_confidence': analysis_results['average_confidence'],
            'interest_aligned_students': analysis_results['interest_aligned_students']
        },
        'sample_recommendations': demo_results
    }
    
    # Save report
    with open('system_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("  Saved: system_report.json")
    
    # Create human-readable summary
    with open('system_summary.txt', 'w') as f:
        f.write("AI ACADEMIC ADVISOR - SYSTEM SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CURRICULUM OVERVIEW:\n")
        f.write(f"  - Total Courses: {len(list(curriculum.graph.nodes()))}\n")
        f.write(f"  - Prerequisite Relationships: {len(list(curriculum.graph.edges()))}\n")
        f.write(f"  - Interest Areas: {len(curriculum.interest_areas)}\n\n")
        
        f.write("STUDENT POPULATION:\n")
        f.write(f"  - Total Students: {len(students)}\n")
        f.write(f"  - Average GPA: {np.mean([s.gpa for s in students]):.2f}\n")
        f.write(f"  - Average Completed Courses: {np.mean([len(s.completed_courses) for s in students]):.1f}\n\n")
        
        f.write("RECOMMENDATION SYSTEM PERFORMANCE:\n")
        f.write(f"  - Total Recommendations: {analysis_results['total_recommendations']}\n")
        f.write(f"  - Validity Rate: {analysis_results['valid_recommendations']/max(analysis_results['total_recommendations'],1)*100:.1f}%\n")
        f.write(f"  - Average Confidence: {analysis_results['average_confidence']:.2f}\n")
        f.write(f"  - Interest Alignment: {analysis_results['interest_aligned_students']}/{len(demo_results)} students\n\n")
        
        f.write("SAMPLE STUDENT RECOMMENDATIONS:\n")
        for i, result in enumerate(demo_results[:3]):
            f.write(f"\nStudent {i+1}: {result['student_name']}\n")
            f.write(f"  - Current Term: {result['current_term']}\n")
            f.write(f"  - GPA: {result['gpa']:.2f}\n")
            f.write(f"  - Interests: {', '.join(result['interests'])}\n")
            f.write(f"  - Recommendations: {len(result['recommended_courses'])} courses\n")
            f.write(f"  - Confidence: {result['confidence_score']:.2f}\n")
    
    print("  Saved: system_summary.txt")

def display_completion_summary():
    """Display completion summary and generated files"""
    print(f"\nSYSTEM DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Generated Files:")
    print("  - curriculum_data.json (Curriculum graph data)")
    print("  - student_data.json (Student population data)")  
    print("  - trained_recommendation_model.pth (Trained AI model)")
    print("  - curriculum_graph.png (Graph visualization)")
    print("  - system_analysis.png (Analysis charts)")
    print("  - system_report.json (Detailed system report)")
    print("  - system_summary.txt (Human-readable summary)")
    print("\nAI Academic Advisor is ready for deployment!")

def main():
    """Main execution function"""
    print("AI ACADEMIC ADVISOR - 48 HOURS CHALLENGE")
    print("=" * 60)
    print("Starting complete system demonstration...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Phase 1: System Initialization
        curriculum, students, simulator = initialize_system()
        
        # Phase 2: AI Training
        agent = train_recommendation_system(curriculum, students)
        
        # Phase 3: Demonstration
        demo_results = demonstrate_recommendations(agent, students, curriculum, num_demo_students=5)
        
        # Phase 4: Analysis
        analysis_results = analyze_recommendation_quality(demo_results, curriculum)
        
        # Phase 5: Visualizations
        create_visualizations(curriculum, students, demo_results, agent)
        
        # Phase 6: Report Generation
        generate_system_report(curriculum, students, demo_results, analysis_results)
        
        # Phase 7: Completion Summary
        display_completion_summary()
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 