"""
AI Academic Advisor - Source Code Package
48 Hours Challenge Implementation

This package contains all the core implementation modules for the AI Academic Advisor system.
"""

# Import main modules for easy access
from .university_curriculum_modeling import CurriculumGraph, create_sample_curriculum
from .egyptian_student_generator import StudentSimulator, Student
from .deep_learning_advisor import CourseRecommendationAgent

__version__ = "1.0.0"
__author__ = "AI Academic Advisor Team"
__description__ = "48 Hours Challenge - AI-Powered Academic Advisor with Egyptian Student Integration" 