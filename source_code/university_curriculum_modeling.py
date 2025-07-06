import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Set, Tuple, Optional
import json
import pickle

class CurriculumGraph:
    """
    Graph-based curriculum model for university courses with prerequisites.
    Nodes = courses, Edges = prerequisite relations
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph for prerequisites
        self.course_info = {}  # Additional course metadata
        self.interest_areas = {
            'AI': ['Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'Computer Vision', 'NLP'],
            'Security': ['Cybersecurity', 'Network Security', 'Cryptography', 'Information Security'],
            'Data Science': ['Data Mining', 'Statistics', 'Database Systems', 'Big Data Analytics'],
            'Software Engineering': ['Software Architecture', 'Design Patterns', 'Web Development', 'Mobile Development'],
            'Systems': ['Operating Systems', 'Computer Networks', 'Distributed Systems', 'Cloud Computing']
        }
        
    def add_course(self, course_id: str, name: str, credits: int = 3, 
                   difficulty: float = 5.0, interest_area: str = 'General'):
        """Add a course to the curriculum graph"""
        self.graph.add_node(course_id)
        self.course_info[course_id] = {
            'name': name,
            'credits': credits,
            'difficulty': difficulty,  # 1-10 scale
            'interest_area': interest_area,
            'prerequisites': []
        }
    
    def add_prerequisite(self, prerequisite_course: str, target_course: str):
        """Add prerequisite relationship between courses"""
        if prerequisite_course in self.graph and target_course in self.graph:
            self.graph.add_edge(prerequisite_course, target_course)
            self.course_info[target_course]['prerequisites'].append(prerequisite_course)
    
    def get_eligible_courses(self, completed_courses: Set[str]) -> Set[str]:
        """Get courses that can be taken given completed courses"""
        eligible = set()
        for course in self.graph.nodes():
            if course not in completed_courses:
                prerequisites = set(self.graph.predecessors(course))
                if prerequisites.issubset(completed_courses):
                    eligible.add(course)
        return eligible
    
    def get_courses_by_interest(self, interest_area: str) -> List[str]:
        """Get all courses in a specific interest area"""
        return [course_id for course_id, info in self.course_info.items() 
                if info['interest_area'] == interest_area]
    
    def validate_course_sequence(self, course_sequence: List[str]) -> bool:
        """Validate if a sequence of courses respects prerequisites"""
        completed = set()
        for course in course_sequence:
            prerequisites = set(self.graph.predecessors(course))
            if not prerequisites.issubset(completed):
                return False
            completed.add(course)
        return True
    
    def get_graduation_path_length(self, completed_courses: Set[str]) -> int:
        """Estimate minimum terms to graduation"""
        remaining_courses = set(self.graph.nodes()) - completed_courses
        if not remaining_courses:
            return 0
        
        # Simplified calculation - actual implementation would use topological sorting
        return max(1, len(remaining_courses) // 4)  # Assuming 4 courses per term average
    
    def visualize_graph(self, highlight_courses: Set[str] = None, save_path: str = None):
        """Visualize the curriculum graph"""
        plt.figure(figsize=(15, 10))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Color nodes by interest area
        color_map = {
            'AI': 'lightblue',
            'Security': 'lightcoral', 
            'Data Science': 'lightgreen',
            'Software Engineering': 'lightyellow',
            'Systems': 'lightpink',
            'General': 'lightgray'
        }
        
        node_colors = [color_map.get(self.course_info[node]['interest_area'], 'lightgray') 
                      for node in self.graph.nodes()]
        
        # Highlight specific courses if provided
        if highlight_courses:
            node_colors = ['red' if node in highlight_courses else color 
                          for node, color in zip(self.graph.nodes(), node_colors)]
        
        # Draw the graph
        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=800,
                font_size=8,
                font_weight='bold',
                arrows=True,
                edge_color='gray',
                arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(self.graph, pos, 
                               {node: node for node in self.graph.nodes()})
        
        plt.title("University Curriculum Graph\n(Nodes=Courses, Edges=Prerequisites)", 
                 fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=10, label=area)
                          for area, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_graph(self, filepath: str):
        """Save the curriculum graph to file"""
        data = {
            'graph': nx.node_link_data(self.graph),
            'course_info': self.course_info,
            'interest_areas': self.interest_areas
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_graph(self, filepath: str):
        """Load curriculum graph from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.graph = nx.node_link_graph(data['graph'])
        self.course_info = data['course_info']
        self.interest_areas = data['interest_areas']

def create_sample_curriculum() -> CurriculumGraph:
    """Create a comprehensive sample university curriculum"""
    curriculum = CurriculumGraph()
    
    # Core CS Courses
    curriculum.add_course('CS101', 'Introduction to Programming', 3, 4.0, 'General')
    curriculum.add_course('CS102', 'Data Structures', 3, 6.0, 'General')
    curriculum.add_course('CS201', 'Algorithms', 3, 7.0, 'General')
    curriculum.add_course('CS301', 'Software Engineering', 3, 6.5, 'Software Engineering')
    
    # Math Foundation
    curriculum.add_course('MATH101', 'Calculus I', 4, 6.0, 'General')
    curriculum.add_course('MATH102', 'Calculus II', 4, 6.5, 'General')
    curriculum.add_course('MATH201', 'Linear Algebra', 3, 5.5, 'General')
    curriculum.add_course('MATH301', 'Statistics', 3, 5.0, 'Data Science')
    
    # AI Track
    curriculum.add_course('AI101', 'Introduction to AI', 3, 7.0, 'AI')
    curriculum.add_course('AI201', 'Machine Learning', 3, 8.0, 'AI')
    curriculum.add_course('AI301', 'Deep Learning', 3, 9.0, 'AI')
    curriculum.add_course('AI401', 'Computer Vision', 3, 8.5, 'AI')
    curriculum.add_course('AI402', 'Natural Language Processing', 3, 8.5, 'AI')
    
    # Data Science Track
    curriculum.add_course('DS101', 'Data Mining', 3, 6.5, 'Data Science')
    curriculum.add_course('DS201', 'Database Systems', 3, 6.0, 'Data Science')
    curriculum.add_course('DS301', 'Big Data Analytics', 3, 7.5, 'Data Science')
    curriculum.add_course('DS401', 'Data Visualization', 3, 5.5, 'Data Science')
    
    # Security Track
    curriculum.add_course('SEC101', 'Introduction to Cybersecurity', 3, 6.0, 'Security')
    curriculum.add_course('SEC201', 'Network Security', 3, 7.0, 'Security')
    curriculum.add_course('SEC301', 'Cryptography', 3, 8.0, 'Security')
    curriculum.add_course('SEC401', 'Information Security', 3, 7.5, 'Security')
    
    # Systems Track
    curriculum.add_course('SYS101', 'Operating Systems', 3, 7.5, 'Systems')
    curriculum.add_course('SYS201', 'Computer Networks', 3, 7.0, 'Systems')
    curriculum.add_course('SYS301', 'Distributed Systems', 3, 8.5, 'Systems')
    curriculum.add_course('SYS401', 'Cloud Computing', 3, 7.0, 'Systems')
    
    # Software Engineering Track
    curriculum.add_course('SE101', 'Software Architecture', 3, 6.5, 'Software Engineering')
    curriculum.add_course('SE201', 'Design Patterns', 3, 7.0, 'Software Engineering')
    curriculum.add_course('SE301', 'Web Development', 3, 5.5, 'Software Engineering')
    curriculum.add_course('SE401', 'Mobile Development', 3, 6.0, 'Software Engineering')
    
    # Prerequisites setup
    prerequisites = [
        # Core sequence
        ('CS101', 'CS102'),
        ('CS102', 'CS201'),
        ('CS201', 'CS301'),
        
        # Math sequence
        ('MATH101', 'MATH102'),
        ('MATH102', 'MATH201'),
        ('MATH201', 'MATH301'),
        
        # AI track prerequisites
        ('CS101', 'AI101'),
        ('CS102', 'AI101'),
        ('MATH201', 'AI101'),
        ('AI101', 'AI201'),
        ('MATH301', 'AI201'),
        ('AI201', 'AI301'),
        ('AI201', 'AI401'),
        ('AI201', 'AI402'),
        
        # Data Science prerequisites
        ('CS102', 'DS101'),
        ('MATH301', 'DS101'),
        ('CS102', 'DS201'),
        ('DS101', 'DS301'),
        ('DS201', 'DS301'),
        ('DS101', 'DS401'),
        
        # Security prerequisites
        ('CS101', 'SEC101'),
        ('SEC101', 'SEC201'),
        ('MATH201', 'SEC301'),
        ('SEC201', 'SEC401'),
        
        # Systems prerequisites
        ('CS102', 'SYS101'),
        ('SYS101', 'SYS201'),
        ('SYS201', 'SYS301'),
        ('SYS201', 'SYS401'),
        
        # Software Engineering prerequisites
        ('CS201', 'SE101'),
        ('SE101', 'SE201'),
        ('CS101', 'SE301'),
        ('SE201', 'SE401'),
    ]
    
    for prereq, course in prerequisites:
        curriculum.add_prerequisite(prereq, course)
    
    return curriculum

if __name__ == "__main__":
    # Create and test the curriculum
    curriculum = create_sample_curriculum()
    
    print("Curriculum Graph Created!")
    print(f"Total courses: {len(curriculum.graph.nodes())}")
    print(f"Total prerequisite relationships: {len(curriculum.graph.edges())}")
    
    # Test eligibility
    completed = {'CS101', 'MATH101'}
    eligible = curriculum.get_eligible_courses(completed)
    print(f"\nWith completed courses {completed}:")
    print(f"Eligible next courses: {eligible}")
    
    # Save the curriculum
    curriculum.save_graph('curriculum_data.json')
    print("\nCurriculum saved to 'curriculum_data.json'") 