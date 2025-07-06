import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import random
from collections import deque, defaultdict
import json
import pickle

from university_curriculum_modeling import CurriculumGraph
from egyptian_student_generator import Student, StudentSimulator

@dataclass
class CourseRecommendationState:
    """Represents the state for RL-based course recommendation"""
    student_id: str
    completed_courses: Set[str]
    failed_courses: Set[str] 
    current_gpa: float
    current_term: int
    interest_weights: Dict[str, float]
    max_courses_per_term: int
    graduation_goal_terms: int
    eligible_courses: Set[str]
    
    def to_vector(self, curriculum: CurriculumGraph) -> np.ndarray:
        """Convert state to numerical vector for RL model"""
        # Create feature vector
        features = []
        
        # Basic student info
        features.extend([
            self.current_gpa / 4.0,  # Normalized GPA
            self.current_term / 10.0,  # Normalized term
            len(list(self.completed_courses)) / len(list(curriculum.graph.nodes())),  # Progress ratio
            len(list(self.failed_courses)) / max(len(list(curriculum.graph.nodes())), 1),  # Failure ratio
            self.max_courses_per_term / 5.0,  # Normalized course load
            (self.graduation_goal_terms - self.current_term) / 10.0  # Time to graduation
        ])
        # Interest weights (one-hot encoded by area)
        for area in curriculum.interest_areas.keys():
            features.append(self.interest_weights.get(area, 0.0))
        
        # Course completion status (binary vector for each course)
        for course_id in sorted(curriculum.graph.nodes()):
            if course_id in self.completed_courses:
                features.append(1.0)
            elif course_id in self.failed_courses:
                features.append(-0.5)  # Failed courses get negative encoding
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)

@dataclass 
class CourseRecommendationAction:
    """Represents an action (course selection) for the recommendation system"""
    selected_courses: Set[str]
    
    def to_vector(self, curriculum: CurriculumGraph) -> np.ndarray:
        """Convert action to binary vector"""
        nodes = list(curriculum.graph.nodes())
        action_vector = np.zeros(len(nodes), dtype=np.float32)
        for i, course_id in enumerate(sorted(nodes)):
            if course_id in self.selected_courses:
                action_vector[i] = 1.0
        return action_vector

class CourseRecommendationEnvironment:
    """Environment for training the course recommendation RL agent"""
    
    def __init__(self, curriculum: CurriculumGraph, students: List[Student]):
        self.curriculum = curriculum
        self.students = students
        self.current_student_idx = 0
        self.current_student = None
        self.original_state = None
        
    def reset(self, student_idx: Optional[int] = None) -> CourseRecommendationState:
        """Reset environment with a specific student or random student"""
        if student_idx is not None:
            self.current_student_idx = student_idx
        else:
            self.current_student_idx = random.randint(0, len(self.students) - 1)
        
        self.current_student = self.students[self.current_student_idx]
        
        # Create initial state
        eligible_courses = self.curriculum.get_eligible_courses(self.current_student.completed_courses)
        
        state = CourseRecommendationState(
            student_id=self.current_student.student_id,
            completed_courses=self.current_student.completed_courses.copy(),
            failed_courses=self.current_student.failed_courses.copy(),
            current_gpa=self.current_student.gpa,
            current_term=self.current_student.current_term,
            interest_weights=self.current_student.interest_weights.copy(),
            max_courses_per_term=self.current_student.max_courses_per_term,
            graduation_goal_terms=self.current_student.graduation_goal_terms,
            eligible_courses=eligible_courses
        )
        
        self.original_state = state
        return state
    
    def step(self, action: CourseRecommendationAction) -> Tuple[CourseRecommendationState, float, bool]:
        """Execute action and return new state, reward, and done flag"""
        if self.current_student is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # Validate action
        reward = self.calculate_reward(action)
        
        # Simulate taking the courses (simplified)
        if self.original_state is None:
            raise ValueError("Environment not reset. Call reset() first.")
            
        new_completed = self.original_state.completed_courses.copy()
        new_failed = self.original_state.failed_courses.copy()
        new_gpa = self.original_state.current_gpa
        
        # Simulate course outcomes based on student characteristics
        for course in action.selected_courses:
            if course in self.curriculum.course_info:
                # Simulate grade based on difficulty and interest
                difficulty = self.curriculum.course_info[course]['difficulty']
                interest_area = self.curriculum.course_info[course]['interest_area']
                interest_weight = self.original_state.interest_weights.get(interest_area, 0.2)
                
                # Predict grade
                predicted_grade = min(4.0, max(0.0, 
                    self.original_state.current_gpa + 
                    (interest_weight - 0.5) * 2.0 - 
                    (difficulty - 5.0) * 0.2 + 
                    random.gauss(0, 0.3)
                ))
                
                if predicted_grade >= 2.0:  # Pass
                    new_completed.add(course)
                    new_failed.discard(course)
                else:  # Fail
                    new_failed.add(course)
        
        # Update state
        new_eligible = self.curriculum.get_eligible_courses(new_completed)
        
        new_state = CourseRecommendationState(
            student_id=self.original_state.student_id,
            completed_courses=new_completed,
            failed_courses=new_failed,
            current_gpa=new_gpa,
            current_term=self.original_state.current_term + 1,
            interest_weights=self.original_state.interest_weights,
            max_courses_per_term=self.original_state.max_courses_per_term,
            graduation_goal_terms=self.original_state.graduation_goal_terms,
            eligible_courses=new_eligible
        )
        
        # Check if episode is done (graduated or no more eligible courses)
        done = len(list(new_eligible)) == 0 or len(new_completed) >= len(list(self.curriculum.graph.nodes())) * 0.8
        
        return new_state, reward, done
    
    def calculate_reward(self, action: CourseRecommendationAction) -> float:
        """Calculate reward for the given action"""
        reward = 0.0
        
        # Constraint violations (penalties)
        if self.original_state and len(action.selected_courses) > self.original_state.max_courses_per_term:
            reward -= 10.0 * (len(action.selected_courses) - self.original_state.max_courses_per_term)
        
        # Check prerequisite violations
        for course in action.selected_courses:
            prerequisites = set(self.curriculum.graph.predecessors(course))
            if self.original_state and not prerequisites.issubset(self.original_state.completed_courses):
                reward -= 20.0  # Heavy penalty for prerequisite violations
        
        # Check if courses are eligible
        if self.original_state and self.original_state.eligible_courses:
            for course in action.selected_courses:
                if course not in self.original_state.eligible_courses:
                    reward -= 15.0
        
        # Positive rewards
        for course in action.selected_courses:
            if course in self.curriculum.course_info:
                course_info = self.curriculum.course_info[course]
                
                # Interest alignment reward
                if self.original_state and self.original_state.interest_weights:
                    interest_area = course_info['interest_area']
                    interest_weight = self.original_state.interest_weights.get(interest_area, 0.2)
                    reward += interest_weight * 10.0
                
                # Progress toward graduation
                reward += 2.0  # Base reward for taking any course
        # Calculate progress toward graduation
        if self.original_state and self.original_state.completed_courses is not None:
            total_courses = len(list(self.curriculum.graph.nodes()))
            current_progress = len(self.original_state.completed_courses) / total_courses
            potential_progress = (len(self.original_state.completed_courses) + len(action.selected_courses)) / total_courses
            progress_gain = potential_progress - current_progress
            reward += progress_gain * 20.0
        # Time to graduation consideration
        if self.original_state and self.original_state.graduation_goal_terms is not None and self.original_state.current_term is not None:
            terms_remaining = self.original_state.graduation_goal_terms - self.original_state.current_term
            if terms_remaining > 0:
                optimal_courses_per_term = (total_courses - len(self.original_state.completed_courses)) / terms_remaining
                if len(action.selected_courses) >= optimal_courses_per_term * 0.8:
                    reward += 5.0  # Bonus for staying on track
        
        return reward

class DQN(nn.Module):
    """Deep Q-Network for course recommendation"""
    
    def __init__(self, state_size: int, num_courses: int, hidden_size: int = 256):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.num_courses = num_courses
        
        # Shared layers for processing state
        self.state_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layer for each course (Q-value for selecting each course)
        self.course_q_values = nn.Linear(hidden_size, num_courses)
        
    def forward(self, state):
        state_features = self.state_layers(state)
        q_values = self.course_q_values(state_features)
        return q_values

class CourseRecommendationAgent:
    """RL Agent for course recommendation using DQN"""
    
    def __init__(self, curriculum: CurriculumGraph, students: List[Student], 
                 learning_rate: float = 0.001, epsilon: float = 0.1):
        self.curriculum = curriculum
        self.students = students
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        
        # Initialize environment
        self.env = CourseRecommendationEnvironment(curriculum, students)
        
        # Calculate state size
        sample_state = self.env.reset()
        state_vector = sample_state.to_vector(curriculum)
        self.state_size = len(state_vector)
        self.num_courses = len(list(curriculum.graph.nodes()))
        
        # Initialize neural networks
        self.q_network = DQN(self.state_size, self.num_courses)
        self.target_network = DQN(self.state_size, self.num_courses)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
        
    def select_action(self, state: CourseRecommendationState, training: bool = True) -> CourseRecommendationAction:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            eligible_courses = list(state.eligible_courses)
            if not eligible_courses:
                return CourseRecommendationAction(set())
            
            num_courses = min(random.randint(1, state.max_courses_per_term), len(eligible_courses))
            selected = set(random.sample(eligible_courses, num_courses))
            return CourseRecommendationAction(selected)
        else:
            # Greedy action based on Q-values
            state_tensor = torch.FloatTensor(state.to_vector(self.curriculum)).unsqueeze(0)
            q_values = self.q_network(state_tensor).squeeze()
            
            # Mask out ineligible courses
            eligible_indices = []
            course_list = sorted(self.curriculum.graph.nodes())
            for i, course in enumerate(course_list):
                if course in state.eligible_courses:
                    eligible_indices.append(i)
            
            if not eligible_indices:
                return CourseRecommendationAction(set())
            
            # Select top courses based on Q-values, respecting course load limit
            eligible_q_values = [(i, q_values[i].item()) for i in eligible_indices]
            eligible_q_values.sort(key=lambda x: x[1], reverse=True)
            
            num_courses = min(state.max_courses_per_term, len(eligible_q_values))
            selected_indices = [idx for idx, _ in eligible_q_values[:num_courses]]
            selected_courses = {course_list[idx] for idx in selected_indices}
            
            return CourseRecommendationAction(selected_courses)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        try:
            states = torch.FloatTensor([exp[0].to_vector(self.curriculum) for exp in batch])
            actions = torch.LongTensor([self._action_to_indices(exp[1]) for exp in batch])
            rewards = torch.FloatTensor([exp[2] for exp in batch])
            next_states = torch.FloatTensor([exp[3].to_vector(self.curriculum) for exp in batch])
            dones = torch.BoolTensor([exp[4] for exp in batch])
            
            current_q_values = self.q_network(states)
            next_q_values = self.target_network(next_states)
            
            # Calculate target Q-values
            target_q_values = rewards + (0.99 * next_q_values.max(1)[0] * ~dones)
            
            # Calculate loss using action indices
            current_q_selected = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(current_q_selected, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.training_losses.append(loss.item())
            
        except Exception as e:
            # Skip this batch if there's an error
            print(f"Skipping batch due to error: {e}")
        
        # Decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    
    def _action_to_indices(self, action: CourseRecommendationAction) -> int:
        """Convert action to single course index (simplified for tensor compatibility)"""
        course_list = sorted(self.curriculum.graph.nodes())
        if action.selected_courses:
            # Return index of first selected course (simplified)
            first_course = list(action.selected_courses)[0]
            if first_course in course_list:
                return course_list.index(first_course)
        return 0  # Fallback
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_episodes: int = 1000, update_target_freq: int = 100):
        """Train the recommendation agent"""
        print(f"Training course recommendation agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
                # Train on batch
                self.replay()
            
            self.training_rewards.append(episode_reward)
            
            # Update target network
            if episode % update_target_freq == 0:
                self.update_target_network()
            
            # Progress reporting
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:]) if self.training_rewards else 0
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
    
    def recommend_courses(self, student: Student) -> Tuple[Set[str], float]:
        """Get course recommendations for a specific student"""
        # Create state for the student
        eligible_courses = self.curriculum.get_eligible_courses(student.completed_courses)
        
        state = CourseRecommendationState(
            student_id=student.student_id,
            completed_courses=student.completed_courses,
            failed_courses=student.failed_courses,
            current_gpa=student.gpa,
            current_term=student.current_term,
            interest_weights=student.interest_weights,
            max_courses_per_term=student.max_courses_per_term,
            graduation_goal_terms=student.graduation_goal_terms,
            eligible_courses=eligible_courses
        )
        
        # Get recommendation
        action = self.select_action(state, training=False)
        
        # Calculate confidence/quality score
        confidence = self.env.calculate_reward(action)
        
        return action.selected_courses, confidence
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses,
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_rewards = checkpoint['training_rewards']
        self.training_losses = checkpoint['training_losses']
        self.epsilon = checkpoint['epsilon']

if __name__ == "__main__":
    # This will be used for testing
    print("RL Course Recommendation Engine loaded successfully!")
    print("Use this module to create and train recommendation agents.") 