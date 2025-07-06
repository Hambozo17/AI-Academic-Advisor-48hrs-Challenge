# AI Academic Advisor - 48 Hours Challenge

> **AI Curriculum Planner: Adaptive Academic Advising for 100 Simulated Students**  
> Complete implementation of graph-based curriculum modeling with reinforcement learning-based personalization

## 🎯 Challenge Overview

This project implements a personalized academic advising system that:
- Models university curriculum using **graph structures** (29 courses, 32 prerequisites)
- Simulates **100 Egyptian students** with realistic academic histories
- Uses **Deep Q-Network (RL)** for AI-powered course recommendations
- Respects constraints (prerequisites, course load limits, retake policies)
- Aligns recommendations with student interests and graduation goals

## 🏗️ Project Structure

```
├── source_code/                           # Core implementation modules
│   ├── university_curriculum_modeling.py  # Curriculum graph & constraints
│   ├── egyptian_student_generator.py      # Student simulation system
│   ├── deep_learning_advisor.py          # RL-based recommendation engine
│   ├── complete_system_training.py       # Full AI training (500 episodes)
│   ├── system_validation_demo.py         # Quick demonstration & validation
│   └── report_generation_utilities.py    # PDF report creation
├── generated_datasets/                    # Student & curriculum data
│   ├── egyptian_students_dataset.json    # 100 simulated students
│   ├── university_course_structure.json  # Complete curriculum graph
│   └── validation_results.json           # System validation results
├── project_deliverables/                 # Challenge deliverables
│   └── Technical_Implementation_Report.pdf  # 2-page PDF report
├── charts_and_graphs/                    # Visualizations
│   ├── curriculum_graph_improved.png     # Network visualization
│   └── student_analysis.png             # Population statistics
└── setup_and_usage_instructions/         # Documentation
    └── Installation_and_Usage_Guide.md   # Detailed setup guide
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Windows/Linux/macOS**

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/AI-Academic-Advisor-48hrs-Challenge.git
cd AI-Academic-Advisor-48hrs-Challenge
```

2. **Create virtual environment:**
```bash
python -m venv ai_advisor_env
# Windows:
ai_advisor_env\Scripts\activate
# macOS/Linux:
source ai_advisor_env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the System

#### Option 1: Quick Demonstration (Recommended First)
```bash
cd source_code
python system_validation_demo.py
```
**Output:** Validates system functionality with 100 Egyptian students and sample recommendations

#### Option 2: Full AI Training
```bash
cd source_code  
python complete_system_training.py
```
**Output:** Trains Deep Q-Network for 500 episodes with performance metrics

#### Option 3: Generate PDF Report
```bash
cd source_code
python report_generation_utilities.py
```
**Output:** Creates `Technical_Implementation_Report.pdf` in `project_deliverables/`

## 📊 Key Features

### Part 1: Graph-Based Curriculum Modeling
- **29 courses** across 5 specialization tracks (AI, Security, Data Science, Software Engineering, Systems)
- **32 prerequisite relationships** enforcing academic progression
- **Constraint modeling**: Course load limits (3-5 courses/term), prerequisite validation, retake policies
- **NetworkX visualization** with hierarchical layout and color-coded tracks

### Part 2: AI-Based Personalization (Reinforcement Learning)
- **Deep Q-Network (DQN)** with 256-neuron hidden layers
- **State representation**: 40+ dimensional vector (completed courses, GPA, term, interests)
- **Action space**: Course selection respecting constraints
- **Multi-objective rewards**: +10 interest alignment, +20 graduation progress, -20 violations
- **Training**: 500 episodes with epsilon-greedy exploration

### Egyptian Student Integration
- **100 authentic Egyptian students** with names like Ziad Ismail, Yasmin El-Dakrory, Malak Nasser
- **Realistic academic histories** across terms 1-8
- **Diverse performance**: Average GPA 3.36, varied course completion patterns
- **Cultural inclusivity** while maintaining technical rigor

## 📈 Results & Validation

### System Performance
- **✅ Prerequisite validation**: 100% compliance
- **✅ Course load limits**: All students within 3-5 courses/term  
- **✅ Interest alignment**: Personalized recommendations by specialization
- **✅ AI training convergence**: Stable reward improvement over 500 episodes

### Sample Student Results
- **Ziad Ismail** (Term 6, GPA 3.72): Recommended advanced AI courses
- **Yasmin El-Dakrory** (Term 5, GPA 3.81): Security specialization track
- **Malak Nasser** (Term 4, GPA 3.54): Data Science pathway

## 🛠️ Technical Implementation

### Technologies Used
- **Python 3.8+** - Core implementation
- **PyTorch** - Deep Q-Network training
- **NetworkX** - Graph modeling and visualization  
- **NumPy/Pandas** - Data processing and analysis
- **Matplotlib** - Visualization and reporting
- **JSON** - Data persistence and exchange

### Architecture Highlights
- **Modular design** with clear separation of concerns
- **Constraint satisfaction** with graph-based prerequisite checking
- **Scalable RL framework** supporting different reward functions
- **Professional code organization** with comprehensive documentation

## 📋 Challenge Compliance

### Part 1 Requirements ✅
- [x] Graph structure curriculum modeling (29 courses, 32 edges)
- [x] 100 simulated students with completed courses, GPA, interests
- [x] Constraint modeling (load limits, prerequisites, retake policy)
- [x] Script generating student data + curriculum graph
- [x] Graph visualization with NetworkX

### Part 2 Requirements ✅  
- [x] RL-based personalization algorithm (Deep Q-Network)
- [x] Recommendations respecting constraints and aligning with interests
- [x] State/Action/Reward definition for academic planning
- [x] Model training for 10+ students (trained on all 100)

### Deliverables ✅
- [x] GitHub repository with all code, models, and sample data
- [x] README explaining setup and execution
- [x] 2-page PDF report with schema, logic, strategy, and results

## 🏆 Key Achievements

1. **Complete 48 Hours Challenge Implementation** - All requirements met
2. **Cultural Integration** - Egyptian student names and authentic data
3. **Professional Code Quality** - Modular, documented, and maintainable
4. **Advanced AI Implementation** - Deep Q-Network with multi-objective optimization
5. **Comprehensive Validation** - System testing and performance metrics

## 📞 Support & Documentation

- **Detailed Setup Guide**: `setup_and_usage_instructions/Installation_and_Usage_Guide.md`
- **Challenge Compliance**: `CHALLENGE_COMPLIANCE_VERIFICATION.md`  
- **Project Achievements**: `Project_Overview_and_Achievements.md`
- **Technical Report**: `project_deliverables/Technical_Implementation_Report.pdf`

---

**Developed for the 48 Hours Challenge: AI-Powered Academic Advisor**  
*Demonstrating graph-based curriculum modeling with reinforcement learning-based personalization* 