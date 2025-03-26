import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def sanitize_path(path):
    """Convert Windows paths to Python-safe format"""
    return path.replace('\\', '/')

python_repl = PythonREPL()

def save_report(content, input_path):
    """Save report to text file"""
    directory = os.path.dirname(input_path)
    report_path = os.path.join(directory, "ml_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(str(content))  # Ensure content is string
    return report_path

# Define agents with proper tool configuration
eda_agent = Agent(
    role='Data Analyst',
    goal='Perform efficient EDA',
    backstory="Expert in quick data analysis and visualization",
    verbose=True,
    tools=[Tool.from_function(
        func=lambda cmd: python_repl.run(cmd),
        name="python_repl",
        description="Executes Python code"
    )]
)

ml_engineer = Agent(
    role='ML Engineer',
    goal='Select best model',
    backstory="Expert in model selection",
    verbose=True,
    allow_delegation=False
)

trainer = Agent(
    role='Trainer',
    goal='Train models efficiently',
    backstory="Expert in efficient model training",
    verbose=True,
    tools=[Tool.from_function(
        func=lambda cmd: python_repl.run(cmd),
        name="python_repl",
        description="Executes Python code"
    )]
)

reporter = Agent(
    role='Reporter',
    goal='Generate concise report',
    backstory="Technical writer expert",
    verbose=True
)

def ml_pipeline(input_path):
    sanitized_path = sanitize_path(input_path)
    
    # File loading task
    load_task = Task(
        description=f"Load data from {sanitized_path}",
        agent=eda_agent,
        expected_output="Data loaded successfully with basic validation",
        config={'path': sanitized_path}
    )

    # EDA Task
    eda_task = Task(
        description="Perform quick data analysis",
        agent=eda_agent,
        context=[load_task],
        expected_output="Key statistics and data overview",
        config={'max_columns': 10}
    )

    # Model Selection Task
    model_task = Task(
        description="Select best model type",
        agent=ml_engineer,
        context=[eda_task],
        expected_output="Recommended model with justification"
    )

    # Training Task
    train_task = Task(
        description="Train model and generate metrics",
        agent=trainer,
        context=[model_task],
        expected_output="Trained model with evaluation metrics",
        config={'max_iter': 100}
    )

    # Report Generation Task
    report_task = Task(
        description="Generate final report with code",
        agent=reporter,
        context=[train_task],
        expected_output="Complete report file in markdown format"
    )

    crew = Crew(
        agents=[eda_agent, ml_engineer, trainer, reporter],
        tasks=[load_task, eda_task, model_task, train_task, report_task],
        verbose=True,
        process=Process.sequential
    )
    
    result = crew.kickoff()
    return save_report(result, input_path)

if __name__ == "__main__":
    input_path = input("Enter dataset path: ")
    report_path = ml_pipeline(input_path)
    print(f"Report generated at: {report_path}")
