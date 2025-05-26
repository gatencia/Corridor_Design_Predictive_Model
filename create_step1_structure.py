#!/usr/bin/env python3
"""
Step 1: Environment Setup - Directory Structure Generator
Creates the complete folder structure and placeholder files for the elephant corridor project environment setup.
"""

import os
from pathlib import Path

def create_step1_structure():
    """
    Creates the directory structure and placeholder files for Step 1: Environment Setup
    """
    
    # Define the project root
    project_root = Path("/Users/guillaumeatencia/Documents/Projects_2025/Elephant_Corridor_Research/STEP 1")
    
    # Define directory structure
    directories = [
        project_root,
        project_root / "scripts",
        project_root / "config"
    ]
    
    # Define files to create with their paths
    files = {
        # Root level files
        project_root / "environment.yml": "# Conda environment specification",
        project_root / "requirements.txt": "# Pip requirements as backup", 
        project_root / "setup.py": "# Package setup configuration",
        project_root / ".env": "# Environment variables and configuration",
        project_root / ".gitignore": "# Git ignore file",
        project_root / "README.md": "# Project overview and setup instructions",
        
        # Scripts directory
        project_root / "scripts" / "setup_environment.py": "# Script to verify environment setup",
        project_root / "scripts" / "install_dependencies.sh": "# Shell script for additional setup",
        
        # Config directory  
        project_root / "config" / "project_config.py": "# Project configuration settings"
    }
    
    print("Creating Step 1: Environment Setup directory structure...")
    print("=" * 60)
    
    # Create directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\nCreating placeholder files...")
    print("-" * 40)
    
    # Create files with placeholder content
    for file_path, placeholder_content in files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(placeholder_content + "\n")
        print(f"📄 Created file: {file_path}")
    
    print("\n" + "=" * 60)
    print("✅ Step 1 directory structure created successfully!")
    print(f"📂 Project root: {project_root.absolute()}")
    
    # Display the structure
    print("\n📋 Directory Structure:")
    print_directory_tree(project_root)

def print_directory_tree(path, prefix="", is_last=True):
    """
    Recursively prints a directory tree structure
    """
    if path.is_dir():
        print(f"{prefix}{'└── ' if is_last else '├── '}{path.name}/")
        children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            extension = "    " if is_last else "│   "
            print_directory_tree(child, prefix + extension, is_last_child)
    else:
        print(f"{prefix}{'└── ' if is_last else '├── '}{path.name}")

if __name__ == "__main__":
    create_step1_structure()
    
    print("\n" + "🚀 Next Steps:")
    print("1. Navigate to the elephant-corridors-project directory")
    print("2. Review the placeholder files")  
    print("3. We'll implement each file individually starting with environment.yml")
    print("4. Run the setup verification once all files are completed")