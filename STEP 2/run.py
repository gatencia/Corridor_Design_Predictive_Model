#!/usr/bin/env python3
"""
Runs all main Step 2 processing and visualization scripts sequentially.
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_name: str, script_dir: Path) -> bool:
    """Runs a Python script and returns True if successful, False otherwise."""
    script_path = script_dir / script_name
    if not script_path.exists():
        print(f"❌ Error: Script not found: {script_path}")
        return False

    command = [sys.executable, str(script_path)]
    print(f"\n▶️  Running: {' '.join(command)}")
    print("-" * 70)
    
    try:
        # Run the script and stream its output
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        process.wait() # Wait for the process to complete
        
        if process.returncode == 0:
            print(f"\n✅ Successfully finished: {script_name}")
            print("-" * 70)
            return True
        else:
            print(f"\n❌ Error: {script_name} failed with return code {process.returncode}")
            print("-" * 70)
            return False
    except FileNotFoundError:
        print(f"❌ Error: Python interpreter '{sys.executable}' not found or script '{script_path}' not found.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred while running {script_name}: {e}")
        return False

def main():
    """Main function to execute all Step 2 scripts."""
    step2_dir = Path(__file__).parent.resolve()
    
    scripts_to_run = [
        "step2_mini_test.py",
        "comprehensive_step2_testing.py",
        "run_gps_processing.py",
        "create_individual_site_maps.py",
        "visualize_gps_data.py"
    ]
    
    print("🚀 Starting Step 2 Full Workflow")
    print("=================================")
    
    all_successful = True
    for script_name in scripts_to_run:
        if not run_script(script_name, step2_dir):
            all_successful = False
            print(f"🛑 Halting workflow due to error in {script_name}.")
            break
            
    if all_successful:
        print("\n🎉 All Step 2 scripts completed successfully!")
    else:
        print("\n⚠️  One or more Step 2 scripts failed. Please check the output above.")

if __name__ == "__main__":
    main()
