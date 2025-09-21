import os

# Define project structure
structure = {
    ".vscode": ["settings.json", "launch.json"],
    "data/raw/data_input_output/p00001": ["input.txt", "output.txt"],
    "data/raw/data_input_output/p00002": [],
    "data/raw/data_input_output/p00003": [],
    "data/raw/data_input_output/p00005": [],
    "data/raw/data_input_output/p00007": [],
    "data/raw/human_code/p00001": [],  # placeholder for human code files
    "data/raw/human_code/p00007": ["s004464571.py"],
    "data/raw/problem_desc": ["p00001.html", "p00007.html"],
    "src": [
        "custom_data_processor.py",
        "ai_code_generator.py",
        "style_metrics.py",
        "pipeline.py",
    ],
    "scripts": ["setup.sh", "run_pipeline.sh"],
    "results": ["ai_codes.json", "style_comparison.csv"],
    "tests": ["test_pipeline.py"],
}

# Files at root
root_files = ["requirements.txt", "README.md"]

created = []
already_present = []

def ensure_path(path):
    """Ensure directory exists"""
    if not os.path.exists(path):
        os.makedirs(path)
        created.append(f"DIR: {path}")
    else:
        already_present.append(f"DIR: {path}")

def ensure_file(path):
    """Ensure file exists (empty placeholder if new)"""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("")  # placeholder
        created.append(f"FILE: {path}")
    else:
        already_present.append(f"FILE: {path}")

# Build structure
for folder, files in structure.items():
    ensure_path(folder)
    for file in files:
        ensure_file(os.path.join(folder, file))

# Root files
for f in root_files:
    ensure_file(f)

# Summary
print("\n✅ Creation Summary")
print("---------------------")
print("Created:")
for c in created:
    print("  -", c)

print("\nAlready Present:")
for a in already_present:
    print("  -", a)
