#!/usr/bin/env python3
import os
import zipfile

# Name of the zip file to create
ZIP_FILENAME = "rag-visualizer-clean.zip"

# Folders to skip
SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    ".next",
    ".pytest_cache",
    ".mypy_cache"
}

# File patterns to skip (extensions or exact names)
SKIP_FILES = {
    ".DS_Store",
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test"
}
SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".log"
}

def should_skip(path):
    """Check if a file or directory should be skipped."""
    parts = set(path.split(os.sep))
    if parts & SKIP_DIRS:
        return True
    filename = os.path.basename(path)
    if filename in SKIP_FILES:
        return True
    if os.path.splitext(filename)[1] in SKIP_EXTENSIONS:
        return True
    return False

def zip_clean_repo(base_dir, zip_filename):
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Modify dirs in-place to skip entire directories
            dirs[:] = [d for d in dirs if not should_skip(os.path.join(root, d))]
            for file in files:
                filepath = os.path.join(root, file)
                if should_skip(filepath):
                    continue
                arcname = os.path.relpath(filepath, base_dir)
                zipf.write(filepath, arcname)
                print(f"Added: {arcname}")
    print(f"\n✅ Created {zip_filename} without venv, node_modules, or build artifacts.")

if __name__ == "__main__":
    repo_root = os.path.abspath(".")
    zip_clean_repo(repo_root, ZIP_FILENAME)
