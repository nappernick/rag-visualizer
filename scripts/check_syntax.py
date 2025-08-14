#!/usr/bin/env python3
"""Fast syntax and import checker for all Python files"""
import os
import sys
import ast
import importlib.util
from pathlib import Path

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports(filepath):
    """Check if all imports in a file can be resolved"""
    errors = []
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just check imports
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            __import__(alias.name)
                        except ImportError as e:
                            errors.append(f"Cannot import {alias.name}: {e}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        try:
                            __import__(node.module)
                        except ImportError as e:
                            if "src." not in node.module:  # Skip internal imports
                                errors.append(f"Cannot import from {node.module}: {e}")
    except Exception as e:
        errors.append(str(e))
    
    return errors

def check_undefined_names(filepath):
    """Check for undefined names in Python files"""
    errors = []
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        tree = ast.parse(source)
        
        # Track defined names
        defined = set()
        # Add builtins
        defined.update(dir(__builtins__))
        
        # First pass: collect all definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname if alias.asname else alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != '*':
                        defined.add(alias.asname if alias.asname else alias.name)
        
        # Check for common service name issues
        service_patterns = [
            ('graph_service', 'get_graph_service'),
            ('entity_service', 'get_entity_service'),
            ('storage_service', 'get_storage_service'),
            ('vector_service', 'get_vector_service'),
            ('embedding_service', 'get_embedding_service'),
        ]
        
        lines = source.split('\n')
        for i, line in enumerate(lines, 1):
            for var_name, func_name in service_patterns:
                if var_name in line and func_name not in source[:source.find(line)]:
                    if 'Depends(' not in line and '=' not in line.split(var_name)[0]:
                        errors.append(f"Line {i}: '{var_name}' used but not defined. Did you mean to call '{func_name}()'?")
                        
    except Exception as e:
        errors.append(str(e))
    
    return errors

def main():
    src_dir = Path(__file__).parent / "src"
    
    print("🔍 Checking Python syntax and imports...\n")
    
    total_files = 0
    syntax_errors = []
    import_errors = []
    undefined_errors = []
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                total_files += 1
                
                # Check syntax
                valid, error = check_file_syntax(filepath)
                if not valid:
                    syntax_errors.append((filepath, error))
                
                # Check imports
                import_errs = check_imports(filepath)
                if import_errs:
                    import_errors.append((filepath, import_errs))
                
                # Check undefined names
                undefined_errs = check_undefined_names(filepath)
                if undefined_errs:
                    undefined_errors.append((filepath, undefined_errs))
    
    # Report results
    print(f"Checked {total_files} Python files\n")
    
    if syntax_errors:
        print("❌ SYNTAX ERRORS:")
        for filepath, error in syntax_errors:
            print(f"  {filepath.relative_to(src_dir.parent)}:")
            print(f"    {error}\n")
    
    if undefined_errors:
        print("❌ UNDEFINED NAME ERRORS:")
        for filepath, errors in undefined_errors:
            print(f"  {filepath.relative_to(src_dir.parent)}:")
            for error in errors:
                print(f"    {error}")
            print()
    
    if import_errors:
        print("⚠️  IMPORT WARNINGS (may be false positives):")
        for filepath, errors in import_errors:
            print(f"  {filepath.relative_to(src_dir.parent)}:")
            for error in errors[:3]:  # Limit to first 3
                print(f"    {error}")
            print()
    
    if not syntax_errors and not undefined_errors:
        print("✅ All files have valid syntax and no undefined service issues!")
        return 0
    else:
        print(f"❌ Found {len(syntax_errors)} syntax errors and {len(undefined_errors)} undefined name issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())