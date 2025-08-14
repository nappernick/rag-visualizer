#!/usr/bin/env python3
"""Fix all service import issues automatically"""
import re
from pathlib import Path

def fix_service_references(filepath):
    """Fix service references in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix patterns where services are used without being defined
    patterns = [
        (r'\basync def \w+\([^)]*\):', 'function_def'),
        (r'def \w+\([^)]*\):', 'function_def'),
    ]
    
    service_fixes = {
        'storage_service': 'get_storage_service',
        'vector_service': 'get_vector_service', 
        'embedding_service': 'get_embedding_service',
        'graph_service': 'get_graph_service',
        'entity_service': 'get_entity_service',
    }
    
    # Find function definitions
    functions = []
    for pattern, ptype in patterns:
        for match in re.finditer(pattern, content):
            functions.append((match.start(), match.group()))
    
    # For each function, check if it uses undefined services
    for service_var, service_func in service_fixes.items():
        # Check if service is used without Depends
        pattern = rf'(?<!\w){service_var}\.(?=\w)'
        
        for match in re.finditer(pattern, content):
            pos = match.start()
            
            # Find which function this is in
            func_start = 0
            for fpos, fdef in functions:
                if fpos < pos:
                    func_start = fpos
            
            # Check if service_func is called before this usage
            func_content = content[func_start:pos]
            if f'{service_var} = {service_func}()' not in func_content:
                if 'Depends(' not in content[pos-100:pos]:
                    # Need to add the service initialization
                    # Find the line start
                    line_start = content.rfind('\n', 0, pos) + 1
                    indent = len(content[line_start:pos]) - len(content[line_start:pos].lstrip())
                    
                    # Add service initialization at the start of the function
                    # Find the first line after function definition
                    func_body_start = content.find('\n', func_start) + 1
                    next_line_start = content.find('\n', func_body_start) + 1
                    
                    # Get indent of the function body
                    body_indent = '    '
                    if next_line_start < len(content):
                        next_line = content[next_line_start:content.find('\n', next_line_start)]
                        if next_line.strip():
                            body_indent = next_line[:len(next_line) - len(next_line.lstrip())]
                    
                    # Check if we haven't already added it
                    check_str = f'{service_var} = {service_func}()'
                    if check_str not in content[func_start:pos]:
                        # Insert the service initialization
                        insert_pos = func_body_start
                        # Skip any docstring
                        if '"""' in content[func_body_start:func_body_start+10]:
                            # Find end of docstring
                            doc_end = content.find('"""', func_body_start+3)
                            insert_pos = content.find('\n', doc_end) + 1
                        
                        content = (content[:insert_pos] + 
                                  f'{body_indent}{service_var} = {service_func}()\n' + 
                                  content[insert_pos:])
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    # Files with undefined service issues from the check
    files_to_fix = [
        'src/api/documents.py',
        'src/api/graph.py', 
        'src/api/chunking.py',
        'src/api/query.py',
    ]
    
    src_dir = Path(__file__).parent
    
    print("🔧 Fixing service references...\n")
    
    for file_path in files_to_fix:
        filepath = src_dir / file_path
        if filepath.exists():
            if fix_service_references(filepath):
                print(f"✅ Fixed {file_path}")
            else:
                print(f"⏭️  No changes needed in {file_path}")
    
    print("\n✅ Service references fixed!")

if __name__ == "__main__":
    main()