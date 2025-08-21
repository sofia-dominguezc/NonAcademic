import json

def count_lines_of_code(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_lines = 0
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code_lines += len(cell['source'])
    
    return code_lines

# Example usage
notebook_path = 'python_files\diffussion_models_MNIST.ipynb'
lines_of_code = count_lines_of_code(notebook_path)
print(f'Total lines of code: {lines_of_code}')
