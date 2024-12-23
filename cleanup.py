# cleanup.py

import os
import shutil

def cleanup_python_cache(directory):
    """
    Recursively remove Python cache files and directories from the given directory.
    This includes __pycache__, .pyc, .pyo, and .pyd files.
    """
    for root, dirs, files in os.walk(directory):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print(f"Removing directory: {pycache_path}")
            shutil.rmtree(pycache_path)
            dirs.remove('__pycache__')  # Prevent os.walk from recursing into __pycache__

        # Remove .pyc, .pyo, and .pyd files
        for file in files:
            if file.endswith(('.pyc', '.pyo', '.pyd')):
                cache_file_path = os.path.join(root, file)
                print(f"Removing file: {cache_file_path}")
                os.remove(cache_file_path)

def cleanup_pytest_cache(directory):
    """
    Remove pytest cache directory and files.
    """
    pytest_cache = os.path.join(directory, '.pytest_cache')
    if os.path.exists(pytest_cache):
        print(f"Removing directory: {pytest_cache}")
        shutil.rmtree(pytest_cache)

def cleanup_coverage_files(directory):
    """
    Remove coverage files generated by pytest-cov.
    """
    for file in os.listdir(directory):
        if file.startswith('.coverage') or file == 'coverage.xml' or file == 'htmlcov':
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                print(f"Removing file: {file_path}")
                os.remove(file_path)
            elif os.path.isdir(file_path):
                print(f"Removing directory: {file_path}")
                shutil.rmtree(file_path)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    cleanup_python_cache(project_root)
    cleanup_pytest_cache(project_root)
    cleanup_coverage_files(project_root)
    print("Cleanup completed.")