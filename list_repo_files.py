import os
import argparse
from pathlib import Path

def get_human_readable_size(size_bytes):
    """Converts bytes to KB, MB, GB, etc."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.
        i += 1
    # Format with 1 decimal place for KB and above
    f = f"{size_bytes:.1f}" if i > 0 else f"{size_bytes:.0f}"
    return f"{f} {size_name[i]}"

def list_directory_contents(start_path, ignore_dirs=None, ignore_ext=None):
    """
    Recursively lists directory contents with sizes and paths.

    Args:
        start_path (str): The root directory to start listing from.
        ignore_dirs (list): List of directory names to ignore.
        ignore_ext (list): List of file extensions to ignore.
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.ipynb_checkpoints', 'venv', 'env']
    if ignore_ext is None:
        ignore_ext = ['.pyc', '.log', '.DS_Store'] # Add other extensions if needed

    print(f"Directory Outline for: {Path(start_path).resolve()}")
    print("-" * 60)

    total_size = 0
    file_count = 0

    for root, dirs, files in os.walk(start_path, topdown=True):
        # Filter out ignored directories *before* descending into them
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')

        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            # Filter out ignored extensions
            if any(f.endswith(ext) for ext in ignore_ext):
                continue

            file_path = os.path.join(root, f)
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
                print(f'{sub_indent}{f} ({get_human_readable_size(file_size)}) - Path: {file_path}')
            except OSError as e:
                print(f'{sub_indent}{f} - Error accessing: {e}')

    print("-" * 60)
    print(f"Total Files Scanned (excluding ignored): {file_count}")
    print(f"Total Size (excluding ignored): {get_human_readable_size(total_size)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List directory contents with file sizes and paths.")
    parser.add_argument(
        "start_dir",
        nargs='?', # Makes the argument optional
        default='.', # Default to current directory
        help="The directory to start listing from (default: current directory)."
    )
    args = parser.parse_args()

    start_directory = args.start_dir
    if not os.path.isdir(start_directory):
        print(f"Error: Directory not found - {start_directory}")
    else:
        list_directory_contents(start_directory)