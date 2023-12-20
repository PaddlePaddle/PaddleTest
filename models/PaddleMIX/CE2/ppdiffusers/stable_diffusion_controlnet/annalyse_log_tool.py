import argparse

def check_log(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'FAILED' in line:
                    exit(1)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit(1)
             
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', required=True, help='file_path of the log file')
    args = parser.parse_args()
    file_path = args.file_path
    check_log(file_path)
