import hashlib
import os
import argparse

def load_folder(path: str) -> set:
    '''
    Load all files in the folder in sha512.
    Parameters:
        path: The path to the folder.
    Returns:
        A set of sha512 of all files in the folder.
    '''
    hashset = set()
    for root, dirs, files in os.walk(path):
        for file in files:
            hashset.add(hashlib.sha512(open(os.path.join(root, file), 'rb').read()).hexdigest())
    return hashset

def check_folder(path: str, hashset: set) -> tuple[list[str], list[str]]:
    '''
    Check all files in the folder.
    Parameters:
        path: The path to the folder.
        hashset: A set of sha512 of all files to compare.
    Returns:
        unique: A list of unique files.
        duplicated: A list of duplicated files.
    '''
    unique = []
    duplicated = []
    for root, dirs, files in os.walk(path):
        for file in files:
            sha = hashlib.sha512(open(os.path.join(root, file), 'rb').read()).hexdigest()
            if sha in hashset:
                duplicated.append(file)
            else:
                unique.append(file)
    return unique, duplicated

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Check sha512 of all files in the folder.")
    parser.add_argument('--src', type=str, required=True, help='The path to the folder of source.')
    parser.add_argument('--dst', type=str, required=True, help='The path to the folder of destination.')
    parser.add_argument('--cnt', action='store_true', help='Print the count of unique and duplicated files.')
    parser.add_argument('--output', type=str, help='The path to the output file.')
    parser.add_argument('--unique-only', action='store_true', help='Only print the unique files.')
    parser.add_argument('--duplicated-only', action='store_true', help='Only print the duplicated files.')

    args = parser.parse_args()
    src_hashset = load_folder(args.src)
    unique, duplicated = check_folder(args.dst, src_hashset)
    if args.cnt:
        print(f'Unique: {len(unique)}\nDuplicated: {len(duplicated)}')
    if args.output:
        with open(args.output, 'w') as f:
            if not args.duplicated_only:
                f.write("Unique:\n")
                for file in unique:
                    f.write(file + '\n')
            if not args.unique_only:
                f.write("\nDuplicated:\n")
                for file in duplicated:
                    f.write(file + '\n')
