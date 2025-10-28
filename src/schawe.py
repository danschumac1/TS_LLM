'''
2025-10-24
Author: Dan Schumacher
How to run:
   python ./src/schawe.py
'''

from typing import Tuple

def main():
    errors = [1.5, -6.0, 0.5, -1.2, 3.0, -0.7]
    sorted_list = sorted(errors, key=lambda error: error *-1 if error< 0 else error)
    print(sorted_list)

if __name__ == "__main__":
    main()