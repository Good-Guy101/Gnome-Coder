import os
import sys
import hashlib
from pathlib import Path

DIR = "./test"

files = os.listdir(DIR)
unique = dict()

for file in files:
	file_path = Path(os.path.join(DIR, file))
	hash_file = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
	if hash_file not in unique:
		unique[hash_file] = file_path
	else:
		os.remove(file_path)
		print(f"{file_path} has been deleted")

files = os.listdir(DIR)
count = 0
for file in files:
	file_path = Path(os.path.join(DIR, file))
	os.rename(file_path, Path(os.path.join(DIR, str(count) + 'g.jpg')))
	count += 1