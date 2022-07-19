import os

# get list of files in current directory
files = os.listdir()

# get last modified file in current directory
last_modified = max(files, key=os.path.getmtime)
