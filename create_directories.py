import os

directories = ['data', 'src', 'notebooks', 'logs', 'models', 'reports']

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

