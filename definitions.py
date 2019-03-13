import os

# This is project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# This is the base data path
BASE_DATA_PATH = os.path.join(ROOT_DIR, 'data')

# This is csv data path
CSV_DATA_PATH = os.path.join(BASE_DATA_PATH, 'csv')

# This is json data path
JSON_DATA_PATH = os.path.join(BASE_DATA_PATH, 'json')

# This is images data path
IMAGES_DATA_PATH = os.path.join(BASE_DATA_PATH, 'images')

# This is png data path
PNG_DATA_PATH = os.path.join(BASE_DATA_PATH, 'png')

# This is the base resources path
BASE_RESOURCES_PATH = os.path.join(ROOT_DIR, 'resources')

# This is the configuration path
BASE_CONFIG_PATH = os.path.join(BASE_RESOURCES_PATH, 'config')
