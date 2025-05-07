import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Memory
MEMORY_PATH = "memory_dir"
FILE_SIZE_LIMIT = 1024 * 1024 # 1MB
DIR_SIZE_LIMIT = 1024 * 1024 * 10 # 10MB
MEMORY_SIZE_LIMIT = 1024 * 1024 * 100 # 100MB

# Engine
SANDBOX_TIMEOUT = 20

# Path settings
SYSTEM_PROMPT_PATH = "obsidian_agent/system_prompt.txt"