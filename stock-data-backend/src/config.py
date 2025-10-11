import os
from dotenv import load_dotenv

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Handles application configuration by loading environment variables from a .env file.
    """
    def __init__(self):
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dotenv_path = os.path.join(project_root, '.env')

        if not os.path.exists(dotenv_path):
            # This is not an error in a containerized environment where env vars are passed directly.
            print(f"Info: .env file not found at {dotenv_path}. Assuming environment variables are set directly.")
            load_dotenv() # Load from environment if .env is not present
        else:
            load_dotenv(dotenv_path=dotenv_path)

        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", 5432))
        self.news_api_key = os.getenv("NEWS_API_KEY")

        
        if not all([self.db_name, self.db_user, self.db_password]):
            raise ConfigError(
                "One or more required database environment variables (DB_NAME, DB_USER, DB_PASSWORD) are not set."
            )

config = Config()
