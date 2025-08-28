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
            raise ConfigError(
                f"Configuration file .env not found at the expected location: {project_root}"
            )

        load_dotenv(dotenv_path=dotenv_path)

        self.db_name = os.getenv("DB_NAME")
        self.db_user = os.getenv("DB_USER")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", 5432))

        
        if not all([self.db_name, self.db_user, self.db_password]):
            raise ConfigError(
                "One or more required database environment variables (DB_NAME, DB_USER, DB_PASSWORD) are not set in your .env file."
            )



config = Config()
