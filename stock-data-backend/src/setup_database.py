import os
from data.storage import Storage
from config import config

def main():
    """
    Connects to the database and runs the initial schema setup.
    This should be run once to prepare the database for the application.
    """
    print("--- Database Setup Script ---")
    storage = None
    try:
        storage = Storage(config=config)
        storage.setup_database()
    except Exception as e:
        print(f"\nAn error occurred during database setup: {e}")
        print("Please ensure the database container is running and the .env file is configured correctly.")
    finally:
        if storage:
            storage.close()
    print("--- Setup script finished ---")

if __name__ == "__main__":
    main()