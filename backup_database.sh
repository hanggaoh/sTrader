#!/bin/bash
# This script creates a compressed backup of the production database.

# --- Script Configuration ---
# set -e: Exit immediately if a command exits with a non-zero status.
# set -o pipefail: The return value of a pipeline is the status of the last command
#                  to exit with a non-zero status, or zero if no command exited
#                  with a non-zero status.
set -e
set -o pipefail

# --- Configuration ---
BACKUP_DIR="./backups"
DB_SERVICE="db"
DB_USER="strader_user"
DB_NAME="strader_db"

# Get the directory of the script to ensure paths are correct
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# --- Main Logic ---
echo "--- Starting Database Backup ---"

# 1. Create the backup directory if it doesn't exist.
mkdir -p "$BACKUP_DIR"

# 2. Define the backup file path with a timestamp.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/backup-$TIMESTAMP.dump"

# 3. Execute pg_dump inside the running container.
#    The --verbose flag is added to show progress during the dump.
#    The output is piped directly to the backup file.
#    With `set -o pipefail`, if pg_dump fails, the script will exit immediately.
echo "Dumping database '$DB_NAME' to $BACKUP_FILE..."
docker compose exec -T "$DB_SERVICE" pg_dump --verbose -U "$DB_USER" -d "$DB_NAME" -F c > "$BACKUP_FILE"

echo "Database dump successful."

# 4. Clean up old backups (older than 7 days).
echo "Cleaning up backups older than 7 days..."
find "$BACKUP_DIR" -type f -mtime +7 -name '*.dump' -print -delete

echo "--- Database Backup Complete ---"
