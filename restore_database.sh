#!/bin/bash
# This script restores the production database from a specified backup file.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
BACKUP_DIR="./backups"
DB_SERVICE="db"
DB_CONTAINER_NAME="timescaledb" # The explicit container_name from docker-compose.yml
APP_SERVICE="backend"           # The application service that connects to the DB
DB_USER="strader_user"
DB_NAME="strader_db"

# Get the directory of the script to ensure paths are correct
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# --- Cleanup Function ---
# This function will be called on script exit to ensure the app container is restarted.
function cleanup {
    echo "---"
    echo "Restarting application service '$APP_SERVICE'..."
    docker compose start "$APP_SERVICE"
    echo "Application service restarted."
}

# --- Main Logic ---
echo "--- Starting Database Restore ---"

# 1. Check if a backup file was provided as an argument.
if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file_name>"
    echo "       (The file is assumed to be in the '$BACKUP_DIR' directory)"
    echo ""
    echo "Available backups in $BACKUP_DIR:"
    ls -1 "$BACKUP_DIR" | grep \.dump
    exit 1
fi

# 2. Construct the full path to the backup file.
if [[ "$1" == */* ]]; then
    BACKUP_FILE="$1"
else
    BACKUP_FILE="$BACKUP_DIR/$1"
fi

# 3. Verify the backup file exists.
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found at '$BACKUP_FILE'" >&2
    exit 1
fi

# 4. CRITICAL: Warn the user and ask for confirmation.
echo ""
_RED='\033[0;31m'
_NC='\033[0m' # No Color
printf "${_RED}WARNING: This is a DESTRUCTIVE operation.${_NC}\n"
printf "The existing database '$DB_NAME' will be dropped and completely replaced with the contents of the backup.\n"
printf "The application service ('$APP_SERVICE') will be temporarily stopped during the restore.\n"
printf "Are you sure you want to continue? (type 'yes' to proceed): "
read -r CONFIRMATION

if [ "$CONFIRMATION" != "yes" ]; then
    echo "Restore cancelled by user." >&2
    exit 1
fi

# 5. Perform the restore.
# Register the cleanup function to run on script exit (normal or error).
trap cleanup EXIT

echo "---"
echo "Stopping application service '$APP_SERVICE' to release database connections..."
docker compose stop "$APP_SERVICE"
echo "Application service stopped."

echo "Starting restore process..."

# We need to copy the backup file into the container to make it accessible to pg_restore.
CONTAINER_BACKUP_PATH="/tmp/$(basename "$BACKUP_FILE")"
docker cp "$BACKUP_FILE" "$DB_CONTAINER_NAME:$CONTAINER_BACKUP_PATH"

# Execute pg_restore inside the container.
docker compose exec -T "$DB_SERVICE" pg_restore \
    -U "$DB_USER" \
    -d postgres \
    --clean \
    --create \
    --exit-on-error \
    "$CONTAINER_BACKUP_PATH"

# 6. Clean up the copied backup file from the container.
docker compose exec -T "$DB_SERVICE" rm "$CONTAINER_BACKUP_PATH"

echo "--- Database Restore Complete ---"
# The 'trap' will automatically call the cleanup function to restart the app service.
