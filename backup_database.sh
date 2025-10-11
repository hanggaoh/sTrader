#!/bin/bash
# This script creates a compressed backup of the production database.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration (from environment variables) ---
BACKUP_DIR="${BACKUP_DIR:-./backups}"
DB_HOST="${POSTGRES_HOST?Missing environment variable: POSTGRES_HOST}"
DB_USER="${POSTGRES_USER?Missing environment variable: POSTGRES_USER}"
DB_NAME="${POSTGRES_DB?Missing environment variable: POSTGRES_DB}"
# PGPASSWORD is used by pg_dump automatically if set.
export PGPASSWORD="${POSTGRES_PASSWORD?Missing environment variable: POSTGRES_PASSWORD}"

# --- Main Logic ---
echo "--- Starting Database Backup ---"

# 1. Create the backup directory if it doesn't exist.
mkdir -p "$BACKUP_DIR"

# 2. Define the backup file path with a timestamp.
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/backup-$TIMESTAMP.dump"

# 3. Execute pg_dump, connecting to the DB host.
#    The --verbose flag is added to show progress during the dump.
echo "Dumping database '$DB_NAME' from host '$DB_HOST' to $BACKUP_FILE..."
pg_dump --verbose --host="$DB_HOST" --username="$DB_USER" --dbname="$DB_NAME" -F c > "$BACKUP_FILE"

echo "Database dump successful."

# 4. Clean up old backups (older than 7 days).
echo "Cleaning up backups older than 7 days..."
find "$BACKUP_DIR" -type f -mtime +7 -name '*.dump' -print -delete

echo "--- Database Backup Complete ---"
