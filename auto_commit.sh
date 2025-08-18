#!/bin/bash

# MyVoice Auto Commit Script
# This script automatically commits and pushes changes to the GitHub repository

# セキュリティ設定
set -euo pipefail
umask 077

# Set the project directory
PROJECT_DIR="/home/ryuga/MyVoice"
readonly PROJECT_DIR
LOG_FILE="$PROJECT_DIR/auto_commit.log"

# Function to log messages with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Change to project directory
cd "$PROJECT_DIR" || {
    log_message "ERROR: Cannot change to project directory: $PROJECT_DIR"
    exit 1
}

log_message "Starting auto commit process..."

# Check if GITHUB_TOKEN is set
if [ -z "${GITHUB_TOKEN:-}" ]; then
    log_message "ERROR: GITHUB_TOKEN environment variable not set"
    exit 1
fi

# Set git remote URL with token
git remote set-url origin https://0rnot:${GITHUB_TOKEN}@github.com/0rnot/MyVoice.git

# Add all changes (respecting .gitignore) - 機密情報フィルタリング
git add . 2>&1 | grep -v "password\|token\|key\|credential" | tee -a "$LOG_FILE"

# Check if there are staged changes
if git diff --cached --quiet; then
    log_message "No staged changes after git add. Creating empty commit."
    COMMIT_MSG="Auto commit (no changes) - $(date '+%Y-%m-%d %H:%M:%S')

- Automatic daily commit of MyVoice project
- No changes detected on $(date '+%A, %B %d, %Y at %H:%M:%S')

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    COMMIT_ARGS="--allow-empty"
else
    log_message "Changes detected. Creating regular commit."
    COMMIT_MSG="Auto commit - $(date '+%Y-%m-%d %H:%M:%S')

- Automatic daily commit of MyVoice project
- Updated on $(date '+%A, %B %d, %Y at %H:%M:%S')

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    COMMIT_ARGS=""
fi

# Commit changes - 機密情報フィルタリング
if git commit $COMMIT_ARGS -m "$COMMIT_MSG" 2>&1 | grep -v "password\|token\|key\|credential" | tee -a "$LOG_FILE"; then
    log_message "Successfully committed changes"
    
    # Push to remote repository - 機密情報フィルタリング
    if git push origin master 2>&1 | grep -v "password\|token\|key\|credential" | tee -a "$LOG_FILE"; then
        log_message "Successfully pushed to GitHub"
    else
        log_message "ERROR: Failed to push to GitHub"
        exit 1
    fi
else
    log_message "ERROR: Failed to commit changes"
    exit 1
fi

log_message "Auto commit process completed successfully"