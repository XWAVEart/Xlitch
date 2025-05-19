#!/bin/bash
# Helper script to deploy to Heroku

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "Error: Heroku CLI is not installed. Please install it first."
    echo "Visit: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Check if Heroku app is already set up
if ! heroku apps:info &> /dev/null; then
    echo "No Heroku app found. Creating a new one..."
    echo "What name would you like for your Heroku app? (leave blank for random name)"
    read app_name
    
    if [ -z "$app_name" ]; then
        heroku create
    else
        heroku create "$app_name"
    fi
    
    # Set buildpack
    heroku buildpacks:set heroku/python
    
    # Create upload and processed directories
    heroku config:set CREATE_DIRS=True
fi

# Set necessary environment variables
echo "Setting environment variables..."
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=$(openssl rand -hex 32)

# Push to Heroku
echo "Pushing to Heroku..."
git add .
git commit -m "Deploy to Heroku: $(date)"
git push heroku master || git push heroku main

echo "Deployment complete! Your app should be available at:"
heroku open

echo "To view logs, run: heroku logs --tail" 