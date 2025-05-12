#!/bin/bash
# Script to deploy the fixes and debugging tools to Heroku

echo "Committing changes..."
git add app.py Procfile simple_app.py check_path.py requirements.txt
git commit -m "Add debugging tools and simplified app for Heroku deployment troubleshooting"

echo "Pushing to Heroku..."
git push heroku main || git push heroku master

echo "Running debug check script on Heroku..."
heroku run python check_path.py

echo "Ensuring Heroku environment variables are set..."
heroku config:set FLASK_ENV=production
heroku config:set IS_HEROKU=True

echo "Restarting the application..."
heroku restart

echo "Viewing logs to check for errors..."
heroku logs --tail 