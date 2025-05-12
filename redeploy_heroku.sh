#!/bin/bash
# Script to quickly redeploy to Heroku after making fixes

echo "Committing changes..."
git add app.py Procfile
git commit -m "Fix Method Not Allowed error by updating Heroku configuration"

echo "Pushing to Heroku..."
git push heroku main || git push heroku master

echo "Ensuring Heroku environment variables are set..."
heroku config:set FLASK_ENV=production
heroku config:set IS_HEROKU=True

echo "Restarting the application..."
heroku restart

echo "Viewing logs to check for errors..."
heroku logs --tail 