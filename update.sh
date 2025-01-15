#!/bin/bash

# Copy updated files
scp -i chatchipmunk-key.pem chatbot.py ec2-user@44.204.108.174:~/
scp -i chatchipmunk-key.pem .env ec2-user@44.204.108.174:~/
scp -i chatchipmunk-key.pem requirements.txt ec2-user@44.204.108.174:~/

# SSH and rebuild/restart
ssh -i chatchipmunk-key.pem ec2-user@44.204.108.174 '
    # Rebuild the image with new code
    docker build -t chatchipmunk-api .
    
    # Restart the container
    docker-compose down
    docker-compose up -d
' 

# For code changes: ./update.sh
# For env only: ./update.sh env