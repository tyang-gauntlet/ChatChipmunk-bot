#!/bin/bash

# Copy files to EC2
scp -i chatchipmunk-key.pem docker-compose.yml ec2-user@44.204.108.174:~/
scp -i chatchipmunk-key.pem Dockerfile ec2-user@44.204.108.174:~/
scp -i chatchipmunk-key.pem requirements.txt ec2-user@44.204.108.174:~/
scp -i chatchipmunk-key.pem chatbot.py ec2-user@44.204.108.174:~/
scp -i chatchipmunk-key.pem .env ec2-user@44.204.108.174:~/

# First SSH session to set up Docker
ssh -i chatchipmunk-key.pem ec2-user@44.204.108.174 '
    # Install Docker
    sudo yum update -y
    sudo yum install -y docker
    sudo service docker start
    sudo usermod -a -G docker ec2-user

    # Install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
'

# Second SSH session to build and run (with new group permissions)
ssh -i chatchipmunk-key.pem ec2-user@44.204.108.174 '
    # Build the image
    docker build -t chatchipmunk-api .
    
    # Run with docker-compose
    docker-compose up -d
' 