#!/bin/bash

# Start Backend script

echo "Starting Vapor Backend..."

# Build if needed
echo "Building Swift project..."
swift build

# Run the service
echo "Starting Vapor service on port 8080..."
swift run App serve --hostname 0.0.0.0 --port 8080

