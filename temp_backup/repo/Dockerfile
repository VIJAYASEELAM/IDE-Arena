FROM node:18-slim

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Create non-root user (Required for grading)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command
ENTRYPOINT ["./run_tests.sh"]