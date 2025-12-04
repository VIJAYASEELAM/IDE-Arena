FROM node:18-slim
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
ENTRYPOINT ["./run_tests.sh"]
