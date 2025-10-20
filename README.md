# SingN'Seek 🎵

A Streamlit-based application for song search and discovery.

## Docker Deployment

This application has been containerized and can be run using Docker.

### Prerequisites

- Docker installed on your system ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose (included with Docker Desktop)

### Quick Start with Docker Compose

1. **Build and run the application**:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

3. **Stop the application**:
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build the Docker image**:
   ```bash
   docker build -t singnseek:latest .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 --name singnseek-app singnseek:latest
   ```

3. **Stop and remove the container**:
   ```bash
   docker stop singnseek-app
   docker rm singnseek-app
   ```

### Docker Commands Reference

- **View running containers**:
  ```bash
  docker ps
  ```

- **View logs**:
  ```bash
  docker-compose logs -f
  # or for direct docker
  docker logs -f singnseek-app
  ```

- **Rebuild without cache**:
  ```bash
  docker-compose build --no-cache
  docker-compose up
  ```

- **Run in detached mode (background)**:
  ```bash
  docker-compose up -d
  ```

### Local Development Setup (Without Docker)

The application can be run locally without Docker using standard Python tools:

1. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run main.py
   ```

4. **Deactivate virtual environment** (when done):
   ```bash
   deactivate
   ```

### Environment Variables

The following environment variables can be configured in `docker-compose.yml`:

- `STREAMLIT_SERVER_PORT`: Port for Streamlit server (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)
- `STREAMLIT_SERVER_HEADLESS`: Run without browser (default: true)
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: Disable usage stats (default: false)

### Project Structure

```
singnseek/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore          # Docker ignore patterns
├── .streamlit/            # Streamlit configuration
│   └── config.toml
├── images/                # Application images
└── dataset/               # Application dataset
```

### Troubleshooting

**Issue**: Container fails to start
- Check logs: `docker-compose logs`
- Ensure port 8501 is not already in use
- Verify all required files are present

**Issue**: Application not accessible
- Ensure container is running: `docker ps`
- Check if firewall is blocking port 8501
- Try accessing: `http://localhost:8501` or `http://127.0.0.1:8501`

**Issue**: Changes not reflected
- Rebuild without cache: `docker-compose build --no-cache`
- Ensure volumes are properly mounted in `docker-compose.yml`

### Production Considerations

For production deployment:

1. Remove volume mounts in `docker-compose.yml` to bake data into the image
2. Consider using environment-specific configurations
3. Set up proper logging and monitoring
4. Use a reverse proxy (nginx) for SSL/TLS termination
5. Implement proper secrets management for sensitive data

### License

[Add your license information here]

### Contributing

[Add contribution guidelines here]
