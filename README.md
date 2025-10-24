# SingN'Seek

**Can't remember the song? Just hum, type, or guess — we'll find it!**

A powerful multimodal song search application that enables searching songs by text (lyrics, metadata, natural language queries) and audio (humming, recordings) using Elasticsearch and Google Vertex AI.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It's Different](#how-its-different)
- [Architecture](#architecture)
- [Models Used](#models-used)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Usage](#usage)
- [Query Vetter (Natural Language Processing)](#query-vetter-natural-language-processing)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Query Samples](#query-samples)
- [License](#license)

---

## Overview

SingN'Seek is an intelligent music search engine that combines the power of traditional text search, semantic embeddings, and audio fingerprinting to deliver accurate song retrieval. Whether you remember a few lyrics, know the artist's name, or can only hum a melody, SingN'Seek helps you find the song you're looking for.

Built with Elasticsearch for scalable search infrastructure and Google Vertex AI for state-of-the-art embeddings, this application demonstrates the practical implementation of multimodal retrieval systems.

---

## Key Features

### Multimodal Search Capabilities
- **Text Search**: Search by song name, lyrics, composer, artist, genre, or album
- **Natural Language Queries**: Use conversational queries like "pop songs by voiceofruthie about apocalypse"
- **Audio Search**: Upload audio files or record live audio (humming, singing) to find matching songs
- **Hybrid Search**: Combines BM25 keyword matching with vector similarity for optimal results

### Intelligent Query Processing
- **Query Vetter**: Powered by Gemini 2.5 Flash Lite to parse natural language into structured Elasticsearch queries
- **Field-Specific Filtering**: Automatically detects and filters by composer, artist, genre, album, and more
- **Lyrics-Aware Search**: Identifies lyrics queries and applies semantic search with higher relevance scoring
- **Fuzzy Matching**: Handles typos and variations in search terms

### Technical Excellence
- **Scalable Architecture**: Built on Elasticsearch for production-ready performance
- **Real-time Search**: Instant results with relevance scoring and ranking
- **Audio Embeddings**: Uses OpenMuQ's MuQ-large-msd-iter model (1024-dimensional vectors)
- **Text Embeddings**: Google text-embedding-005 model (768-dimensional vectors)
- **Clean UI**: Intuitive Streamlit interface with text-only design

---

## How It's Different

### Comparison with Shazam

**Shazam: Audio Fingerprinting**
- **Approach**: Creates unique "fingerprints" based on spectrogram peaks and time-frequency patterns
- **Advantages**: Extremely fast, works with noisy audio, highly accurate for exact matches
- **Disadvantages**: 
  - Requires the exact recording to be in the database
  - Cannot match covers, remixes, or variations
  - Fails with humming or singing (no exact fingerprint match)
  - Limited to audio-only queries
  - Cannot understand context or semantic meaning

**SingN'Seek: Semantic + Multimodal Search**
- **Approach**: Converts audio and text into semantic embeddings that capture musical features
- **Advantages**:
  - Matches similar songs, covers, and variations (not just exact recordings)
  - Works with humming, singing, or partial audio
  - Supports text queries (lyrics, metadata, natural language)
  - Understands semantic relationships ("songs about love", "upbeat pop tracks")
  - Hybrid scoring combines multiple signals for better ranking
  - Flexible and extensible to new search modalities
- **Use Case**: Better for discovery, exploration, and when you don't have the exact recording

**In Summary**: Shazam excels at identifying exact recordings quickly, while SingN'Seek is designed for semantic understanding and multimodal retrieval, making it ideal for situations where you only remember fragments or characteristics of a song.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit UI (main.py)                     │
│  - Home (Search Interface: Text + Audio Upload + Recording)     │
│  - All Songs (Browse Complete Collection)                       │
│  - Manage (Index Management & Data Loading)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Utility Layer (utils.py)                      │
│  - ElasticsearchClient (Connection & Operations)                │
│  - EmbeddingGenerator (Audio & Text Embeddings)                 │
│  - Query Vetter (Natural Language Parsing)                      │
│  - Search Functions (Hybrid Scoring & Ranking)                  │
│  - Index Management (Create, Delete, Load)                      │
└─────────────┬────────────────────────┬──────────────────────────┘
              │                        │
              ▼                        ▼
┌──────────────────────────┐  ┌───────────────────────────────────┐
│    Elasticsearch         │  │       Google Vertex AI            │
│  - Vector Search (KNN)   │  │  - Text Embeddings                │
│  - BM25 Text Search      │  │    (text-embedding-005)           │
│  - Index Storage         │  │  - Query Parsing                  │
│  - Hybrid Scoring        │  │    (gemini-2.0-flash-lite)        │
└──────────────────────────┘  └───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OpenMuQ Audio Embeddings                       │
│             MuQ-large-msd-iter Model (~300M params)              │
│                  (1024-dimensional vectors)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Indexing Process:**
```
CSV Dataset → Parse Metadata → Generate Text Embeddings (Vertex AI)
                            → Load Audio Files → Generate Audio Embeddings (MuQ)
                            → Combine Metadata + Vectors → Index to Elasticsearch
```

**Search Process:**
```
User Query → [Text Query] → Query Vetter (Gemini) → Parsed Filters + Search Text
          → [Audio Query] → Audio Embedding (MuQ)
          → Elasticsearch Hybrid Search (BM25 + KNN Vector Similarity)
          → Score & Rank Results → Return Top Matches
```

---

## Models Used

### 1. OpenMuQ (MuQ-large-msd-iter)

**Purpose**: Audio embedding generation for semantic music understanding

**Key Details**:
- **Model**: Self-supervised music representation learning model
- **Architecture**: ~300M parameters trained on Million Song Dataset
- **Output**: 1024-dimensional audio embeddings
- **Sample Rate**: Requires 24 kHz audio input
- **Approach**: Uses Mel Residual Vector Quantization (Mel-RVQ) for SSL

**Advantages**:
- State-of-the-art performance on music information retrieval (MIR) tasks
- Captures semantic musical features (melody, rhythm, timbre, mood)
- Works with partial audio, humming, and covers (not just exact matches)
- Robust to variations in recording quality
- Achieves SOTA on MARBLE Benchmark

**Research Paper**: [MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization](https://arxiv.org/abs/2501.01108)

**HuggingFace**: [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter)

---

### 2. Vertex AI text-embedding-005

**Purpose**: Text embedding generation for semantic text understanding

**Key Details**:
- **Model**: Google's latest text embedding model
- **Output**: 768-dimensional text embeddings
- **Languages**: Multi-language support (optimized for English)
- **Use Cases**: Semantic search, similarity matching, classification

**Advantages**:
- High-quality semantic representations
- Excellent performance on text similarity tasks
- Scalable and production-ready
- Integrated with Google Cloud infrastructure

---

### 3. Gemini 2.5 Flash Lite

**Purpose**: Natural language query parsing and structuring

**Key Details**:
- **Model**: Lightweight, fast generative AI model
- **Task**: Parse user queries into structured Elasticsearch filters
- **Latency**: ~200-500ms per query
- **Temperature**: 0.1 (deterministic parsing)

**Advantages**:
- Understands conversational queries
- Extracts structured filters (composer, genre, artist, etc.)
- Cost-optimized for high-volume applications
- Graceful degradation with fallback mechanism

---

## Setup Instructions

### Prerequisites

- **Python**: 3.9 or higher
- **Package Manager**: pip (Python's package manager)
- **Disk Space**: 2GB+ free storage
- **Docker**: For running Elasticsearch (optional for serverless)
- **Google Cloud Account**: For Vertex AI API access

---

### Option 1: Local Setup

#### Step 1: Clone the Repository

```bash
git clone https://github.com/mmohanram13/singnseek.git
cd singnseek
```

#### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Setup Elasticsearch

**For Local Elasticsearch** (using start-local):

```bash
curl -fsSL https://elastic.co/start-local | sh
```

This starts:
- Elasticsearch at `http://localhost:9200`
- Kibana at `http://localhost:5601`
- Default credentials: `elastic` / `changeme`

**For Elasticsearch Cloud** (Serverless):
1. Create an account at [Elastic Cloud](https://cloud.elastic.co/)
2. Create a deployment and note the Cloud ID and API Key
3. Configure in `.env` file (see Configuration section)

**For Self-Managed Elasticsearch**:
Follow the official Elasticsearch installation guide for your platform.

#### Step 5: Configure Google Cloud (Vertex AI)

1. Create a Google Cloud project
2. Enable Vertex AI API
3. Create a service account and download credentials JSON
4. Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
export VERTEX_AI_PROJECT_ID="your-project-id"
export VERTEX_AI_LOCATION="us-central1"
```

#### Step 6: Configure the Application

Create a `.env` file in the project root (or set environment variables):

```bash
# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=changeme

# OR for Elasticsearch Cloud
# ELASTICSEARCH_CLOUD_ID=your-cloud-id
# ELASTICSEARCH_API_KEY=your-api-key

# Google Cloud / Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1
```

Edit `src/config/config.yaml` for application settings (see Configuration section).

#### Step 7: Run the Application

```bash
streamlit run src/main.py
```

The application opens at `http://localhost:8501`

#### Step 8: Initialize the Database

1. Navigate to the **Manage** tab
2. Click **Create Index** button
3. Click **Load Demo Data** button (takes 2-5 minutes depending on dataset size)
4. Wait for success confirmation

---

### Option 2: Docker Setup

#### Using Docker Compose

```bash
docker-compose up --build
```

Access the application at `http://localhost:8501`

To stop:
```bash
docker-compose down
```

#### Using Docker Directly

**Build:**
```bash
docker build -t singnseek:latest .
```

**Run:**
```bash
docker run -p 8501:8501 \
  -e ELASTICSEARCH_URL=http://host.docker.internal:9200 \
  -e ELASTICSEARCH_USERNAME=elastic \
  -e ELASTICSEARCH_PASSWORD=changeme \
  -e VERTEX_AI_PROJECT_ID=your-project-id \
  -e VERTEX_AI_LOCATION=us-central1 \
  -v /path/to/credentials.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  --name singnseek-app \
  singnseek:latest
```

**For Elasticsearch Cloud (Serverless):**
```bash
docker run -p 8501:8501 \
  -e ELASTICSEARCH_CLOUD_ID=your-cloud-id \
  -e ELASTICSEARCH_API_KEY=your-api-key \
  -e VERTEX_AI_PROJECT_ID=your-project-id \
  -e VERTEX_AI_LOCATION=us-central1 \
  -v /path/to/credentials.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  --name singnseek-app \
  singnseek:latest
```

**Stop:**
```bash
docker stop singnseek-app
docker rm singnseek-app
```

---

### Deployment on Google Cloud Run

The Streamlit application can be deployed to Google Cloud Run using Docker:

1. **Build and push Docker image:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/singnseek
   ```

2. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy singnseek \
     --image gcr.io/PROJECT_ID/singnseek \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

3. **Set environment variables** in Cloud Run console or via CLI

---

## Configuration

### config.yaml

Static configuration file located at `src/config/config.yaml`:

```yaml
# Elasticsearch Configuration
elasticsearch:
  index_name: singnseek
  timeout: 30
  max_retries: 3
  retry_on_timeout: true

# Embedding Dimensions
embeddings:
  audio_dims: 1024    # MuQ model output
  text_dims: 768      # Vertex AI text-embedding-005

# Search Parameters
search:
  hybrid_alpha: 0.6   # 60% text (BM25), 40% audio (vector)
  top_k: 5            # Number of results to return
  rerank_top_k: 10    # Results to consider for reranking

# MuQ Model Settings
muq:
  model: OpenMuQ/MuQ-large-msd-iter
  sample_rate: 24000

# Vertex AI Models
vertex_ai:
  text_embedding_model: text-embedding-005
  gemini_model: gemini-2.0-flash-lite

# UI Features
ui:
  enable_save_song: false          # Enable/disable save song button
  enable_initialize_data: false    # Enable/disable data initialization button

# Logging
logging:
  level: INFO
  file: server.log
  max_bytes: 10485760  # 10MB
  backup_count: 5
```

### Environment Variables

Sensitive credentials should be set as environment variables (via `.env` file or system):

```bash
# Elasticsearch (Local)
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=changeme

# Elasticsearch (Cloud/Serverless)
ELASTICSEARCH_CLOUD_ID=your-cloud-id
ELASTICSEARCH_API_KEY=your-api-key

# Google Cloud / Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
VERTEX_AI_PROJECT_ID=your-project-id
VERTEX_AI_LOCATION=us-central1
```

---

## Usage

### Text Search

1. Navigate to the **Home** tab
2. Enter your search query in the text box:
   - Song name: `"Best Thing"`
   - Lyrics: `"lyrics about love"`
   - Natural language: `"pop songs by voiceofruthie"`
   - Metadata: `"song with composer AiCanvas"`
3. Click **Search** button
4. View results with relevance scores and metadata

### Audio Search

**Option A: Upload Audio File**
1. Navigate to the **Home** tab
2. Click **Browse files** under "Upload Audio File"
3. Select a `.wav` file (max 20MB, 24 kHz recommended)
4. Click **Search** button
5. View audio similarity scores and matches

**Option B: Record Live Audio**
1. Navigate to the **Home** tab
2. Click **Record Audio** button
3. Allow microphone permissions
4. Hum, sing, or play the melody
5. Stop recording
6. Click **Search** button
7. View matches based on your recording

### Browse All Songs

1. Navigate to the **All Songs** tab
2. View complete song collection with metadata
3. Play audio samples directly in the browser
4. Explore dataset composition and statistics

### Manage Index

1. Navigate to the **Manage** tab
2. **Create Index**: Initialize a new Elasticsearch index
3. **Load Demo Data**: Index sample songs with embeddings
4. **Delete Index**: Remove the index and all data
5. **View Statistics**: Check index health, document count, and size

---

## Query Vetter (Natural Language Processing)

### Overview

The Query Vetter uses Gemini 2.0 Flash Lite to parse natural language queries into structured Elasticsearch filters. This enables conversational search instead of rigid syntax.

### Features

#### 1. Field-Specific Filtering
Automatically extracts filters when mentioned:
- **Composer**: "song with composer AiCanvas"
- **Artists**: "songs by voiceofruthie"
- **Genre**: "pop songs"
- **Album**: "songs from album Pixabay"

#### 2. Lyrics-Aware Search
Detects lyrics queries and applies:
- Text + semantic hybrid search
- Higher boost for lyrics field
- Vector similarity matching

Examples:
- "lyrics about love"
- "songs with apocalypse in the lyrics"

#### 3. Combined Filtering
Multiple filters in one query:
- "pop songs by voiceofruthie about apocalypse"
  - Filters: genre=Pop, singers=voiceofruthie
  - Search: "apocalypse"

### Query Vetter Architecture

```
User Query → Query Vetter (Gemini) → Parsed Structure → Elasticsearch Query
                                                          ↓
                                    [Filters] + [Search Text] + [Search Type]
```

### Example Queries

**Example 1: Field Filter**
- **Query**: "song with composer AiCanvas"
- **Parsed**: `{"filters": {"composer": "AiCanvas"}, "search_text": ""}`
- **Result**: All songs by composer AiCanvas

**Example 2: Lyrics Search**
- **Query**: "lyrics about love and apocalypse"
- **Parsed**: `{"filters": {}, "search_text": "love and apocalypse", "search_type": "lyrics", "use_hybrid": true}`
- **Result**: Hybrid search on lyrics with semantic matching

**Example 3: Combined**
- **Query**: "pop songs by voiceofruthie about apocalypse"
- **Parsed**: `{"filters": {"genre": "Pop", "singers": "voiceofruthie"}, "search_text": "apocalypse"}`
- **Result**: Pop songs by artist containing "apocalypse"

### Implementation

Located in `src/utils/query_vetter.py`:
- `parse_query(user_query)`: Main parsing entry point
- Uses structured prompts with few-shot examples
- Temperature 0.1 for consistent parsing
- Fallback to basic search if parsing fails

---

## Docker Deployment

### docker-compose.yml

```yaml
version: '3.8'

services:
  singnseek:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200
      - VERTEX_AI_PROJECT_ID=${VERTEX_AI_PROJECT_ID}
    volumes:
      - ./credentials.json:/app/credentials.json
    depends_on:
      - elasticsearch

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
```

### Dockerfile

The application uses a multi-stage build optimized for production:
- Base image: `python:3.11-slim`
- Installs system dependencies and Python packages
- Copies application code
- Exposes port 8501
- Health check endpoint
- Runs Streamlit server

---

## Project Structure

```
singnseek/
├── src/
│   ├── main.py                      # Streamlit UI application
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.yaml              # Static configuration
│   ├── converters/
│   │   ├── __init__.py
│   │   └── mp3_to_wav_converter.py  # Audio format conversion
│   └── utils/
│       ├── __init__.py
│       ├── utils.py                 # Core backend utilities
│       ├── query_vetter.py          # NLP query parsing
│       ├── logging_config.py        # Logging setup
│       └── muq_test.py              # Audio embedding tests
├── test/
│   └── test_query_vetter.py         # Query vetter unit tests
├── dataset/
│   ├── dataset_meta.csv             # Song metadata
│   └── *.wav                        # Audio files
├── images/
│   └── logo.png                     # Application logo
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker image definition
├── docker-compose.yml               # Docker Compose configuration
├── docker-start.sh                  # Docker startup script
└── README.md                        # This file
```

---

## Troubleshooting

### Elasticsearch Connection Issues

**Problem**: Cannot connect to Elasticsearch

**Solutions**:
- Verify Elasticsearch is running: `curl http://localhost:9200`
- Check credentials in `.env` file
- Ensure port 9200 is not blocked by firewall
- For Docker: Use `http://host.docker.internal:9200` instead of `localhost`

### Audio Embedding Errors

**Problem**: MuQ model fails to generate embeddings

**Solutions**:
- Verify audio files are in WAV format (use `converters/mp3_to_wav_converter.py` if needed)
- Check sample rate is 24 kHz: `ffmpeg -i input.wav`
- Ensure MuQ model is downloaded (auto-downloads on first run to `~/.cache/huggingface/`)
- Check available disk space (model is ~1.2GB)

### Vertex AI Authentication Errors

**Problem**: Cannot access Vertex AI APIs

**Solutions**:
- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to valid JSON key file
- Check service account has required permissions (Vertex AI User, AI Platform User)
- Ensure Vertex AI API is enabled in your GCP project
- Verify project ID and region are correct

### Memory Issues

**Problem**: Out of memory errors during indexing

**Solutions**:
- Reduce batch size in `load_demo_data()` function
- Use CPU instead of GPU if VRAM is limited (set `device='cpu'`)
- Close other applications to free up memory
- For Docker: Increase memory allocation in Docker settings

### Search Returns No Results

**Problem**: Queries return empty results

**Solutions**:
- Verify index is created: Check **Manage** tab
- Ensure data is loaded: Check document count in **Manage** tab
- Try broader search terms
- Check Elasticsearch logs for errors
- Verify embeddings were generated during indexing

### Query Vetter Parsing Errors

**Problem**: Natural language queries not parsed correctly

**Solutions**:
- Check Gemini API credentials and quota
- Review query vetter logs for errors
- Try simpler query phrasing
- Verify `gemini_model` setting in `config.yaml`
- System falls back to basic search if vetter fails

---

## Query Samples

### Text Queries

**Basic Search:**
- "Best Thing"
- "apocalypse"
- "hollow"

**Field-Specific:**
- "song with composer AiCanvas"
- "songs by voiceofruthie"
- "artist Rangga Fermata"
- "pop songs"
- "rhythm and blues tracks"
- "songs from album Pixabay"

**Lyrics Search:**
- "lyrics about love"
- "find lyrics with the word hollow"
- "songs with apocalypse in the lyrics"
- "lyrics that mention heartbreak"

**Combined Queries:**
- "pop songs by voiceofruthie about apocalypse"
- "upbeat songs with composer AiCanvas"
- "rhythm and blues songs about love"

**Natural Language:**
- "find me some sad songs"
- "what are some energetic pop tracks"
- "show me songs similar to Best Thing"

### Audio Queries

- Upload or record:
  - Humming a melody
  - Singing a portion of the song
  - Playing an instrument
  - Any audio snippet (even low quality)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Author

**Mohan Ram**
- GitHub: [@mmohanram13](https://github.com/mmohanram13)
- LinkedIn: [Mohan Ram](https://www.linkedin.com/in/mmohanram13)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The MuQ model weights are licensed under CC-BY-NC 4.0 (non-commercial use).

---

## Acknowledgments

- **Elasticsearch** for powerful search infrastructure
- **Google Vertex AI** for state-of-the-art text embeddings and language models
- **Tencent AI Lab** for the OpenMuQ (MuQ-large-msd-iter) audio embedding model
- **Streamlit** for the intuitive UI framework
- **HuggingFace** for model hosting and distribution
- **Dataset Contributors** for royalty-free music samples

---

## References

- [MuQ Paper: Self-Supervised Music Representation Learning](https://arxiv.org/abs/2501.01108)
- [OpenMuQ HuggingFace Repository](https://huggingface.co/OpenMuQ)
- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Streamlit Documentation](https://docs.streamlit.io)

---

## Future Enhancements

- Support for additional audio formats (MP3, FLAC, OGG)
- User authentication and personalized playlists
- Real-time streaming audio search
- Multi-language lyrics support
- Advanced filtering by duration, tempo, key, mood
- Collaborative filtering and recommendations
- Music theory analysis (chord progressions, scale detection)
- Export search results to CSV/JSON
- Mobile-responsive design
- API endpoint for programmatic access

---

For issues, feature requests, or contributions, please visit the [GitHub repository](https://github.com/mmohanram13/singnseek)
│  - Index Management (create, delete, load)                  │
└──────────────┬──────────────────────┬──────────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐    ┌──────────────────────┐
│   Elasticsearch      │    │    Vertex AI         │
│   - Vector Search    │    │    - Text Embeddings │
│   - BM25 Search      │    │    - Re-ranking      │
│   - Index Storage    │    │    (text-embed-004)  │
└──────────────────────┘    └──────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Audio Embeddings                          │
│                MuQ-large-msd-iter Model                      │
│                (512-dimensional vectors)                     │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **utils.py** - Backend Logic
- `ElasticsearchClient`: Manages ES connections (local & cloud)
- `EmbeddingGenerator`: Generates audio (MuQ) & text (Vertex AI) embeddings
- `create_song_index()`: Creates ES index with vector mappings
- `load_demo_data()`: Indexes songs with embeddings
- `search_songs()`: Hybrid search combining BM25 + vector similarity

#### 2. **main.py** - User Interface
- **Home**: Search interface (text + audio upload + recording)
- **All Songs**: Browse complete collection
- **Manage**: Index management (create, load, delete)

#### 3. **Data Flow**

**Indexing Process:**
```
CSV Data → Load Metadata → Generate Text Embeddings (Vertex AI)
                        → Load Audio → Generate Audio Embeddings (MuQ)
                        → Combine → Index to Elasticsearch
```

**Search Process:**
```
User Query → Text Embedding (Vertex AI) + Audio Embedding (MuQ)
          → Elasticsearch Hybrid Search (BM25 + Vector Similarity)
          → Score & Rank Results → Display
```

---

## 📖 Usage

### Text Search
1. Navigate to **Home** tab
2. Enter search terms (song name, composer, lyrics)
3. Click **Search** button
4. Results show with relevance scores

### Audio Search
1. Navigate to **Home** tab
2. **Option A**: Upload audio file (.wav, max 20MB)
3. **Option B**: Click **Record Audio** to record using your microphone
4. Click **Search** button
5. Results show audio similarity scores

### Browse All Songs
1. Navigate to **All Songs** tab
2. View complete collection with metadata
3. Play audio samples directly

### Manage Index
1. Navigate to **Manage** tab
2. Create new index, load demo data, or delete index
3. View index statistics and health

---

## ⚙️ Configuration

### config.yaml

Static configuration for the application:

```yaml
elasticsearch:
  index_name: singnseek
  timeout: 30
  max_retries: 3

embeddings:
  audio_dims: 1024    # MuQ model output
  text_dims: 768     # Vertex AI embedding dimensions

search:
  hybrid_alpha: 0.6  # 60% text, 40% audio weighting
  top_k: 10          # Number of results to return

muq:
  model: OpenMuQ/MuQ-large-msd-iter
  sample_rate: 24000
```

### Environment Variables

For sensitive credentials, use environment variables:

- `ELASTICSEARCH_URL`: Elasticsearch URL (default: http://localhost:9200)
- `ELASTICSEARCH_API_KEY`: Your ElasticSearch API key
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID (for Vertex AI)
- `GOOGLE_CLOUD_REGION`: GCP region (default: us-central1)

---

## 🐳 Docker Deployment

### Using Docker Compose

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**:
   Open `http://localhost:8501`

3. **Stop**:
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build**:
   ```bash
   docker build -t singnseek:latest .
   ```

2. **Run**:
   ```bash
   docker run -p 8501:8501 --name singnseek-app singnseek:latest
   ```

3. **Stop**:
   ```bash
   docker stop singnseek-app
   docker rm singnseek-app
   ```

---

## 🧪 Testing

Run tests to verify functionality:

```bash
# Test Elasticsearch connection
python -c "from utils import ElasticsearchClient; client = ElasticsearchClient(); print('Connected:', client.ping())"

# Test audio embeddings
python muq_test.py

# Full application test
streamlit run main.py
```

---

## 📊 Project Structure

```
singnseek/
├── main.py                  # Streamlit UI application
├── utils.py                 # Backend utilities and logic
├── config.yaml              # Static configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── muq_test.py             # Audio embedding tests
├── mp3_to_wav_converter.py # Audio format converter
├── dataset/                # Song dataset
└── images/                 # UI assets
```

---

## 🛠️ Troubleshooting

### Elasticsearch Connection Issues
- Verify Elasticsearch is running: `curl http://localhost:9200`
- Check credentials in `config.yaml`
- Ensure port 9200 is not blocked

### Audio Embedding Errors
- Verify audio files are in WAV format
- Check sample rate is 24kHz
- Ensure MuQ model is downloaded (auto-downloads on first run)

### Memory Issues
- Reduce batch size in `load_demo_data()`
- Use CPU instead of GPU if VRAM is limited
- Close other applications

### Search Returns No Results
- Verify index is created and loaded
- Check index stats in Manage tab
- Try broader search terms

---

## 👤 Author

**Mohan Ram**
- GitHub: [@mmohanram13](https://github.com/mmohanram13)
- Email: mmohanram13@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Elasticsearch** for powerful search capabilities
- **Google Vertex AI** for text embeddings
- **OpenMuQ** for audio embedding model (MuQ-large-msd-iter)
- **Streamlit** for the intuitive UI framework
- **Dataset Contributors** for royalty-free music samples

---

## 🚀 Future Enhancements

- [ ] Add more audio formats support (MP3, FLAC)
- [ ] Implement user authentication
- [ ] Add playlist creation features
- [ ] Support for real-time streaming search
- [ ] Multi-language lyrics support
- [ ] Advanced filtering (genre, year, duration)
- [ ] Export search results

---

For issues, feature requests, or contributions, please visit the [GitHub repository](https://github.com/mmohanram13/singnseek)
