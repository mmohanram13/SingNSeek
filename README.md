# SingN'Seek 🎵

A powerful multimodal song search application that enables searching songs by text (lyrics, metadata) and audio (humming, recordings) using **Elasticsearch** and **Google Vertex AI**.

## 🎯 Features

- **Multimodal Search**: Search by typing lyrics, composer, or upload/record audio
- **Hybrid Search**: Combines BM25 (text) and vector similarity (embeddings) for best results
- **Audio Embeddings**: Uses MuQ-large-msd-iter model for audio fingerprinting
- **Text Embeddings**: Vertex AI for semantic text understanding
- **Real-time Search**: Instant results with relevance scoring
- **Elasticsearch Backend**: Scalable indexing and retrieval
- **Beautiful UI**: Clean, intuitive Streamlit interface

---

## 🚀 Quick Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 2GB+ free disk space
- Docker (for Elasticsearch)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mmohanram13/singnseek.git
   cd singnseek
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # OR: .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Elasticsearch**:
   
   Start Elasticsearch with Docker:
   ```bash
   curl -fsSL https://elastic.co/start-local | sh
   ```
   
   This starts:
   - Elasticsearch at http://localhost:9200
   - Kibana at http://localhost:5601
   - Default credentials: `elastic` / `changeme`

5. **Configure the application**:
   
   Update `config.yaml` with your Elasticsearch credentials:
   ```yaml
   elasticsearch:
     host: "localhost"
     port: 9200
     scheme: "http"
     username: "elastic"
     password: "changeme"
   ```

6. **Run the application**:
   ```bash
   streamlit run main.py
   ```
   
   The app will open at `http://localhost:8501`

7. **Initialize the database**:
   - Click **"Manage"** tab
   - Click **"Create Index"** button
   - Click **"Load Demo Data"** button (takes 2-5 minutes)
   - Wait for success message

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Streamlit UI (main.py)                   │
│  - Home (Search Interface)                                  │
│  - All Songs (Browse)                                       │
│  - Manage (Index Management)                                │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│                 Utility Layer (utils.py)                    │
│  - ElasticsearchClient (connection & operations)            │
│  - EmbeddingGenerator (audio & text embeddings)             │
│  - Search Functions (hybrid scoring)                        │
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
  audio_dims: 512    # MuQ model output
  text_dims: 768     # Vertex AI embedding dimensions

search:
  hybrid_alpha: 0.6  # 60% text, 40% audio weighting
  top_k: 10          # Number of results to return
  rerank_top_k: 20   # Candidates for reranking

muq:
  model: OpenMuQ/MuQ-large-msd-iter
  sample_rate: 24000
```

### Environment Variables

For sensitive credentials, use environment variables:

- `ELASTICSEARCH_HOST`: Elasticsearch host (default: localhost)
- `ELASTICSEARCH_PORT`: Elasticsearch port (default: 9200)
- `ELASTICSEARCH_API_KEY`: Your ElasticSearch API key
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID (for Vertex AI)

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
│   ├── copyright/
│   └── royalty_free/
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
