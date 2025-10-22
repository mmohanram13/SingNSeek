# SingN'Seek - Multimodal Song Search System

A powerful multimodal song search application that enables searching songs by text (lyrics, metadata) and audio (humming, recordings) using **Elasticsearch** and **Google Vertex AI**.

![SingN'Seek](images/logo.png)

## 🎯 Features

- **Multimodal Search**: Search by typing lyrics, composer, or upload/record audio
- **Hybrid Search**: Combines BM25 (text) and vector similarity (embeddings) for best results
- **Audio Embeddings**: Uses MuQ-large-msd-iter model for audio fingerprinting
- **Text Embeddings**: Vertex AI for semantic text understanding
- **Real-time Search**: Instant results with relevance scoring
- **Elasticsearch Backend**: Scalable indexing and retrieval
- **Beautiful UI**: Clean, intuitive Streamlit interface

## 🏗️ Architecture

```
Streamlit UI (main.py)
    ↓
Utils Layer (utils.py)
    ↓
┌─────────────────────┬─────────────────────┐
│   Elasticsearch     │   Vertex AI         │
│   (Index & Search)  │   (Embeddings)      │
└─────────────────────┴─────────────────────┘
    ↓                       ↓
Audio Embeddings        Text Embeddings
(MuQ Model)             (Gemini/Embedding)
```

## 📋 Prerequisites

1. **Python 3.9+**
2. **Elasticsearch 8.x** (local or Elastic Cloud)
3. **Google Cloud Account** with Vertex AI enabled (for text embeddings)
4. **MuQ Model** (automatically downloaded)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mmohanram13/singnseek.git
cd singnseek
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: On Apple Silicon Macs, PyTorch will automatically use MPS (Metal Performance Shaders) for GPU acceleration.

### 4. Configure Elasticsearch

#### Option A: Local Elasticsearch

1. Install Elasticsearch 8.x:
   ```bash
   # macOS (using Homebrew)
   brew tap elastic/tap
   brew install elastic/tap/elasticsearch-full
   ```

2. Start Elasticsearch:
   ```bash
   elasticsearch
   ```

3. Get the enrollment token or API key:
   ```bash
   # In another terminal
   elasticsearch-service-tokens create elastic/my-token
   ```

4. Update `config/elastic_config.yaml`:
   ```yaml
   elasticsearch:
     host: "localhost"
     port: 9200
     scheme: "http"
     api_key: "your-api-key-here"
   ```

#### Option B: Elastic Cloud

1. Sign up at [cloud.elastic.co](https://cloud.elastic.co)
2. Create a deployment
3. Get your Cloud ID and API Key
4. Update `config/elastic_config.yaml`:
   ```yaml
   elasticsearch:
     cloud_id: "your-cloud-id"
     api_key: "your-api-key"
   ```

### 5. Configure Vertex AI

1. Create a Google Cloud Project at [console.cloud.google.com](https://console.cloud.google.com)

2. Enable Vertex AI API:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

3. Create a service account and download the JSON key:
   ```bash
   gcloud iam service-accounts create singnseek-sa
   gcloud iam service-accounts keys create service-account-key.json \
       --iam-account=singnseek-sa@your-project-id.iam.gserviceaccount.com
   ```

4. Create `.env` file (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

5. Update `.env`:
   ```env
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_REGION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

## 🎵 Usage

### 1. Start the Application

```bash
streamlit run main.py
```

### 2. Initialize Elasticsearch Index

1. Navigate to **Manage** tab in the UI
2. Click **"Create Index"** to create the Elasticsearch index
3. Click **"Load Demo Data"** to index songs with embeddings (this may take several minutes)

### 3. Search for Songs

#### Text Search
- Go to **Home** tab
- Type lyrics, song name, composer, or any text
- Click **"Find My Song!"**

#### Audio Search
- Go to **Home** tab
- Upload a `.wav` file OR record audio
- Click **"Find My Song!"**

#### Hybrid Search
- Combine text and audio for best results
- The system will merge relevance scores

### 4. Browse All Songs

- Navigate to **All Songs** tab
- View complete indexed collection with metadata

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python test_utils.py
```

This will test:
- ✅ Elasticsearch connection
- ✅ Index creation/deletion
- ✅ Embedding generation (text & audio)
- ✅ Search functionality
- ✅ Data retrieval

## 📁 Project Structure

```
singnseek/
├── main.py                    # Streamlit UI
├── utils.py                   # Core functionality (Elasticsearch, embeddings)
├── muq_test.py               # Audio embedding reference
├── test_utils.py             # Test suite
├── requirements.txt          # Dependencies
├── config/
│   └── elastic_config.yaml   # Elasticsearch configuration
├── .env                      # Environment variables (create from .env.example)
├── dataset/
│   └── copyright/
│       ├── *.wav            # Audio files
│       └── dataset_meta.csv # Metadata
└── images/
    └── logo.png
```

## ⚙️ Configuration

### Elasticsearch Settings (`config/elastic_config.yaml`)

```yaml
search:
  hybrid_alpha: 0.6     # BM25 vs Vector weight (0.6 = 60% text, 40% vector)
  top_k: 10            # Number of results
  rerank_top_k: 20     # Results before re-ranking

embeddings:
  audio_dims: 512      # MuQ output dimension
  text_dims: 768       # Vertex AI embedding dimension
```

### Environment Variables (`.env`)

```env
# Vertex AI
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
VERTEX_TEXT_EMBEDDING_MODEL=text-embedding-004

# MuQ Audio Model
MUQ_MODEL=OpenMuQ/MuQ-large-msd-iter
MUQ_SAMPLE_RATE=24000
```

## 🔧 Troubleshooting

### Elasticsearch Connection Issues

```bash
# Test connection
curl -X GET "localhost:9200" -u elastic:password
```

### MuQ Model Not Loading

```bash
# Ensure transformers and torch are installed
pip install --upgrade torch transformers
```

### Vertex AI Authentication

```bash
# Test authentication
gcloud auth application-default login
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 🎨 Customization

### Add More Songs

1. Place `.wav` files in `dataset/copyright/`
2. Update `dataset_meta.csv` with metadata
3. Go to **Manage** tab
4. Click **"Load Demo Data"**

### Adjust Search Weights

Edit `config/elastic_config.yaml`:
```yaml
search:
  hybrid_alpha: 0.7  # More weight to text search
```

### Change Embedding Models

Edit `.env`:
```env
VERTEX_TEXT_EMBEDDING_MODEL=textembedding-gecko@003
MUQ_MODEL=OpenMuQ/MuQ-base
```

## 📊 Performance

- **Index Creation**: ~1 minute for 100 songs
- **Embedding Generation**: ~2-5 seconds per song
- **Search Latency**: <500ms for typical queries
- **Storage**: ~10MB per 100 songs (including vectors)

## 🛣️ Roadmap

- [ ] Vertex AI re-ranking with Gemini
- [ ] Sliding window for long audio files
- [ ] Support for more audio formats (MP3, FLAC)
- [ ] Real-time recording with live feedback
- [ ] User authentication and playlists
- [ ] Batch upload interface
- [ ] Mobile-responsive UI

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Mohan Ram M**
- GitHub: [@mmohanram13](https://github.com/mmohanram13)
- LinkedIn: [mohan-ram-m](https://www.linkedin.com/in/mohan-ram-m/)

## 🙏 Acknowledgments

- **OpenMuQ** for the MuQ audio embedding model
- **Google Vertex AI** for text embeddings
- **Elasticsearch** for powerful search capabilities
- **Streamlit** for the beautiful UI framework

---

Made with ❤️ for forgetful music lovers
