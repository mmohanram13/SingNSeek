# SingN'Seek

**Can't remember the song? Just hum, type, or guess — we'll find it!**

📺 **[Watch Demo on YouTube](https://www.youtube.com/watch?v=JCv2n1I46uA)**

A multimodal song search application that finds music through text (lyrics, metadata, natural language) and audio (humming, recordings) using Elasticsearch, Google Vertex AI, and OpenMuQ embeddings.

---

## ✨ Key Features

**Multimodal Search**
- Text search: song name, lyrics, composer, artist, genre, album
- Natural language queries: "pop songs by voiceofruthie about apocalypse"
- Audio search: upload files or record humming/singing
- Hybrid search: combine text AND audio queries for maximum precision (e.g., "pop songs" + humming sample)
- Hybrid scoring: combines BM25 keyword matching with vector similarity

**Intelligent Processing**
- Query Vetter powered by Gemini 2.5 Flash Lite parses natural language into structured queries
- Automatic field detection and filtering (composer, artist, genre, album)
- Lyrics-aware search with semantic matching and higher relevance scoring
- Fuzzy matching for typos and variations

**Technical Stack**
- Elasticsearch for production-ready search infrastructure
- Google Vertex AI text-embedding-005 (768-dim vectors)
- OpenMuQ MuQ-large-msd-iter (1024-dim audio vectors)
- Streamlit UI

---

## 🔍 How It's Different from Shazam

**Shazam: Audio Fingerprinting**
- Creates unique "fingerprints" from spectrogram peaks
- Fast and accurate for exact recording matches
- ❌ Requires exact recording in database
- ❌ Can't match covers, remixes, or variations
- ❌ Fails with humming/singing
- ❌ Audio-only queries

**SingN'Seek: Semantic + Multimodal**
- Semantic embeddings capture musical features
- ✅ Matches similar songs, covers, and variations
- ✅ Works with humming, singing, partial audio
- ✅ Supports text and natural language queries
- ✅ Understands relationships: _"songs about love"_
- ✅ Ideal for discovery and exploration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit UI (main.py)                 │
│  Home | All Songs | Manage                          │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│           Core Logic (utils.py)                     │
│  ElasticsearchClient | EmbeddingGenerator           │
│  Query Vetter | Search | Index Management           │
└────────┬────────────────────┬───────────────────────┘
         │                    │
         ▼                    ▼
┌──────────────────┐   ┌────────────────────────────┐
│  Elasticsearch   │   │    Google Vertex AI        │
│  - Vector KNN    │   │  - Text Embeddings         │
│  - BM25 Search   │   │  - Gemini Query Parser     │
└──────────────────┘   └────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│         OpenMuQ Audio Embeddings                    │
│   MuQ-large-msd-iter (1024-dim vectors)             │
└─────────────────────────────────────────────────────┘
```

**Indexing**: CSV → Parse Metadata → Generate Embeddings (Text + Audio) → Index to Elasticsearch

**Search**: User Query → Parse/Embed → Hybrid Search (BM25 + Vector KNN) → Ranked Results

---

## 🤖 Models Used

### OpenMuQ (MuQ-large-msd-iter)
- **Purpose**: Audio embedding generation for semantic music understanding
- **Architecture**: ~300M parameters, self-supervised learning on Million Song Dataset
- **Output**: 1024-dimensional vectors from 24 kHz audio
- **Advantages**: Captures melody, rhythm, timbre, mood; works with humming/covers; SOTA on MARBLE Benchmark
- **Paper**: [arxiv.org/abs/2501.01108](https://arxiv.org/abs/2501.01108) | **HuggingFace**: [OpenMuQ/MuQ-large-msd-iter](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter)

### Vertex AI text-embedding-005
- **Purpose**: Text embedding for semantic text understanding
- **Output**: 768-dimensional text embeddings
- **Advantages**: High-quality semantic representations, multi-language support, production-ready

### Gemini 2.5 Flash Lite
- **Purpose**: Natural language query parsing
- **Task**: Parse conversational queries into structured Elasticsearch filters
- **Latency**: ~200-500ms | **Temperature**: 0.1 (deterministic)

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.13+, pip package manager
- Elasticsearch (local or cloud)
- Google Cloud account with Vertex AI enabled

### Quick Start

**1. Clone and Setup**
```bash
git clone https://github.com/mmohanram13/singnseek.git
cd singnseek
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Setup Elasticsearch**

For local Elasticsearch:
```bash
curl -fsSL https://elastic.co/start-local | sh
```
This will provide you with an API key. Runs at `http://localhost:9200`

Alternatively, use [Elastic Cloud](https://cloud.elastic.co/) and get your Cloud URL and API key.

**3. Authenticate with Google Cloud**
```bash
gcloud auth login
gcloud config set project your-project-id
```

**4. Configure Application**

Create `.env` file:
```bash
# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_API_KEY=your-api-key

# Google Cloud / Vertex AI
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_REGION=asia-south1
VERTEX_AI_REGION=us-central1
```

Edit `src/config/config.yaml` for additional settings (optional).

**5. Run Application**
```bash
streamlit run src/main.py
```
Opens at `http://localhost:8501`

**6. Initialize Database**
- Go to **Manage** tab
- Click **Create Index**
- Click **Load Demo Data** (takes 2-5 minutes). For the first time, the MuQ model gets downloaded in the backend, which might take additional time to complete.

---

## ⚙️ Configuration

Key settings in `src/config/config.yaml`:

```yaml
elasticsearch:
  index_name: singnseek
  timeout: 30

embeddings:
  audio_dims: 1024    # MuQ model
  text_dims: 768      # Vertex AI

search:
  hybrid_alpha: 0.6   # 60% text, 40% audio weighting
  top_k: 5

muq:
  model: OpenMuQ/MuQ-large-msd-iter
  sample_rate: 24000

vertex_ai:
  text_embedding_model: text-embedding-005
  gemini_model: gemini-2.5-flash-lite
```

---

## 📖 Usage

**Text Search**
- Enter queries: song name, lyrics, artist, genre, natural language
- Examples: `"Best Thing"`, `"lyrics about love"`, `"pop songs by voiceofruthie"`

**Audio Search**
- Upload `.wav` file (max 20MB, 24 kHz recommended)
- OR click **Record Audio** to hum/sing using your microphone

**Hybrid Search (Text + Audio)**
- Combine text and audio queries for more precise results
- Enter a text query AND upload/record audio simultaneously
- The system combines both signals with configurable weighting (default: 60% text, 40% audio)
- **Use Cases**:
  - Find a specific song when you remember partial lyrics and can hum the tune
  - Narrow down results by genre/artist while providing audio sample
  - Search for covers or versions by combining text filters with audio similarity
- **Example**: Type `"pop songs"` and upload a humming sample to find pop songs similar to your tune

**Browse Songs**
- Navigate to **All Songs** tab to view complete collection

**Manage Index**
- **Manage** tab: create, load, or delete index

---

## 🧠 Query Vetter (Natural Language Processing)

Powered by Gemini 2.5 Flash Lite, the Query Vetter parses conversational queries into structured Elasticsearch filters.

**Features**
- **Field-Specific Filtering**: Extracts composer, artist, genre, album filters automatically
- **Lyrics-Aware Search**: Detects lyrics queries and applies semantic hybrid search
- **Combined Filtering**: Handles multi-constraint queries like _"pop songs by voiceofruthie about apocalypse"_

**Examples**

| Query | Parsed Result |
|-------|---------------|
| `"song with composer AiCanvas"` | Filters: `{composer: "AiCanvas"}` |
| `"lyrics about love"` | Search: `"love"`, Type: `lyrics`, Hybrid: `true` |
| `"pop songs by voiceofruthie about apocalypse"` | Filters: `{genre: "Pop", singers: "voiceofruthie"}`, Search: `"apocalypse"` |

Implementation: `src/utils/query_vetter.py`

---

## 📝 Query Samples

**Text Queries**
- Basic: `"Best Thing"`, `"apocalypse"`, `"hollow"`
- Field-specific: `"song with composer AiCanvas"`, `"songs by voiceofruthie"`, `"pop songs"`
- Lyrics: `"lyrics about love"`, `"songs with apocalypse in the lyrics"`
- Combined: `"pop songs by voiceofruthie about apocalypse"`
- Natural language: `"find me some sad songs"`, `"energetic pop tracks"`

**Audio Queries**
- Upload or record: humming, singing, instrument playing, any audio snippet

**Hybrid Queries (Text + Audio)**
- Genre + Audio: Type `"pop songs"` + upload humming → finds pop songs similar to the melody
- Artist + Audio: Type `"songs by voiceofruthie"` + record singing → finds artist's songs matching the tune
- Lyrics + Audio: Type `"lyrics about love"` + upload audio → finds love songs with similar musical features
- Filters + Audio: Type `"song with composer AiCanvas"` + record humming → finds composer's songs matching the melody
- Natural language + Audio: Type `"energetic tracks"` + upload sample → finds high-energy songs with similar sound

**Why Use Hybrid Search?**
- **More Precision**: Combines semantic understanding of text with musical similarity from audio
- **Better Disambiguation**: When multiple songs match text, audio narrows down to the right one
- **Flexible Discovery**: Find songs that match both conceptual criteria (genre, mood) and sonic characteristics

---

## 🙏 Acknowledgments

- **Elasticsearch** - Search infrastructure
- **Google Vertex AI** - Text embeddings and language models
- **Tencent AI Lab** - OpenMuQ audio embedding model
- **Streamlit** - UI framework
- **HuggingFace** - Model hosting

---

## 🔗 References

- [MuQ Paper](https://arxiv.org/abs/2501.01108) | [OpenMuQ HuggingFace](https://huggingface.co/OpenMuQ)
- [Elasticsearch Docs](https://www.elastic.co/guide/index.html) | [Vertex AI Docs](https://cloud.google.com/vertex-ai/docs)
