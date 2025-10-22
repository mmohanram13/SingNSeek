# 🚀 Quick Start Guide - SingN'Seek

Get up and running with SingN'Seek in 5 minutes!

## ⚡ Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.9 or higher
- ✅ pip (Python package manager)
- ✅ 2GB+ free disk space

## 📦 Installation

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/mmohanram13/singnseek.git
cd singnseek

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# OR: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Elasticsearch

#### Option A: Quick Start with Docker (Recommended)

```bash
# Pull and run Elasticsearch
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

#### Option B: Use Elastic Cloud (Free Trial)

1. Sign up at https://cloud.elastic.co
2. Create a deployment (free tier available)
3. Copy your Cloud ID and create an API Key

### 3. Configure the Application

Run the setup wizard:

```bash
python setup.py
```

**OR** manually configure:

```bash
# Copy configuration templates
cp .env.example .env

# Edit config/elastic_config.yaml
# For local Docker Elasticsearch:
#   host: localhost
#   port: 9200
#   scheme: http
#   (no authentication needed with security disabled)
```

### 4. Run the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## 🎵 First Steps in the App

### Initialize the Database

1. Click **"Manage"** tab
2. Click **"Create Index"** button
3. Click **"Load Demo Data"** button (takes 2-5 minutes)
4. Wait for success message

### Search for Songs

1. Go to **"Home"** tab
2. Try these example searches:
   - Type: `"romantic love"`
   - Type: `"Rahman"`
   - Upload an audio file
   - Record yourself humming

### Browse All Songs

1. Click **"All Songs"** tab
2. Browse the complete catalog

## 🎯 Without Vertex AI (Quick Demo)

You can use SingN'Seek without Vertex AI for audio-only search:

1. Skip the Vertex AI setup in `setup.py`
2. The app will work with:
   - ✅ Audio search (using MuQ embeddings)
   - ✅ Basic text search (BM25)
   - ❌ Semantic text embeddings (requires Vertex AI)

## 🔍 Verify Everything Works

Run the test suite:

```bash
python test_utils.py
```

Expected output:
```
✅ Elasticsearch Connection .... PASSED
✅ Index Creation ............... PASSED
✅ Embedding Generation ......... PASSED
✅ Index Operations ............. PASSED
✅ Search Functionality ......... PASSED
```

## 🐛 Quick Troubleshooting

### "Connection refused" error

```bash
# Check if Elasticsearch is running
curl http://localhost:9200

# If not, start it:
docker start elasticsearch
```

### "Module not found" error

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "MuQ model not loading"

```bash
# MuQ will auto-download on first use
# Ensure you have stable internet and 1GB+ free space
```

### Search returns no results

1. Ensure index is created: Check **Manage** tab
2. Ensure data is loaded: Look for document count
3. Try a broader search term

## 🎓 Next Steps

- **Add More Songs**: See "Add More Songs" section in README_NEW.md
- **Configure Vertex AI**: For semantic text search
- **Customize Search**: Adjust `hybrid_alpha` in config
- **Explore API**: Check `utils.py` for programmatic access

## 💡 Example Searches

Try these to test functionality:

| Search Type | Example | Expected Result |
|-------------|---------|----------------|
| Song Name | `Kadhal Rojave` | Finds the song |
| Composer | `Rahman` | All A.R. Rahman songs |
| Lyrics | `love romantic` | Songs with romantic lyrics |
| Genre | `folk` | Folk songs |
| Audio | Upload/record | Similar sounding songs |

## 📞 Need Help?

- 📖 Full documentation: `README_NEW.md`
- 🧪 Run tests: `python test_utils.py`
- 🐛 Check logs: Look for error messages in terminal
- 💬 GitHub Issues: https://github.com/mmohanram13/singnseek/issues

---

**Time to completion**: ~5 minutes (excluding data loading)  
**Difficulty**: ⭐⭐ (Easy-Medium)

Happy searching! 🎵✨
