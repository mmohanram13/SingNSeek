# ✅ SingN'Seek Setup Checklist

Use this checklist to ensure everything is configured correctly.

## 📋 Pre-Installation

- [ ] Python 3.9+ installed (`python3 --version`)
- [ ] pip installed (`pip --version`)
- [ ] Git installed (if cloning from GitHub)
- [ ] 2GB+ free disk space
- [ ] Stable internet connection (for model downloads)

## 📦 Installation

- [ ] Repository cloned/downloaded
- [ ] Virtual environment created (`.venv/`)
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] No installation errors in terminal

## 🔧 Elasticsearch Setup

Choose one:

### Option A: Docker (Recommended for Testing)
- [ ] Docker installed and running
- [ ] Elasticsearch container started
- [ ] Can access http://localhost:9200

### Option B: Local Installation
- [ ] Elasticsearch 8.x installed
- [ ] Elasticsearch service running
- [ ] Can access http://localhost:9200

### Option C: Elastic Cloud
- [ ] Account created at cloud.elastic.co
- [ ] Deployment created
- [ ] Cloud ID copied
- [ ] API Key created
- [ ] Can access deployment URL

## ⚙️ Configuration

- [ ] `config/` directory exists
- [ ] `config/elastic_config.yaml` created and configured
  - [ ] Connection details filled in
  - [ ] Authentication configured
  - [ ] Settings adjusted if needed

### Optional: Vertex AI
- [ ] Google Cloud project created
- [ ] Vertex AI API enabled
- [ ] Service account created
- [ ] JSON key file downloaded
- [ ] `.env` file created
- [ ] Environment variables set in `.env`

## ✅ Verification

- [ ] Run `python check_env.py` - all checks pass
- [ ] No red ❌ errors in output
- [ ] Elasticsearch connection successful

## 🎵 Data Setup

- [ ] Dataset files exist in `dataset/copyright/`
- [ ] `dataset_meta.csv` exists and has data
- [ ] Audio files (.wav) present

## 🚀 First Run

- [ ] `streamlit run main.py` starts without errors
- [ ] App opens in browser (http://localhost:8501)
- [ ] No error messages on home page
- [ ] Can see "Home", "All Songs", "Manage" tabs

## 🗄️ Database Initialization

In the "Manage" tab:
- [ ] Click "Create Index" - Success message appears
- [ ] Index status shows "Active"
- [ ] Click "Load Demo Data" - Loading completes
- [ ] Document count > 0

## 🔍 Testing Search

In the "Home" tab:
- [ ] Text search works (try: "love")
  - [ ] Results appear
  - [ ] Audio plays
  - [ ] Metadata displays
  
- [ ] Audio upload works
  - [ ] Can select .wav file
  - [ ] File uploads successfully
  - [ ] Can search with uploaded file

- [ ] Audio recording works (optional)
  - [ ] Microphone permission granted
  - [ ] Can start/stop recording
  - [ ] Recording captured

## 📚 Browse Functionality

In the "All Songs" tab:
- [ ] Songs list loads
- [ ] Can see song metadata
- [ ] Audio players work
- [ ] All songs display correctly

## 🧪 Optional: Run Tests

- [ ] `python test_utils.py` runs
- [ ] All 5 tests pass
- [ ] No errors in output

## 🎯 Production Checklist (Optional)

- [ ] All demo data removed/replaced with real data
- [ ] API keys secured (not in git)
- [ ] `.gitignore` configured
- [ ] Error logging configured
- [ ] Backup strategy in place
- [ ] Monitoring set up

## 🐛 Troubleshooting Steps

If anything fails:

1. **Check Elasticsearch**
   ```bash
   curl http://localhost:9200
   # Should return cluster info
   ```

2. **Check Python packages**
   ```bash
   python check_env.py
   # Review output for missing packages
   ```

3. **Check logs**
   - Look for error messages in terminal
   - Check Streamlit console output

4. **Verify configuration**
   ```bash
   cat config/elastic_config.yaml
   # Ensure connection details are correct
   ```

5. **Test basic connectivity**
   ```python
   python -c "from elasticsearch import Elasticsearch; es = Elasticsearch('http://localhost:9200'); print(es.ping())"
   # Should print: True
   ```

## 📞 Getting Help

If you're stuck:

1. ✅ Reviewed relevant section in README_NEW.md
2. ✅ Checked QUICKSTART.md for quick solutions
3. ✅ Ran `python check_env.py` for diagnostics
4. ✅ Checked error messages in terminal
5. ✅ Verified Elasticsearch is running

Still stuck? Check:
- GitHub Issues
- Stack Overflow (tag: elasticsearch, streamlit)
- Elastic Community Forums

## 🎉 Success!

When all checks pass:
- ✅ Application runs smoothly
- ✅ Search returns results
- ✅ Audio playback works
- ✅ All songs display correctly

**You're ready to use SingN'Seek!** 🎵✨

---

## 📊 Quick Status Check

Run this to check your setup status:

```bash
python check_env.py
```

Expected output when ready:
```
✅ Python
✅ Dependencies
✅ Config
✅ Dataset
✅ Elasticsearch
⚠️ Vertex AI (optional)

Status: 5/6 checks passed
🎉 All checks passed! You're ready to use SingN'Seek.
```

---

**Tip**: Keep this checklist handy and check items off as you complete them!
