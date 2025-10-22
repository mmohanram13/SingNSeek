# SingN'Seek - System Flow Diagrams

## 🔄 Complete System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                             │
│                         (Streamlit UI)                              │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   Home   │  │All Songs │  │  Manage  │  │ Settings │          │
│  │  Search  │  │  Browse  │  │  Admin   │  │  Config  │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
└────────────────────────┬───────────────────────────────────────────┘
                         │
                         │ main.py
                         ▼
┌────────────────────────────────────────────────────────────────────┐
│                        UTILITY LAYER                                │
│                         (utils.py)                                  │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐│
│  │ Elasticsearch    │  │  Embedding       │  │     Search       ││
│  │    Client        │  │  Generator       │  │    Engine        ││
│  │                  │  │                  │  │                  ││
│  │ • connect()      │  │ • text_embed()   │  │ • hybrid_search()││
│  │ • create_index() │  │ • audio_embed()  │  │ • rank_results() ││
│  │ • bulk_index()   │  │ • combine()      │  │ • score()        ││
│  └──────────────────┘  └──────────────────┘  └──────────────────┘│
└────────┬────────────────────────┬────────────────────┬─────────────┘
         │                        │                    │
         ▼                        ▼                    ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Elasticsearch  │    │   Vertex AI      │    │   MuQ Model      │
│                 │    │                  │    │                  │
│  • Vector Store │    │  • text-embed-   │    │  • Audio         │
│  • BM25 Search  │    │    004           │    │    Fingerprint   │
│  • Indexing     │    │  • Gemini        │    │  • 512-dim       │
│                 │    │    Re-ranking    │    │    vectors       │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

## 📊 Data Indexing Flow

```
START: Load Demo Data
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Read dataset_meta.csv               │
│     • Parse song metadata               │
│     • Validate entries                  │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  2. For Each Song:                      │
│                                          │
│  ┌──────────────────────────────────┐  │
│  │ a. Combine Metadata + Lyrics     │  │
│  │    "Song: X | Composer: Y | ..." │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │ b. Generate Text Embedding       │  │
│  │    Vertex AI → 768-dim vector    │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │ c. Load Audio File (.wav)        │  │
│  │    librosa @ 24kHz               │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │ d. Generate Audio Embedding      │  │
│  │    MuQ Model → 512-dim vector    │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐  │
│  │ e. Create Document               │  │
│  │    {                             │  │
│  │      metadata,                   │  │
│  │      lyrics,                     │  │
│  │      lyrics_vector: [768],       │  │
│  │      audio_vector: [512]         │  │
│  │    }                             │  │
│  └──────────────┬───────────────────┘  │
└─────────────────┼───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  3. Bulk Index to Elasticsearch         │
│     • Batch upload documents            │
│     • Create vector indices             │
│     • Verify indexing                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
              SUCCESS
     (Songs indexed and searchable)
```

## 🔍 Search Flow

```
USER INPUT
    │
    ├─── Text Query: "romantic love"
    │
    └─── Audio Input: humming.wav
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Process Inputs                      │
│                                          │
│  Text Branch:                           │
│  ┌──────────────────────────────────┐  │
│  │ • Parse query text               │  │
│  │ • Generate text embedding        │  │
│  │   (Vertex AI)                    │  │
│  └──────────────────────────────────┘  │
│                                          │
│  Audio Branch:                          │
│  ┌──────────────────────────────────┐  │
│  │ • Load audio file                │  │
│  │ • Generate audio embedding       │  │
│  │   (MuQ Model)                    │  │
│  └──────────────────────────────────┘  │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  2. Query Elasticsearch                 │
│                                          │
│  BM25 Search (Text):                    │
│  ┌──────────────────────────────────┐  │
│  │ query: "romantic love"           │  │
│  │ fields: [song_name, lyrics,      │  │
│  │         composer, genre]         │  │
│  │ boost: song_name^3, lyrics^2     │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 │ score_bm25            │
│                 ▼                       │
│  Vector Search (Text Embedding):        │
│  ┌──────────────────────────────────┐  │
│  │ query_vector: [768-dim]          │  │
│  │ field: lyrics_vector             │  │
│  │ similarity: cosine               │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 │ score_text_vec        │
│                 ▼                       │
│  Vector Search (Audio Embedding):       │
│  ┌──────────────────────────────────┐  │
│  │ query_vector: [512-dim]          │  │
│  │ field: audio_vector              │  │
│  │ similarity: cosine               │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│                 │ score_audio_vec       │
└─────────────────┼───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  3. Hybrid Scoring                      │
│                                          │
│  For each result:                       │
│  ┌──────────────────────────────────┐  │
│  │ final_score = alpha × score_bm25 │  │
│  │             + (1-alpha) × (       │  │
│  │               score_text_vec +    │  │
│  │               score_audio_vec     │  │
│  │             )                     │  │
│  │                                   │  │
│  │ where alpha = 0.6 (configurable) │  │
│  └──────────────┬───────────────────┘  │
└─────────────────┼───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  4. Rank & Return Results               │
│                                          │
│  • Sort by final_score (descending)     │
│  • Take top_k results (default: 10)     │
│  • [Optional] Re-rank with Vertex AI    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
            DISPLAY RESULTS
     (Sorted by relevance with scores)
```

## 🎭 Component Interaction

```
┌─────────────┐
│   Browser   │
│  (User UI)  │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│  Streamlit  │
│   Server    │
└──────┬──────┘
       │ Function Calls
       ▼
┌─────────────┐       ┌──────────────┐
│   utils.py  │◄─────►│config/*.yaml │
│             │       └──────────────┘
└──────┬──────┘
       │
       ├──────────────────┬──────────────────┬───────────────┐
       │                  │                  │               │
       ▼                  ▼                  ▼               ▼
┌─────────────┐    ┌─────────────┐    ┌─────────┐   ┌──────────┐
│Elasticsearch│    │  Vertex AI  │    │   MuQ   │   │  .env    │
│   Client    │    │     API     │    │  Model  │   │  vars    │
└─────────────┘    └─────────────┘    └─────────┘   └──────────┘
```

## 📈 State Management (Streamlit)

```
Session State Variables:
├── search_state: 'idle' | 'searching' | 'searched' | 'no_results'
├── search_results: List[Dict] (search results)
├── search_query: str (last query text)
├── audio_file: UploadedFile | None
├── audio_bytes: bytes | None
├── scroll_to_results: bool
└── current_page: 'home' | 'all_songs' | 'add_song'

State Transitions:
    idle ──[user clicks search]──> searching
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                    [has results]              [no results]
                         │                           │
                         ▼                           ▼
                    searched                   no_results
```

## 🔐 Authentication Flow

```
Elasticsearch:
    ┌────────────────┐
    │  Config File   │
    │   (YAML)       │
    └───────┬────────┘
            │
            ├─── cloud_id? ──> Elastic Cloud
            │                  + API Key
            │
            └─── host/port ──> Local/Custom
                               + API Key OR
                               + Basic Auth (user/pass)

Vertex AI:
    ┌────────────────┐
    │   .env File    │
    └───────┬────────┘
            │
            ├─── GOOGLE_CLOUD_PROJECT
            ├─── GOOGLE_CLOUD_REGION
            └─── GOOGLE_APPLICATION_CREDENTIALS
                        │
                        ▼
                ┌───────────────────┐
                │ Service Account   │
                │   JSON Key        │
                └───────────────────┘
```

## 📊 Data Model

```
Song Document in Elasticsearch:
{
  "_id": "1",
  "_source": {
    "song_index": 1,
    "song_name": "Kadhal Rojave",
    "song_file_path_name": "Kadhal Rojave.wav",
    "composer": "A.R. Rahman",
    "album": "Roja",
    "released_year": 1992,
    "genre": "Romantic",
    "lyricist": "Vairamuthu",
    "singers": "S.P. Balasubrahmanyam",
    "lyrics": "Kadhal rojave vaa...",
    "lyrics_vector": [0.123, 0.456, ..., 0.789],  // 768 dims
    "audio_vector": [0.321, 0.654, ..., 0.987]    // 512 dims
  }
}
```

## 🎯 Performance Optimization

```
Bottlenecks & Solutions:

1. Embedding Generation (Slow)
   Solution: Batch processing, caching
   
2. Large Audio Files
   Solution: Sliding window approach
   
3. Network Latency (Vertex AI)
   Solution: Local caching, batch requests
   
4. Elasticsearch Queries
   Solution: Index optimization, query tuning

Current Performance:
├── Index: 1 min / 100 songs
├── Text Embed: 1-2 sec / song
├── Audio Embed: 2-5 sec / song
├── Search: <500ms
└── Index Size: ~100KB / song
```

---

**Note**: All diagrams are simplified representations. Actual implementation includes error handling, logging, retries, and additional features not shown here for clarity.
