"""
Utility functions for SingN'Seek multimodal song search.
Handles Elasticsearch operations, embeddings generation, and hybrid search.
"""

import os
import yaml
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import gc

import pandas as pd
import numpy as np
import torch
import librosa
import soundfile as sf
from elasticsearch import Elasticsearch, helpers
from google.cloud import aiplatform
from google.oauth2 import service_account
from dotenv import load_dotenv

# Import MuQ model for audio embeddings
try:
    from muq import MuQ
except ImportError:
    MuQ = None

# Configure logging
from .logging_config import setup_logging, get_logger

# Initialize logging system (call once at module import)
setup_logging()
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Load static configuration from YAML
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Helper function to get boolean from env
def get_bool_env(key: str, default: bool = False) -> bool:
    """Parse boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


class ElasticsearchClient:
    """Manages Elasticsearch connection and operations."""
    
    def __init__(self):
        self.client = None
        self.index_name = CONFIG['elasticsearch']['index_name']
        self._connect()
    
    def _connect(self):
        """Establish connection to Elasticsearch."""
        try:
            es_config = CONFIG['elasticsearch']
            timeout = es_config['timeout']
            max_retries = es_config['max_retries']
            retry_on_timeout = es_config['retry_on_timeout']
            
            # API Key is required for authentication
            api_key = os.getenv('ELASTICSEARCH_API_KEY')
            if not api_key:
                raise ValueError("ELASTICSEARCH_API_KEY is required in .env file")
            
            # Elasticsearch URL (works for local, self-hosted, and cloud)
            url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
            
            self.client = Elasticsearch(
                [url],
                api_key=api_key,
                request_timeout=timeout,
                max_retries=max_retries,
                retry_on_timeout=retry_on_timeout
            )
            
            # Test connection
            if self.client.ping():
                logger.info("Successfully connected to Elasticsearch")
            else:
                raise ConnectionError("Could not connect to Elasticsearch")
                
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
    
    def get_client(self) -> Elasticsearch:
        """Return the Elasticsearch client."""
        return self.client


class EmbeddingGenerator:
    """Generates embeddings for audio and text using MuQ and Vertex AI."""
    
    def __init__(self):
        self.audio_model = None
        self.device = self._setup_device()
        self.sample_rate = CONFIG['muq']['sample_rate']
        self._setup_audio_model()
        self._setup_vertex_ai()
    
    def _setup_device(self) -> str:
        """Auto-detect best available device (MPS/CUDA/CPU) with fallback."""
        try:
            # Try MPS (Apple Silicon) first
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logger.info("Using MPS (Apple Silicon GPU) for embeddings")
                return 'mps'
        except Exception as e:
            logger.warning(f"MPS check failed: {e}, trying CUDA")
        
        try:
            # Try CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                logger.info(f"Using CUDA (GPU: {torch.cuda.get_device_name(0)}) for embeddings")
                return 'cuda'
        except Exception as e:
            logger.warning(f"CUDA check failed: {e}, falling back to CPU")
        
        # Fallback to CPU
        logger.info("Using CPU for embeddings")
        return 'cpu'
    
    def _setup_audio_model(self):
        """Initialize MuQ model for audio embeddings."""
        if MuQ is None:
            logger.warning("MuQ not available. Audio embeddings disabled.")
            return
        
        try:
            model_name = CONFIG['muq']['model']
            logger.info(f"Loading MuQ model: {model_name} on {self.device}")
            self.audio_model = MuQ.from_pretrained(model_name)
            
            # Try to move model to device with fallback
            try:
                self.audio_model = self.audio_model.to(self.device).eval()
                logger.info(f"MuQ model loaded successfully on {self.device}")
            except Exception as device_error:
                logger.warning(f"Failed to load model on {self.device}: {device_error}")
                logger.info("Falling back to CPU")
                self.device = 'cpu'
                self.audio_model = self.audio_model.to('cpu').eval()
                logger.info("MuQ model loaded successfully on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load MuQ model: {e}")
            self.audio_model = None
    
    def _setup_vertex_ai(self):
        """Initialize Vertex AI for text embeddings."""
        try:
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            region = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
            
            if not project_id:
                logger.warning("GOOGLE_CLOUD_PROJECT not set. Vertex AI disabled.")
                return
            
            # Initialize Vertex AI
            aiplatform.init(project=project_id, location=region)
            
            logger.info("Vertex AI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
    
    def generate_audio_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Generate embedding for an audio file using MuQ.
        
        Args:
            audio_path: Path to the audio file (.wav)
        
        Returns:
            Numpy array of shape (embedding_dim,) or None if failed
        """
        if self.audio_model is None:
            logger.warning("Audio model not available")
            return None
        
        wav = None
        wavs = None
        output = None
        embedding_tensor = None
        
        try:
            # Load audio file
            wav, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Create tensor and move to device with fallback
            try:
                wavs = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(self.device)
            except Exception as device_error:
                logger.warning(f"Failed to move tensor to {self.device}: {device_error}, using CPU")
                wavs = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to('cpu')
            
            # Generate embedding
            with torch.no_grad():
                output = self.audio_model(wavs, output_hidden_states=True)
                # Use mean pooling over time dimension
                embedding_tensor = output.last_hidden_state.mean(dim=1).squeeze()
                # Always move to CPU before converting to numpy
                embedding = embedding_tensor.cpu().numpy().copy()
            
            logger.info(f"Generated audio embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate audio embedding for {audio_path}: {e}")
            return None
        
        finally:
            # Explicit memory cleanup
            del wav, wavs, output, embedding_tensor
            
            # Clear PyTorch cache
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            elif self.device == 'mps':
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            
            # Force garbage collection
            gc.collect()
    
    def generate_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using Vertex AI.
        
        Args:
            text: Input text (lyrics, metadata, etc.)
        
        Returns:
            Numpy array of shape (embedding_dim,) or None if failed
        """
        if not text or not text.strip():
            return None
        
        model = None
        embeddings = None
        
        try:
            from vertexai.language_models import TextEmbeddingModel
            
            model_name = CONFIG['vertex_ai']['text_embedding_model']
            model = TextEmbeddingModel.from_pretrained(model_name)
            
            # Generate embedding
            embeddings = model.get_embeddings([text])
            embedding = np.array(embeddings[0].values).copy()
            
            logger.info(f"Generated text embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return None
        
        finally:
            # Explicit memory cleanup
            del model, embeddings
            gc.collect()
    
    def generate_combined_text_embedding(self, song_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Generate embedding from combined song metadata and lyrics.
        
        Args:
            song_data: Dictionary with song metadata
        
        Returns:
            Text embedding or None
        """
        # Combine relevant text fields
        text_parts = []
        
        if song_data.get('song_name'):
            text_parts.append(f"Song: {song_data['song_name']}")
        if song_data.get('singers'):
            text_parts.append(f"Singers: {song_data['singers']}")
        if song_data.get('composer'):
            text_parts.append(f"Composer: {song_data['composer']}")
        if song_data.get('lyricist'):
            text_parts.append(f"Lyricist: {song_data['lyricist']}")
        if song_data.get('genre'):
            text_parts.append(f"Genre: {song_data['genre']}")
        if song_data.get('lyrics'):
            text_parts.append(f"Lyrics: {song_data['lyrics']}")
        
        combined_text = " | ".join(text_parts)
        return self.generate_text_embedding(combined_text)


# Global instances
_es_client = None
_embedding_generator = None


def get_es_client() -> ElasticsearchClient:
    """Get or create Elasticsearch client singleton."""
    global _es_client
    if _es_client is None:
        _es_client = ElasticsearchClient()
    return _es_client


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create embedding generator singleton."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def create_song_index(index_name: Optional[str] = None) -> bool:
    """
    Create Elasticsearch index for songs with appropriate mappings.
    
    Args:
        index_name: Name of the index (defaults to config value)
    
    Returns:
        True if successful, False otherwise
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    es_client = get_es_client().get_client()
    
    # Check if index already exists
    if es_client.indices.exists(index=index_name):
        logger.warning(f"Index '{index_name}' already exists")
        return False
    
    # Define index mapping
    mapping = {
        "mappings": {
            "properties": {
                "song_index": {"type": "integer"},
                "song_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "song_file_path_name": {"type": "keyword"},
                "composer": {"type": "keyword"},
                "album": {"type": "keyword"},
                "released_year": {"type": "integer"},
                "genre": {"type": "keyword"},
                "lyricist": {"type": "keyword"},
                "singers": {"type": "keyword"},
                "lyrics": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "lyrics_vector": {
                    "type": "dense_vector",
                    "dims": CONFIG['embeddings']['text_dims'],
                    "index": True,
                    "similarity": "cosine"
                },
                "audio_vector": {
                    "type": "dense_vector",
                    "dims": CONFIG['embeddings']['audio_dims'],
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }
    
    # Add settings only for localhost (not for serverless deployments)
    es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    if 'localhost' in es_url.lower() or '127.0.0.1' in es_url:
        mapping["settings"] = {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
        logger.info("Using localhost Elasticsearch - adding shard/replica settings")
    else:
        logger.info("Using cloud/serverless Elasticsearch - skipping shard/replica settings")
    
    try:
        es_client.indices.create(index=index_name, body=mapping)
        logger.info(f"Successfully created index '{index_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to create index '{index_name}': {e}")
        return False


def delete_song_index(index_name: Optional[str] = None) -> bool:
    """
    Delete Elasticsearch index.
    
    Args:
        index_name: Name of the index (defaults to config value)
    
    Returns:
        True if successful, False otherwise
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    es_client = get_es_client().get_client()
    
    try:
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            logger.info(f"Successfully deleted index '{index_name}'")
            return True
        else:
            logger.warning(f"Index '{index_name}' does not exist")
            return False
    except Exception as e:
        logger.error(f"Failed to delete index '{index_name}': {e}")
        return False


def initialize_and_load_demo_data(
    index_name: Optional[str] = None,
    csv_path: str = "dataset/dataset_meta.csv",
    audio_dir: str = "dataset",
    progress_callback: Optional[callable] = None
) -> Tuple[int, int]:
    """
    Initialize index (delete if exists, create new) and load demo data.
    This is a combined operation that ensures a fresh index with demo data.
    
    Args:
        index_name: Name of the index (defaults to config value)
        csv_path: Path to CSV file with metadata
        audio_dir: Directory containing audio files
        progress_callback: Optional callback function(current, total, song_name) to report progress
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    es_client = get_es_client().get_client()
    
    # Step 1: Delete index if it exists
    try:
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            logger.info(f"Deleted existing index '{index_name}'")
    except Exception as e:
        logger.error(f"Failed to delete existing index '{index_name}': {e}")
        return (0, 0)
    
    # Step 2: Create new index
    if not create_song_index(index_name):
        logger.error(f"Failed to create index '{index_name}'")
        return (0, 0)
    
    # Step 3: Load demo data
    return load_demo_data(index_name, csv_path, audio_dir, progress_callback)


def load_demo_data(
    index_name: Optional[str] = None,
    csv_path: str = "dataset/dataset_meta.csv",
    audio_dir: str = "dataset",
    progress_callback: Optional[callable] = None
) -> Tuple[int, int]:
    """
    Load demo data from CSV and audio files into Elasticsearch.
    Indexes songs one at a time to prevent memory issues with large datasets.
    
    Args:
        index_name: Name of the index
        csv_path: Path to CSV file with metadata
        audio_dir: Directory containing audio files
        progress_callback: Optional callback function(current, total, song_name) to report progress
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    es_client = get_es_client().get_client()
    embedding_gen = get_embedding_generator()
    
    # Check if index exists
    if not es_client.indices.exists(index=index_name):
        logger.error(f"Index '{index_name}' does not exist. Create it first.")
        return (0, 0)
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} songs from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return (0, 0)
    
    # Add song_index column
    df['song_index'] = range(1, len(df) + 1)
    
    successful = 0
    failed = 0
    total_songs = len(df)
    
    # Index songs one at a time
    for idx, row in df.iterrows():
        text_embedding = None
        audio_embedding = None
        
        try:
            # Prepare song data
            song_data = {
                'song_index': int(row['song_index']),
                'song_name': str(row['song_name']),
                'song_file_path_name': str(row['song_file_path_name']),
                'composer': str(row.get('composer', '')),
                'album': str(row.get('album', '')),
                'released_year': int(row['released_year']) if pd.notna(row.get('released_year')) else None,
                'genre': str(row.get('genre', '')),
                'lyricist': str(row.get('lyricist', '')),
                'singers': str(row.get('singers', '')),
                'lyrics': str(row.get('lyrics', ''))
            }
            
            # Generate text embedding from metadata + lyrics
            text_embedding = embedding_gen.generate_combined_text_embedding(song_data)
            if text_embedding is not None:
                song_data['lyrics_vector'] = text_embedding.tolist()
            
            # Generate audio embedding
            audio_path = os.path.join(audio_dir, row['song_file_path_name'])
            if os.path.exists(audio_path):
                audio_embedding = embedding_gen.generate_audio_embedding(audio_path)
                if audio_embedding is not None:
                    song_data['audio_vector'] = audio_embedding.tolist()
            else:
                logger.warning(f"Audio file not found: {audio_path}")
            
            # Index this song immediately
            es_client.index(
                index=index_name,
                id=str(row['song_index']),
                document=song_data
            )
            
            successful += 1
            logger.info(f"Indexed song {idx + 1}/{total_songs}: {row['song_name']}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(idx + 1, total_songs, row['song_name'])
            
        except Exception as e:
            failed += 1
            logger.error(f"Failed to index song {idx + 1}/{total_songs} ({row.get('song_name', 'Unknown')}): {e}")
        
        finally:
            # CRITICAL: Clear embeddings from memory after each song
            del text_embedding, audio_embedding
            
            # Clear PyTorch cache after EVERY song (not periodically)
            # This is crucial for preventing memory buildup
            if embedding_gen.device == 'cuda':
                torch.cuda.empty_cache()
            elif embedding_gen.device == 'mps':
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            
            # Force garbage collection after each song to prevent memory buildup
            gc.collect()
            
            logger.info(f"Memory cleanup performed after song {idx + 1}")
    
    logger.info(f"Indexing complete: {successful} successful, {failed} failed")
    return (successful, failed)


def add_single_song(
    song_name: str,
    composer: str,
    album: str,
    released_year: int,
    genre: str,
    lyricist: str,
    singers: str,
    audio_file_path: str,
    lyrics: str,
    index_name: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Add a single song to the Elasticsearch index with embeddings.
    Uses the existing device detection (CUDA/MPS/CPU) from EmbeddingGenerator.
    Supports both WAV and MP3 formats (MP3 files are automatically converted to WAV).
    Files are saved to the dataset folder.
    
    Args:
        song_name: Name of the song (required)
        composer: Composer name (required)
        album: Album/Movie name (required)
        released_year: Year the song was released (required)
        genre: Genre of the song (required)
        lyricist: Lyricist name (required)
        singers: Singer names (required, comma-separated)
        audio_file_path: Path to the audio file (required, .wav or .mp3 format)
        lyrics: Lyrics of the song (required)
        index_name: Name of the index (defaults to config value)
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    es_client = get_es_client().get_client()
    embedding_gen = get_embedding_generator()
    
    # Check if index exists
    if not es_client.indices.exists(index=index_name):
        error_msg = f"Index '{index_name}' does not exist. Please initialize the index first."
        logger.error(error_msg)
        return (False, error_msg)
    
    # Validate audio file
    if not os.path.exists(audio_file_path):
        error_msg = f"Audio file not found: {audio_file_path}"
        logger.error(error_msg)
        return (False, error_msg)
    
    # Check file extension and convert MP3 to WAV if needed
    file_ext = audio_file_path.lower()
    converted_file = None
    original_audio_path = audio_file_path  # Keep original path for filename
    
    if file_ext.endswith('.mp3'):
        logger.info(f"MP3 file detected, converting to WAV: {audio_file_path}")
        try:
            # Load MP3 and convert to WAV using librosa and soundfile
            audio, sample_rate = librosa.load(audio_file_path, sr=None, mono=False)
            
            # Create temporary WAV file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
                converted_file = tmp_wav.name
            
            # Save as WAV
            sf.write(converted_file, audio.T if audio.ndim > 1 else audio, sample_rate)
            logger.info(f"Successfully converted MP3 to WAV: {converted_file}")
            
            # Use the converted file for processing
            audio_file_path = converted_file
            
        except Exception as e:
            error_msg = f"Failed to convert MP3 to WAV: {str(e)}"
            logger.error(error_msg)
            return (False, error_msg)
    
    elif not file_ext.endswith('.wav'):
        error_msg = "Only .wav and .mp3 files are supported"
        logger.error(error_msg)
        return (False, error_msg)
    
    try:
        # Get the next song index by counting existing documents
        count_response = es_client.count(index=index_name)
        next_song_index = count_response['count'] + 1
        
        # Extract filename from original path (preserve original extension)
        song_file_name = os.path.basename(original_audio_path)
        
        # Create dataset directory if it doesn't exist
        dataset_dir = Path(__file__).parent / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save audio file to dataset folder
        dataset_file_path = dataset_dir / song_file_name
        
        # Copy/convert file to dataset folder
        if converted_file and os.path.exists(converted_file):
            # If we converted from MP3, save the converted WAV
            import shutil
            shutil.copy2(converted_file, dataset_file_path)
            logger.info(f"Saved converted audio file to: {dataset_file_path}")
        else:
            # Copy the original file
            import shutil
            shutil.copy2(audio_file_path, dataset_file_path)
            logger.info(f"Saved audio file to: {dataset_file_path}")
        
        # Prepare song data
        song_data = {
            'song_index': next_song_index,
            'song_name': song_name,
            'song_file_path_name': song_file_name,
            'composer': composer,
            'album': album,
            'released_year': released_year,
            'genre': genre,
            'lyricist': lyricist,
            'singers': singers,
            'lyrics': lyrics
        }
        
        logger.info(f"Processing song: {song_name} on device: {embedding_gen.device}")
        
        # Generate text embedding from metadata + lyrics
        text_embedding = None
        try:
            text_embedding = embedding_gen.generate_combined_text_embedding(song_data)
            if text_embedding is not None:
                song_data['lyrics_vector'] = text_embedding.tolist()
                logger.info(f"Generated text embedding for: {song_name}")
            else:
                logger.warning(f"Failed to generate text embedding for: {song_name}")
        except Exception as e:
            logger.warning(f"Text embedding generation failed: {e}")
        
        # Generate audio embedding using the saved file
        audio_embedding = None
        try:
            audio_embedding = embedding_gen.generate_audio_embedding(str(dataset_file_path))
            if audio_embedding is not None:
                song_data['audio_vector'] = audio_embedding.tolist()
                logger.info(f"Generated audio embedding for: {song_name}")
            else:
                logger.warning(f"Failed to generate audio embedding for: {song_name}")
        except Exception as e:
            logger.warning(f"Audio embedding generation failed: {e}")
        
        # Index the song
        es_client.index(
            index=index_name,
            id=str(next_song_index),
            document=song_data
        )
        
        logger.info(f"Successfully indexed song: {song_name} (index: {next_song_index})")
        
        return (True, None)
        
    except Exception as e:
        error_msg = f"Failed to index song: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return (False, error_msg)
    
    finally:
        # Clean up converted MP3 file if it was created
        if converted_file and os.path.exists(converted_file):
            try:
                os.unlink(converted_file)
                logger.debug(f"Cleaned up converted file: {converted_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup converted file: {e}")
        
        # Clean up memory
        if 'text_embedding' in locals():
            del text_embedding
        if 'audio_embedding' in locals():
            del audio_embedding
        
        # Clear device cache
        if embedding_gen.device == 'cuda':
            torch.cuda.empty_cache()
        elif embedding_gen.device == 'mps':
            try:
                torch.mps.empty_cache()
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        logger.debug("Memory cleanup performed after adding song")


def get_all_songs(index_name: Optional[str] = None, size: int = 1000) -> List[Dict]:
    """
    Retrieve all songs from the index.
    
    Args:
        index_name: Name of the index
        size: Maximum number of songs to retrieve
    
    Returns:
        List of song dictionaries
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    es_client = get_es_client().get_client()
    
    try:
        response = es_client.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": size,
                "_source": {
                    "excludes": ["lyrics_vector", "audio_vector"]  # Exclude large vectors
                }
            }
        )
        
        songs = []
        for hit in response['hits']['hits']:
            song = hit['_source']
            song['_id'] = hit['_id']
            song['_score'] = hit['_score']
            songs.append(song)
        
        logger.info(f"Retrieved {len(songs)} songs from index")
        return songs
        
    except Exception as e:
        logger.error(f"Failed to retrieve songs: {e}")
        return []


def hybrid_score(bm25_score: float, vector_score: float, alpha: Optional[float] = None) -> float:
    """
    Calculate hybrid score combining BM25 and vector similarity.
    
    Args:
        bm25_score: BM25 relevance score
        vector_score: Cosine similarity score
        alpha: Weight for BM25 (0-1). If None, uses config value.
    
    Returns:
        Combined score
    """
    if alpha is None:
        alpha = CONFIG['search']['hybrid_alpha']
    
    return alpha * bm25_score + (1 - alpha) * vector_score


def search_songs(
    query_text: Optional[str] = None,
    query_audio_path: Optional[str] = None,
    index_name: Optional[str] = None,
    top_k: Optional[int] = None
) -> List[Dict]:
    """
    Search songs using hybrid approach (text + audio).
    
    Args:
        query_text: Text query (lyrics, song name, composer, etc.)
        query_audio_path: Path to audio file for similarity search
        index_name: Name of the index
        top_k: Number of results to return
    
    Returns:
        List of matching songs with scores
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    if top_k is None:
        top_k = CONFIG['search']['top_k']
    
    es_client = get_es_client().get_client()
    embedding_gen = get_embedding_generator()
    
    text_embedding = None
    audio_embedding = None
    
    try:
        # Build query
        must_queries = []
        should_queries = []
        
        # Text-based search (BM25)
        if query_text:
            should_queries.append({
                "multi_match": {
                    "query": query_text,
                    "fields": ["song_name^3", "lyrics^2", "singers", "composer", "lyricist", "genre"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
            
            # Vector search on text embedding
            text_embedding = embedding_gen.generate_text_embedding(query_text)
            if text_embedding is not None:
                should_queries.append({
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'lyrics_vector') + 1.0",
                            "params": {"query_vector": text_embedding.tolist()}
                        }
                    }
                })
        
        # Audio-based search (vector similarity)
        if query_audio_path and os.path.exists(query_audio_path):
            audio_embedding = embedding_gen.generate_audio_embedding(query_audio_path)
            if audio_embedding is not None:
                should_queries.append({
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'audio_vector') + 1.0",
                            "params": {"query_vector": audio_embedding.tolist()}
                        }
                    }
                })
        
        # Build final query
        if not should_queries:
            logger.warning("No valid query provided")
            return []
        
        search_query = {
            "query": {
                "bool": {
                    "should": should_queries,
                    "minimum_should_match": 1
                }
            },
            "size": top_k,
            "_source": {
                "excludes": ["lyrics_vector", "audio_vector"]
            }
        }
        
        response = es_client.search(index=index_name, body=search_query)
        
        results = []
        for hit in response['hits']['hits']:
            song = hit['_source']
            song['_id'] = hit['_id']
            song['_score'] = hit['_score']
            results.append(song)
        
        logger.info(f"Found {len(results)} matching songs")
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []
    
    finally:
        # Clear query embeddings from memory
        del text_embedding, audio_embedding
        gc.collect()
        
        # Clear PyTorch cache if using GPU
        if embedding_gen.device == 'cuda':
            torch.cuda.empty_cache()
        elif embedding_gen.device == 'mps':
            try:
                torch.mps.empty_cache()
            except:
                pass


def rerank_with_vertex_ai(query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Re-rank search results using Vertex AI for semantic relevance.
    
    Args:
        query: Original search query
        results: List of search results
        top_k: Number of top results to return
    
    Returns:
        Re-ranked list of results
    """
    # Placeholder for Vertex AI re-ranking
    # This would use Gemini or a ranking model to re-score results
    # For now, just return the original results
    logger.info("Vertex AI re-ranking not yet implemented")
    return results[:top_k]


def get_index_stats(index_name: Optional[str] = None) -> Dict:
    """
    Get statistics about the index.
    Compatible with both traditional and serverless Elasticsearch.
    
    Args:
        index_name: Name of the index
    
    Returns:
        Dictionary with index statistics
    """
    if index_name is None:
        index_name = CONFIG['elasticsearch']['index_name']
    
    es_client = get_es_client().get_client()
    es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    is_serverless = 'localhost' not in es_url.lower() and '127.0.0.1' not in es_url
    
    try:
        # Check if index exists
        exists = es_client.indices.exists(index=index_name)
        
        if not exists:
            return {
                "exists": False,
                "index_name": index_name
            }
        
        # Get document count (available in both modes)
        count_response = es_client.count(index=index_name)
        
        result = {
            "exists": True,
            "index_name": index_name,
            "doc_count": count_response['count']
        }
        
        # Try to get size stats (only available in traditional Elasticsearch)
        if not is_serverless:
            try:
                stats = es_client.indices.stats(index=index_name)
                size_bytes = stats['_all']['primaries']['store']['size_in_bytes']
                
                # Format bytes to human-readable format
                def format_bytes(bytes_val):
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if bytes_val < 1024.0:
                            return f"{bytes_val:.2f} {unit}"
                        bytes_val /= 1024.0
                    return f"{bytes_val:.2f} PB"
                
                result["size_in_bytes"] = size_bytes
                result["size_readable"] = format_bytes(size_bytes)
            except Exception as stats_error:
                logger.warning(f"Could not retrieve size stats: {stats_error}")
        else:
            logger.info("Serverless mode detected - size stats not available")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        return {
            "exists": False,
            "error": str(e)
        }
