import streamlit as st
import os
import random
import time
import base64
from pathlib import Path
import tempfile
import yaml
from audio_recorder_streamlit import audio_recorder

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="SingN'Seek",
    page_icon="images/logo.png",
    layout="centered"
)

# Configure logging
from utils.logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import utility functions for Elasticsearch and search
try:
    from utils import utils
    UTILS_AVAILABLE = True
    logger.info("Utils module loaded successfully")
except ImportError as e:
    UTILS_AVAILABLE = False
    logger.error(f"Utils module not available: {e}")
    st.warning("⚠️ Utils module not available. Search functionality will be limited.")

# Load configuration
CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
        max-width: 800px;
    }
    .song-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .song-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #262730;
        margin-bottom: 0.3rem;
    }
    .song-info {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.2rem;
    }
    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        font-size: 1.3rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Hide download button in audio player - works for Chrome/Brave/Safari */
    audio {
        width: 100%;
    }
    audio::-webkit-media-controls-enclosure {
        overflow: hidden;
    }
    audio::-webkit-media-controls-panel {
        width: calc(100% + 32px);
    }
    audio::-internal-media-controls-download-button {
        display: none !important;
    }
    audio::-webkit-media-controls-download-button {
        display: none !important;
    }
    /* For Chromium-based browsers like Brave */
    audio::part(download-button) {
        display: none !important;
    }
    /* Alternative approach - hide all extra controls */
    [data-testid="stAudio"] audio::-webkit-media-controls-enclosure {
        overflow: hidden;
    }
    [data-testid="stAudio"] audio::-webkit-media-controls-download-button {
        display: none !important;
    }
    /* Hide "Press Enter to submit form" and "Press Enter to apply" text */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    /* Text area styling with line-height */
    .stTextArea textarea {
        line-height: 1.5 !important;
    }
    /* Audio recorder component styling */
    .st-key-audio-recorder-container {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        min-height: 50px !important;
        padding: 1rem 0 !important;
    }
    /* Center the audio recorder iframe */
    iframe[title="audio_recorder_streamlit.audio_recorder"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 350px !important;
        height: 50px !important;
        margin: 0 auto !important;
        border: none !important;
    }
    /* Lyrics preview styling */
    .lyrics-preview {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript to change "Browse files" to "Browse file"
st.components.v1.html("""
    <script>
        function updateFileUploader() {
            // Change "Browse files" to "Browse file"
            const buttons = window.parent.document.querySelectorAll('[data-testid="stFileUploader"] button');
            buttons.forEach(button => {
                if (button.textContent.includes('Browse files')) {
                    button.textContent = button.textContent.replace('Browse files', 'Browse file');
                }
            });
        }
        
        // Run immediately
        updateFileUploader();
        
        // Also run after a short delay to catch dynamically loaded content
        setTimeout(updateFileUploader, 100);
        setTimeout(updateFileUploader, 500);
        
        // Set up a mutation observer to catch future changes
        const observer = new MutationObserver(updateFileUploader);
        observer.observe(window.parent.document.body, { childList: true, subtree: true });
    </script>
""", height=0)

# Initialize session state
if 'search_state' not in st.session_state:
    st.session_state.search_state = 'idle'  # idle, searching, searched
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ''
if 'scroll_to_results' not in st.session_state:
    st.session_state.scroll_to_results = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'show_lyrics_modal' not in st.session_state:
    st.session_state.show_lyrics_modal = False
if 'current_lyrics' not in st.session_state:
    st.session_state.current_lyrics = None
if 'current_song_title' not in st.session_state:
    st.session_state.current_song_title = None
if 'es_connected' not in st.session_state:
    st.session_state.es_connected = None  # None = not checked yet, True = connected, False = failed
if 'es_connection_error' not in st.session_state:
    st.session_state.es_connection_error = None

def get_random_songs(count=5):
    """Get random songs from the dataset folder"""
    dataset_path = Path("dataset")
    if dataset_path.exists():
        songs = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
        selected_songs = random.sample(songs, min(count, len(songs)))
        return selected_songs
    return []

def get_all_songs():
    """Get all songs from the dataset folder"""
    dataset_path = Path("dataset")
    if dataset_path.exists():
        songs = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
        return sorted(songs)  # Return sorted list for consistency
    return []

@st.dialog("Song Lyrics")
def show_lyrics_modal(song_title, lyrics):
    """Display lyrics in a modal dialog with scroll capability"""
    st.markdown(f"### {song_title}")
    st.markdown("---")
    
    # Display lyrics with proper formatting
    st.markdown(
        f"""
        <div style='max-height: 400px; overflow-y: auto; padding: 1rem; 
                    background-color: #f0f2f6; border-radius: 8px; 
                    line-height: 1.8; white-space: pre-wrap;'>
            {lyrics}
        </div>
        """,
        unsafe_allow_html=True
    )

def check_es_connection():
    """
    Check Elasticsearch connection and update session state.
    This is called after the page renders to avoid blocking.
    """
    if not UTILS_AVAILABLE:
        st.session_state.es_connected = False
        st.session_state.es_connection_error = "Utils module not available"
        return
    
    if st.session_state.es_connected is None:
        try:
            logger.info("Attempting to connect to Elasticsearch")
            es_client = utils.get_es_client()
            st.session_state.es_connected = True
            st.session_state.es_connection_error = None
            logger.info("Successfully connected to Elasticsearch")
        except Exception as e:
            st.session_state.es_connected = False
            st.session_state.es_connection_error = str(e)
            logger.error(f"Failed to connect to Elasticsearch: {e}", exc_info=True)

def perform_search(query_text, audio_data=None, audio_file=None):
    """
    Perform actual search using Elasticsearch and embeddings.
    Falls back to simulated search if utils are not available.
    """
    if UTILS_AVAILABLE:
        try:
            logger.info(f"Performing search - query_text: {query_text}, has_audio: {audio_data is not None or audio_file is not None}")
            
            # Save audio data to temporary file if provided
            temp_audio_path = None
            if audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_data)
                    temp_audio_path = tmp.name
                logger.info(f"Saved audio data to temporary file: {temp_audio_path}")
            elif audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_file.getvalue())
                    temp_audio_path = tmp.name
                logger.info(f"Saved uploaded audio file to temporary file: {temp_audio_path}")
            
            # Perform search
            results = utils.search_songs(
                query_text=query_text if query_text and query_text.strip() else None,
                query_audio_path=temp_audio_path
            )
            
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                logger.debug(f"Cleaned up temporary file: {temp_audio_path}")
            
            # Update session state with results
            # search_songs now returns a list of results (up to 2 with >25% confidence)
            if results:
                st.session_state.search_results = results
                st.session_state.search_state = 'searched'
                logger.info(f"Search completed successfully - found {len(results)} result(s)")
            else:
                st.session_state.search_results = []
                st.session_state.search_state = 'no_results'
                logger.info("Search completed - no results found")
                
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            st.error(f"Search error: {str(e)}")
            st.session_state.search_results = []
            st.session_state.search_state = 'no_results'
    else:
        # Fallback: Simulate search
        time.sleep(2)
        num_results = random.randint(0, 3)
        if num_results > 0:
            st.session_state.search_results = get_random_songs(num_results)
            st.session_state.search_state = 'searched'
        else:
            st.session_state.search_results = []
            st.session_state.search_state = 'no_results'

# ===== SECTION 1: LOGO AND TITLE =====
logo_path = "images/logo.png"
if os.path.exists(logo_path):
    col_logo, col_title, col_nav = st.columns([0.6, 1.8, 2.6])
    with col_logo:
        st.image(logo_path, width=50)
    with col_title:
        st.markdown("<h1 style='margin-top: 0; margin-bottom: 0; margin-left: -1rem; padding-top: 0.3rem; font-size: 1.5rem;'>SingN'Seek</h1>", 
                    unsafe_allow_html=True)
    with col_nav:
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            if st.button("Home", key="home_btn", use_container_width=True, 
                        type="primary" if st.session_state.current_page == 'home' else "secondary"):
                st.session_state.current_page = 'home'
                st.rerun()
        with nav_col2:
            if st.button("All Songs", key="all_songs_btn", use_container_width=True,
                        type="primary" if st.session_state.current_page == 'all_songs' else "secondary"):
                st.session_state.current_page = 'all_songs'
                st.rerun()
        with nav_col3:
            if st.button("Manage", key="add_song_btn", use_container_width=True,
                        type="primary" if st.session_state.current_page == 'add_song' else "secondary"):
                st.session_state.current_page = 'add_song'
                st.rerun()
else:
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>SingN'Seek</h1>", 
                unsafe_allow_html=True)

st.markdown("---")

# Create a placeholder for connection status that renders immediately
status_placeholder = st.empty()

# Show Elasticsearch connection status (non-blocking)
if UTILS_AVAILABLE:
    if st.session_state.es_connected is None:
        # Show checking status but don't block - connection check happens below
        status_placeholder.info("🔄 Checking Elasticsearch connection...")
    elif st.session_state.es_connected is False:
        status_placeholder.error(f"❌ Failed to connect to Elasticsearch: {st.session_state.es_connection_error}")
        st.info("💡 Make sure Elasticsearch is running and check your .env configuration.")
    # Don't show success message here - it's already connected, no need to keep showing it

# ===== PAGE ROUTING =====
if st.session_state.current_page == 'home':
    # ===== SECTION 2: SEARCH INPUT =====
    st.markdown("<h3 style='text-align: center;'>Can't remember the song? Just hum, type, or guess & we'll find it! ✨</h3>", unsafe_allow_html=True)

    # Spacing
    st.markdown("""
        <div style='text-align: center; padding: 1rem 1rem; margin-bottom: 1rem;'>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state for audio inputs
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None

    # Variables to store inputs
    search_query = ""
    audio_file = None
    audio_bytes = None

    # TEXT SEARCH - Always visible
    search_query = st.text_area(
        "Search",
        placeholder="Type what you think you heard: lyrics, movie, or composer. Or just hum it and leave the box empty!",
        label_visibility="collapsed",
        key="search_input"
    )

    # AUDIO INPUT - Record or Upload

    # Create tabs for record vs upload
    audio_tab1, audio_tab2 = st.tabs(["Upload Audio", "Record Audio"])

    with audio_tab1:
        
        audio_file = st.file_uploader(
            "Upload audio file",
            type=['wav','mp3'],
            label_visibility="collapsed",
            help="Upload your audio file (Max 200 MB)",
            key="file_uploader",
            accept_multiple_files=False
        )
        
        if audio_file:
            # Check file size (200 MB = 200 * 1024 * 1024 bytes)
            max_size = 200 * 1024 * 1024
            if audio_file.size > max_size:
                st.error(f"File size ({audio_file.size / (1024 * 1024):.1f} MB) exceeds the 200 MB limit. Please upload a smaller file.")
                st.session_state.audio_file = None
            else:
                st.session_state.audio_file = audio_file
                st.success("File uploaded successfully!")
                st.audio(audio_file)

    with audio_tab2:
        # Use Streamlit audio recorder component
        audio_bytes = audio_recorder(
            pause_threshold=5.0,
            text="Click the mic icon to start/stop recording",
            recording_color="#ff4b4b",
            neutral_color="#31333f",
            icon_name="microphone-lines",
            icon_size="2x",
            auto_start=False,
            key="audio-recorder-container"
        )
        if audio_bytes:
            # Store the recorded audio in session state
            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_file = None  # Clear any uploaded file
            st.audio(audio_bytes, format="audio/wav")

    # Get the current audio inputs from session state
    if not audio_file:
        audio_file = st.session_state.audio_file
    if not audio_bytes:
        audio_bytes = st.session_state.audio_bytes

    # Add spacing between audio input and search button
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Search button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        # Determine button state
        if not UTILS_AVAILABLE or st.session_state.es_connected is False:
            button_text = "Elasticsearch Not Connected"
            button_disabled = True
        elif st.session_state.es_connected is None:
            button_text = "Connecting to Elasticsearch..."
            button_disabled = True
        elif st.session_state.search_state == 'searching':
            button_text = "Searching..."
            button_disabled = True
        else:
            button_text = "Find My Song!"
            button_disabled = False
        
        if st.button(button_text, use_container_width=True, type="primary", disabled=button_disabled):
            if search_query or audio_file or audio_bytes:
                # Stop all playing audio before starting new search
                st.components.v1.html("""
                    <script>
                        // Stop all audio elements
                        var audioElements = window.parent.document.getElementsByTagName('audio');
                        for(var i = 0; i < audioElements.length; i++) {
                            audioElements[i].pause();
                            audioElements[i].currentTime = 0;
                        }
                    </script>
                """, height=0)
                
                st.session_state.search_state = 'searching'
                st.session_state.search_query = search_query  # Save search query
                st.session_state.search_results = []  # Clear previous results
                st.session_state.scroll_to_results = True  # Enable scroll for new search
                # Reset audio inputs after search
                st.session_state.show_recorder = False
                st.session_state.show_uploader = False
                st.rerun()
            else:
                st.warning("Please enter a search term, record audio, or upload an audio file!")

    st.markdown("---")

    # ===== SECTION 3: RESULTS/EMPTY STATE =====
    # Add anchor point for auto-scroll
    st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)

    # State: SEARCHING (Loading)
    if st.session_state.search_state == 'searching':
        # Stop all playing audio when searching
        st.components.v1.html("""
            <script>
                var audioElements = window.parent.document.getElementsByTagName('audio');
                for(var i = 0; i < audioElements.length; i++) {
                    audioElements[i].pause();
                    audioElements[i].currentTime = 0;
                }
            </script>
        """, height=0)
        
        # Auto-scroll to searching section immediately
        if st.session_state.scroll_to_results:
            st.components.v1.html("""
                <script>
                    window.parent.document.getElementById('results-section').scrollIntoView({behavior: 'smooth', block: 'start'});
                </script>
            """, height=0)
        
        col_loading1, col_loading2, col_loading3 = st.columns([1, 2, 1])
        with col_loading2:
            with st.spinner("Analyzing your input and matching with our database..."):
                perform_search(
                    st.session_state.get('search_query', ''),
                    audio_data=st.session_state.get('audio_bytes'),
                    audio_file=st.session_state.get('audio_file')
                )
            st.rerun()

    # State: SEARCHED - RESULTS FOUND
    elif st.session_state.search_state == 'searched' and st.session_state.search_results:
        st.markdown("<h2 style='text-align: center; color: #FF4B4B; font-size: 1.5rem;'>Top Matches</h2>", 
                    unsafe_allow_html=True)
        num_results = len(st.session_state.search_results)
        result_text = "result" if num_results == 1 else "results"
        st.markdown(f"<p style='text-align: center; color: #666; margin-bottom: 1.2rem; font-size: 0.9rem;'>Found {num_results} confident {result_text} for your search!</p>", 
                    unsafe_allow_html=True)
        
        # Auto-scroll to results after rendering
        if st.session_state.scroll_to_results:
            st.components.v1.html("""
                <script>
                    window.parent.document.getElementById('results-section').scrollIntoView({behavior: 'smooth', block: 'start'});
                </script>
            """, height=0)
            st.session_state.scroll_to_results = False
        
        for idx, song_result in enumerate(st.session_state.search_results, 1):
            # Handle both dict (from Elasticsearch) and string (fallback) results
            if isinstance(song_result, dict):
                song_data = {
                    'title': song_result.get('song_name', 'Unknown'),
                    'composer': song_result.get('composer', 'Unknown'),
                    'genre': song_result.get('genre', ''),
                    'album': song_result.get('album', ''),
                    'singers': song_result.get('singers', ''),
                    'lyrics': song_result.get('lyrics', ''),
                    'released_year': song_result.get('released_year', ''),
                    'score': song_result.get('_score', 0)
                }
                song_file = song_result.get('song_file_path_name', '')
            else:
                # Handle string fallback case
                song_data = {
                    'title': str(song_result),
                    'composer': 'Unknown',
                    'genre': '',
                    'album': '',
                    'singers': '',
                    'lyrics': '',
                    'released_year': '',
                    'score': 0
                }
                song_file = ''
            
            # Build song card HTML with optional fields
            song_card_html = f"""
                <div class="song-card">
                    <div class="song-title">#{idx} {song_data.get('title', 'Unknown')}</div>
            """
            
            # Always show the raw Elasticsearch _score
            if song_data.get('score'):
                song_card_html += f"<div class='song-info'>📊 ES Score: {song_data['score']:.4f}</div>"
            
            # Add composer
            song_card_html += f"<div class='song-info'>Composed by {song_data.get('composer', 'Unknown')}</div>"
            
            # Add singers if available
            if song_data.get('singers'):
                song_card_html += f"<div class='song-info'>Singers: {song_data['singers']}</div>"
            
            # Add genre if available
            if song_data.get('genre'):
                song_card_html += f"<div class='song-info'>Genre: {song_data['genre']}</div>"
            
            # Add album if available
            if song_data.get('album'):
                song_card_html += f"<div class='song-info'>Album: {song_data['album']}</div>"
            
            # Add year if available
            if song_data.get('released_year'):
                song_card_html += f"<div class='song-info'>Year: {song_data['released_year']}</div>"
            
            song_card_html += "</div>"
            
            st.markdown(song_card_html, unsafe_allow_html=True)
            
            # Add lyrics preview and View Lyrics button if available
            if song_data.get('lyrics'):
                lyrics_col1, lyrics_col2 = st.columns([3, 1])
                with lyrics_col1:
                    lyrics_preview = song_data['lyrics'][:100] + "..." if len(song_data['lyrics']) > 100 else song_data['lyrics']
                    st.markdown(f"<div class='lyrics-preview'>Lyrics: {lyrics_preview}</div>", unsafe_allow_html=True)
                with lyrics_col2:
                    if st.button("View Lyrics", key=f"lyrics_btn_top_{idx}", use_container_width=True):
                        show_lyrics_modal(song_data.get('title', 'Unknown'), song_data['lyrics'])
            
            # Use st.audio with lazy loading (preload="none")
            if song_file:
                song_path = f"dataset/{song_file}"
                if os.path.exists(song_path):
                    st.audio(song_path, format='audio/wav')
            
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    # State: NO RESULTS FOUND
    elif st.session_state.search_state == 'no_results':
        st.markdown("""
            <div style='text-align: center; padding: 2rem 1rem;'>
                <h2 style='color: #FF4B4B; margin-bottom: 0.8rem; font-size: 1.5rem;'>Oops! Nothing Found</h2>
                <p style='font-size: 1rem; color: #555; line-height: 1.5;'>
                    We couldn't find any songs matching your search in our database.
                </p>
                <p style='font-size: 0.95rem; color: #777; margin-top: 0.8rem;'>
                    Try again for a better match!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Auto-scroll to results after rendering
        if st.session_state.scroll_to_results:
            st.components.v1.html("""
                <script>
                    window.parent.document.getElementById('results-section').scrollIntoView({behavior: 'smooth', block: 'start'});
                </script>
            """, height=0)
            st.session_state.scroll_to_results = False

elif st.session_state.current_page == 'all_songs':
    # ===== ALL SONGS PAGE =====
    st.markdown("<h2 style='text-align: center; color: #FF4B4B; font-size: 1.8rem; margin-bottom: 0.5rem;'>All Songs in Database</h2>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 1.5rem; font-size: 0.95rem;'>Browse our complete collection of songs!</p>", 
                unsafe_allow_html=True)
    
    # Try to get songs from Elasticsearch, fall back to file system
    all_songs = []
    if UTILS_AVAILABLE:
        try:
            all_songs = utils.get_all_songs()
        except Exception as e:
            st.warning(f"Could not retrieve songs from Elasticsearch: {str(e)}")
            all_songs = []
    
    # Fallback to file system if Elasticsearch not available or failed
    if not all_songs:
        all_songs_files = get_all_songs()
        # Convert to dict format for consistency
        all_songs = [{'song_name': f.replace('.wav', ''), 'song_file_path_name': f} for f in all_songs_files]
    
    if all_songs:
        
        for idx, song_result in enumerate(all_songs, 1):
            # Handle both dict (from Elasticsearch) and string (fallback) results
            if isinstance(song_result, dict):
                song_data = {
                    'title': song_result.get('song_name', 'Unknown'),
                    'composer': song_result.get('composer', 'Unknown'),
                    'genre': song_result.get('genre', ''),
                    'album': song_result.get('album', ''),
                    'singers': song_result.get('singers', ''),
                    'lyrics': song_result.get('lyrics', ''),
                    'lyricist': song_result.get('lyricist', ''),
                    'released_year': song_result.get('released_year', '')
                }
                song_file = song_result.get('song_file_path_name', '')
            
            # Build song card HTML with optional fields
            song_card_html = f"""
                <div class="song-card">
                    <div class="song-title">#{idx} {song_data.get('title', 'Unknown')}</div>
            """
            
            # Add description if available (from old format)
            if song_data.get('description'):
                song_card_html += f"<div class='song-info'>{song_data['description']}</div>"
            
            # Add composer
            if song_data.get('composer'):
                song_card_html += f"<div class='song-info'>Composed by {song_data['composer']}</div>"
            
            # Add lyricist if available
            if song_data.get('lyricist'):
                song_card_html += f"<div class='song-info'>Lyricist: {song_data['lyricist']}</div>"
            
            # Add singers if available
            if song_data.get('singers'):
                song_card_html += f"<div class='song-info'>Singers: {song_data['singers']}</div>"
            
            # Add genre if available
            if song_data.get('genre'):
                song_card_html += f"<div class='song-info'>Genre: {song_data['genre']}</div>"
            
            # Add album if available
            if song_data.get('album'):
                song_card_html += f"<div class='song-info'>Album: {song_data['album']}</div>"
            
            # Add year if available
            if song_data.get('released_year'):
                song_card_html += f"<div class='song-info'>Year: {song_data['released_year']}</div>"
            
            song_card_html += "</div>"
            
            st.markdown(song_card_html, unsafe_allow_html=True)
            
            # Add lyrics preview and View Lyrics button if available
            if song_data.get('lyrics'):
                lyrics_col1, lyrics_col2 = st.columns([3, 1])
                with lyrics_col1:
                    lyrics_preview = song_data['lyrics'][:100] + "..." if len(song_data['lyrics']) > 100 else song_data['lyrics']
                    st.markdown(f"<div class='lyrics-preview'>Lyrics: {lyrics_preview}</div>", unsafe_allow_html=True)
                with lyrics_col2:
                    if st.button("View Lyrics", key=f"lyrics_btn_all_{idx}", use_container_width=True):
                        show_lyrics_modal(song_data.get('title', 'Unknown'), song_data['lyrics'])
            
            # Use st.audio with lazy loading (preload="none")
            if song_file:
                song_path = f"dataset/{song_file}"
                if os.path.exists(song_path):
                    st.audio(song_path, format='audio/wav')
            
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    else:
        st.info("No songs found in the database.")

elif st.session_state.current_page == 'add_song':
    # ===== MANAGE PAGE =====
    
    # Show index statistics if available
    if UTILS_AVAILABLE:
        try:
            stats = utils.get_index_stats()
            if stats.get('exists'):
                # Build status message
                status_parts = [
                    f"📊 Index Status: **Active**",
                    f"Documents: **{stats.get('doc_count', 0)}**"
                ]
                # Only add size if available (not serverless mode)
                if 'size_readable' in stats:
                    status_parts.append(f"Size: **{stats.get('size_readable')}**")
                
                st.info(" | ".join(status_parts))
            else:
                st.warning("⚠️ Elasticsearch index does not exist. Click below to initialise and enable search.")
        except Exception as e:
            st.warning(f"Could not retrieve index stats: {str(e)}")
    else:
        st.error("⚠️ Utils module not available. Please ensure all dependencies are installed.")
    
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Elasticsearch Index Management Button
    initialize_enabled = CONFIG.get('ui', {}).get('enable_initialize_data', True) and UTILS_AVAILABLE
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("Initialize and Load Demo Data", use_container_width=True, type="primary", disabled=not initialize_enabled):
            # Create placeholders for progress display
            progress_container = st.empty()
            status_container = st.empty()
            
            try:
                # Define progress callback
                def update_progress(current, total, song_name):
                    progress_container.progress(current / total, text=f"Indexing song {current}/{total}")
                    status_container.info(f"🎵 Currently indexing: **{song_name}**")
                
                status_container.info("🔄 Initializing index and loading demo data...")
                
                # Delete existing index if it exists
                es_client = utils.get_es_client().get_client()
                index_name = utils.CONFIG['elasticsearch']['index_name']
                
                if es_client.indices.exists(index=index_name):
                    es_client.indices.delete(index=index_name)
                    logger.info(f"Deleted existing index '{index_name}'")
                
                # Create new index
                if not utils.create_song_index(index_name):
                    status_container.error("❌ Failed to create index")
                else:
                    # Load demo data with progress callback
                    successful, failed = utils.load_demo_data(
                        index_name=index_name,
                        progress_callback=update_progress
                    )
                    
                    # Clear progress displays
                    progress_container.empty()
                    
                    if successful > 0:
                        status_container.success(f"✅ Index initialized successfully! Indexed {successful} songs.")
                        if failed > 0:
                            st.warning(f"⚠️ {failed} songs failed to index.")
                    else:
                        status_container.error("❌ Failed to initialize index or load data. Check logs for details.")
                    
                    # Wait a moment before rerunning to show the final message
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                status_container.error(f"❌ Error during initialization: {str(e)}")
                logger.error(f"Initialization error: {str(e)}", exc_info=True)
    
    # Show disabled message if button is disabled
    if not initialize_enabled:
        with col2:
            st.markdown("""
                <div style='text-align: center; color: #999; font-size: 0.85rem; margin-top: 0.5rem; line-height: 1.4; margin-bottom: 1rem;'>
                    * Disabled for demo due to high compute usage. Follow the GitHub repo link to host it locally and enable it.
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: #FF4B4B; font-size: 1.8rem; margin-bottom: 0.5rem;'>Add New Song</h2>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; margin-bottom: 1rem; font-size: 0.95rem;'>Fill in the details to add a new song to the database</p>", 
                unsafe_allow_html=True)
    
    # Create a form for adding songs
    with st.form("add_song_form", clear_on_submit=True):
        # Song Name - Mandatory
        song_name = st.text_input(
            "Song Name *",
            placeholder="Enter the song name"
        )
        
        # Composer - Mandatory
        composer = st.text_input(
            "Composer *",
            placeholder="Enter the composer name"
        )
        
        # Album - Mandatory
        album = st.text_input(
            "Album *",
            placeholder="Enter the album or movie name"
        )
        
        # Released Year - Mandatory
        released_year = st.number_input(
            "Released Year *",
            min_value=1900,
            max_value=2100,
            value=None,
            placeholder="Enter the release year"
        )
        
        # Genre - Mandatory
        genre = st.text_input(
            "Genre *",
            placeholder="Enter the genre (e.g., Romance, Folk, Pop)"
        )
        
        # Lyricist - Mandatory
        lyricist = st.text_input(
            "Lyricist *",
            placeholder="Enter the lyricist name"
        )
        
        # Singers - Mandatory
        singers = st.text_input(
            "Singers *",
            placeholder="Enter singer names (comma-separated)"
        )
        
        # Lyrics - Mandatory (Multiline)
        lyrics = st.text_area(
            "Lyrics *",
            placeholder="Enter the song lyrics here...",
            height=150
        )
        
        # Song Input - Mandatory
        song_file = st.file_uploader(
            "Song File *",
            type=['wav', 'mp3']
        )
        
        # Add spacing before button
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # Submit button - check config for enabled state
        save_song_enabled = CONFIG.get('ui', {}).get('enable_save_song', True) and UTILS_AVAILABLE
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("Save Song", use_container_width=True, type="primary", disabled=not save_song_enabled)
        
        # Show disabled message if button is disabled
        if not save_song_enabled:
            with col2:
                st.markdown("""
                    <div style='text-align: center; color: #999; font-size: 0.85rem; margin-top: 0.5rem; line-height: 1.4; margin-bottom: 1rem;'>
                        * Disabled for demo due to high compute usage. Follow the GitHub repo link to host it locally and enable it.
                    </div>
                """, unsafe_allow_html=True)
        
        if submit_button:
            # Validate all mandatory fields
            if not song_name or not song_name.strip():
                st.error("❌ Song Name is required!")
            elif not composer or not composer.strip():
                st.error("❌ Composer is required!")
            elif not album or not album.strip():
                st.error("❌ Album is required!")
            elif released_year is None or released_year == 0:
                st.error("❌ Released Year is required!")
            elif not genre or not genre.strip():
                st.error("❌ Genre is required!")
            elif not lyricist or not lyricist.strip():
                st.error("❌ Lyricist is required!")
            elif not singers or not singers.strip():
                st.error("❌ Singers is required!")
            elif not lyrics or not lyrics.strip():
                st.error("❌ Lyrics is required!")
            elif not song_file:
                st.error("❌ Song File is required!")
            else:
                # Process the song
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                try:
                    status_placeholder.info("🔄 Processing song... Generating embeddings and indexing.")
                    
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(song_file.getvalue())
                        temp_audio_path = tmp.name
                    
                    logger.info(f"Saved uploaded file to: {temp_audio_path}")
                    
                    # Add the song using utils function
                    success, error_msg = utils.add_single_song(
                        song_name=song_name.strip(),
                        composer=composer.strip(),
                        album=album.strip(),
                        released_year=int(released_year),
                        genre=genre.strip(),
                        lyricist=lyricist.strip(),
                        singers=singers.strip(),
                        audio_file_path=temp_audio_path,
                        lyrics=lyrics.strip()
                    )
                    
                    # Clean up temporary file
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                        logger.debug(f"Cleaned up temporary file: {temp_audio_path}")
                    
                    if success:
                        status_placeholder.success(f"✅ Successfully added song: **{song_name}**")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        status_placeholder.error(f"❌ Failed to add song: {error_msg}")
                        
                except Exception as e:
                    status_placeholder.error(f"❌ Error adding song: {str(e)}")
                    logger.error(f"Error adding song: {str(e)}", exc_info=True)

# Footer
st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85rem; padding: 1rem 0 0.5rem 0; line-height: 1.6;'>
        Made with ❤️ for forgetful music lovers&nbsp;&nbsp; | &nbsp;&nbsp;<a href='https://github.com/mmohanram13/SingNSeek' target='_blank' style='color: #999; text-decoration: underline dotted;'>GitHub Repository</a>
        <br>
        Developed by <a href='https://www.linkedin.com/in/mohan-ram-m/' target='_blank' style='color: #999; text-decoration: underline dotted;'>Mohan Ram M</a>
    </div>
""", unsafe_allow_html=True)

# Check Elasticsearch connection after page renders (non-blocking)
# This happens at the end so the page loads first
if UTILS_AVAILABLE and st.session_state.es_connected is None:
    check_es_connection()
    # Update the status placeholder with the result
    if st.session_state.es_connected is True:
        status_placeholder.success("✅ Connected to Elasticsearch successfully!")
        # Clear the message after showing it
        time.sleep(2)
        status_placeholder.empty()
    elif st.session_state.es_connected is False:
        status_placeholder.error(f"❌ Failed to connect to Elasticsearch: {st.session_state.es_connection_error}")
    # Trigger a rerun to enable the button
    st.rerun()
