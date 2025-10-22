import streamlit as st
import os
import random
import time
import base64
from pathlib import Path
import tempfile

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="SingN'Seek",
    page_icon="images/logo.png",
    layout="centered"
)

# Import utility functions for Elasticsearch and search
try:
    import utils
    UTILS_AVAILABLE = True
    
    # Check Elasticsearch connection at startup
    try:
        es_client = utils.get_es_client()
        st.success("✅ Connected to Elasticsearch successfully!")
    except Exception as e:
        st.error(f"❌ Failed to connect to Elasticsearch: {str(e)}")
        st.info("💡 Make sure Elasticsearch is running and check your .env configuration.")
        UTILS_AVAILABLE = False
        
except ImportError:
    UTILS_AVAILABLE = False
    st.warning("⚠️ Utils module not available. Search functionality will be limited.")

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
    .audio-recorder-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem 0;
    }
    /* Center the audio recorder */
    iframe[title="audio_recorder_streamlit.audio_recorder"] {
        display: block;
        margin: 0 auto;
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

# Sample song metadata (since we're displaying randomly)
SONG_METADATA = {
    "Aadipaaru Mangatha -May Madham.wav": {
        "title": "Aadipaaru Mangatha",
        "description": "A melodious Tamil song from the movie May Madham",
        "composer": "A.R. Rahman"
    },
    "Aalangati Mazhai -Thenali.wav": {
        "title": "Aalangati Mazhai",
        "description": "A beautiful rain song from Thenali",
        "composer": "A.R. Rahman"
    },
    "Aathi Ena Nee Partha Udana.wav": {
        "title": "Aathi Ena Nee Partha Udana",
        "description": "An energetic and catchy Tamil song",
        "composer": "Yuvan Shankar Raja"
    },
    "Enna Nadanthaalum Penne.wav": {
        "title": "Enna Nadanthaalum Penne",
        "description": "A romantic melody",
        "composer": "Anirudh Ravichander"
    },
    "Kadhal Kan Kattuthe.wav": {
        "title": "Kadhal Kan Kattuthe",
        "description": "A soothing love song",
        "composer": "A.R. Rahman"
    },
    "Kadhal Rojave -Roja.wav": {
        "title": "Kadhal Rojave",
        "description": "Iconic romantic song from the movie Roja",
        "composer": "A.R. Rahman"
    },
    "Maanja Pottuthan.wav": {
        "title": "Maanja Pottuthan",
        "description": "A fun and upbeat Tamil track",
        "composer": "Yuvan Shankar Raja"
    },
    "Maargazhi Poove -May Madham.wav": {
        "title": "Maargazhi Poove",
        "description": "A beautiful melody from May Madham",
        "composer": "A.R. Rahman"
    },
    "Malargal Ketten Vaname Thanthanai-O K Kanmani.wav": {
        "title": "Malargal Ketten",
        "description": "A soul-stirring song from OK Kanmani",
        "composer": "A.R. Rahman"
    },
    "Vennilave Vennilave Vinai Thandi -Minsara Kanavu.wav": {
        "title": "Vennilave Vennilave",
        "description": "Romantic classic from Minsara Kanavu",
        "composer": "A.R. Rahman"
    },
    "What A Karavad.wav": {
        "title": "What A Karavad",
        "description": "A peppy and energetic Tamil song",
        "composer": "Anirudh Ravichander"
    },
    "Why This Kolaveri.wav": {
        "title": "Why This Kolaveri",
        "description": "Viral sensation and internet phenomenon",
        "composer": "Anirudh Ravichander"
    },
    "Yaar Petra Magano Nee.wav": {
        "title": "Yaar Petra Magano Nee",
        "description": "A heartfelt Tamil melody",
        "composer": "Ilaiyaraaja"
    }
}

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

def perform_search(query_text, audio_data=None, audio_file=None):
    """
    Perform actual search using Elasticsearch and embeddings.
    Falls back to simulated search if utils are not available.
    """
    if UTILS_AVAILABLE:
        try:
            # Save audio data to temporary file if provided
            temp_audio_path = None
            if audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_data)
                    temp_audio_path = tmp.name
            elif audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_file.getvalue())
                    temp_audio_path = tmp.name
            
            # Perform search
            results = utils.search_songs(
                query_text=query_text if query_text and query_text.strip() else None,
                query_audio_path=temp_audio_path
            )
            
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            # Update session state with results
            if results:
                st.session_state.search_results = results
                st.session_state.search_state = 'searched'
            else:
                st.session_state.search_results = []
                st.session_state.search_state = 'no_results'
                
        except Exception as e:
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
    audio_tab1, audio_tab2 = st.tabs(["Upload File", "Record Audio"])

    with audio_tab1:
        
        audio_file = st.file_uploader(
            "Upload audio file",
            type=['wav'],
            label_visibility="collapsed",
            help="Upload your audio file (Max 20 MB)",
            key="file_uploader",
            accept_multiple_files=False
        )
        
        if audio_file:
            # Check file size (20 MB = 20 * 1024 * 1024 bytes)
            max_size = 20 * 1024 * 1024
            if audio_file.size > max_size:
                st.error(f"File size ({audio_file.size / (1024 * 1024):.1f} MB) exceeds the 20 MB limit. Please upload a smaller file.")
                st.session_state.audio_file = None
            else:
                st.session_state.audio_file = audio_file
                st.success("File uploaded successfully!")
                st.audio(audio_file)

    with audio_tab2:
        # Load and encode button images
        def get_image_base64(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        
        record_btn_b64 = get_image_base64("images/record-button.png")
        stop_btn_b64 = get_image_base64("images/stop-button.png")
        
        # Native HTML5 Audio Recorder
        recorded_audio = st.components.v1.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    padding: 10px;
                    margin: 0;
                }}
                #instruction {{
                    font-size: 14px;
                    color: #555;
                    margin-bottom: 10px;
                    font-weight: 500;
                }}
                #recordButton {{
                    width: 80px;
                    height: 80px;
                    border: none;
                    background: transparent;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    padding: 0;
                }}
                #recordButton:hover {{
                    transform: scale(1.05);
                }}
                #recordButton:active {{
                    transform: scale(0.95);
                }}
                #recordButton img {{
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }}
                #recordButton.recording img {{
                    animation: pulse 1.5s infinite;
                }}
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.7; }}
                }}
                #audioPreview {{
                    margin-top: 10px;
                    display: none;
                    width: 100%;
                    max-width: 400px;
                }}
                .status-message {{
                    margin-top: 5px;
                    font-size: 13px;
                    color: #28a745;
                }}
            </style>
        </head>
        <body>
            <div id="instruction">Click to start recording</div>
            <button id="recordButton">
                <img id="buttonIcon" src="data:image/png;base64,{record_btn_b64}" alt="Record">
            </button>
            <audio id="audioPreview" controls></audio>
            <div id="statusMessage" class="status-message"></div>
            
            <script>
                let mediaRecorder;
                let audioChunks = [];
                let isRecording = false;
                
                const recordButton = document.getElementById('recordButton');
                const buttonIcon = document.getElementById('buttonIcon');
                const instruction = document.getElementById('instruction');
                const audioPreview = document.getElementById('audioPreview');
                const statusMessage = document.getElementById('statusMessage');
                
                const recordIcon = "data:image/png;base64,{record_btn_b64}";
                const stopIcon = "data:image/png;base64,{stop_btn_b64}";
                
                recordButton.addEventListener('click', async () => {{
                    if (!isRecording) {{
                        try {{
                            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                            mediaRecorder = new MediaRecorder(stream);
                            audioChunks = [];
                            
                            mediaRecorder.ondataavailable = (event) => {{
                                audioChunks.push(event.data);
                            }};
                            
                            mediaRecorder.onstop = async () => {{
                                const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                                const audioUrl = URL.createObjectURL(audioBlob);
                                audioPreview.src = audioUrl;
                                audioPreview.style.display = 'block';
                                
                                // Convert blob to base64 and send to Streamlit
                                const reader = new FileReader();
                                reader.readAsDataURL(audioBlob);
                                reader.onloadend = () => {{
                                    const base64Audio = reader.result.split(',')[1];
                                    window.parent.postMessage({{
                                        type: 'streamlit:setComponentValue',
                                        data: base64Audio
                                    }}, '*');
                                    statusMessage.textContent = 'Recording captured successfully!';
                                }};
                                
                                // Stop all tracks
                                stream.getTracks().forEach(track => track.stop());
                            }};
                            
                            mediaRecorder.start();
                            isRecording = true;
                            recordButton.classList.add('recording');
                            buttonIcon.src = stopIcon;
                            instruction.textContent = 'Click to stop recording';
                            instruction.style.color = '#de1212';
                            audioPreview.style.display = 'none';
                            statusMessage.textContent = '';
                        }} catch (err) {{
                            alert('Error accessing microphone: ' + err.message);
                        }}
                    }} else {{
                        mediaRecorder.stop();
                        isRecording = false;
                        recordButton.classList.remove('recording');
                        buttonIcon.src = recordIcon;
                        instruction.textContent = 'Click to start recording';
                        instruction.style.color = '#555';
                    }}
                }});
            </script>
        </body>
        </html>
        """, height=180)
        
        # Handle the recorded audio - check if it's a valid string (not DeltaGenerator)
        if recorded_audio and isinstance(recorded_audio, str) and recorded_audio.strip():
            try:
                # Decode base64 audio
                audio_data = base64.b64decode(recorded_audio)
                
                # Only update if it's a new recording
                if st.session_state.audio_bytes != audio_data:
                    st.session_state.audio_bytes = audio_data
                    st.success("Recording ready for search!")
                    st.audio(audio_data, format='audio/wav')
            except Exception as e:
                # Silently ignore decoding errors on initial render
                pass

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
        button_text = "Searching..." if st.session_state.search_state == 'searching' else "Find My Song!"
        button_disabled = st.session_state.search_state == 'searching'
        
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
        st.markdown("<p style='text-align: center; color: #666; margin-bottom: 1.2rem; font-size: 0.9rem;'>Here are the songs that match your search!</p>", 
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
                # Fallback for string results (old format)
                song_file = song_result
                song_data = SONG_METADATA.get(song_file, {
                    "title": song_file.replace('.wav', ''),
                    "description": "A beautiful Tamil melody",
                    "composer": "Unknown"
                })
            
            # Build song card HTML with optional fields
            song_card_html = f"""
                <div class="song-card">
                    <div class="song-title">#{idx} {song_data.get('title', 'Unknown')}</div>
            """
            
            # Add score if available
            if song_data.get('score'):
                song_card_html += f"<div class='song-info'>⭐ Relevance Score: {song_data['score']:.2f}</div>"
            
            # Add description if available (from old format)
            if song_data.get('description'):
                song_card_html += f"<div class='song-info'>📝 {song_data['description']}</div>"
            
            # Add composer
            song_card_html += f"<div class='song-info'>🎼 Composed by {song_data.get('composer', 'Unknown')}</div>"
            
            # Add singers if available
            if song_data.get('singers'):
                song_card_html += f"<div class='song-info'>🎤 Singers: {song_data['singers']}</div>"
            
            # Add genre if available
            if song_data.get('genre'):
                song_card_html += f"<div class='song-info'>🎵 Genre: {song_data['genre']}</div>"
            
            # Add album if available
            if song_data.get('album'):
                song_card_html += f"<div class='song-info'>💿 Album: {song_data['album']}</div>"
            
            # Add year if available
            if song_data.get('released_year'):
                song_card_html += f"<div class='song-info'>📅 Year: {song_data['released_year']}</div>"
            
            # Add lyrics if available
            if song_data.get('lyrics'):
                # Show first 150 characters of lyrics
                lyrics_preview = song_data['lyrics'][:150] + "..." if len(song_data['lyrics']) > 150 else song_data['lyrics']
                song_card_html += f"<div class='song-info'>📄 Lyrics: {lyrics_preview}</div>"
            
            song_card_html += "</div>"
            
            st.markdown(song_card_html, unsafe_allow_html=True)
            
            # Use st.audio with lazy loading (preload="none")
            if song_file:
                song_path = f"dataset/copyright/{song_file}"
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
            else:
                # Fallback for string results
                song_file = song_result
                song_data = SONG_METADATA.get(song_file, {
                    "title": song_file.replace('.wav', ''),
                    "description": "A beautiful Tamil melody",
                    "composer": "Unknown"
                })
            
            # Build song card HTML with optional fields
            song_card_html = f"""
                <div class="song-card">
                    <div class="song-title">#{idx} {song_data.get('title', 'Unknown')}</div>
            """
            
            # Add description if available (from old format)
            if song_data.get('description'):
                song_card_html += f"<div class='song-info'>📝 {song_data['description']}</div>"
            
            # Add composer
            if song_data.get('composer'):
                song_card_html += f"<div class='song-info'>🎼 Composed by {song_data['composer']}</div>"
            
            # Add lyricist if available
            if song_data.get('lyricist'):
                song_card_html += f"<div class='song-info'>✍️ Lyricist: {song_data['lyricist']}</div>"
            
            # Add singers if available
            if song_data.get('singers'):
                song_card_html += f"<div class='song-info'>🎤 Singers: {song_data['singers']}</div>"
            
            # Add genre if available
            if song_data.get('genre'):
                song_card_html += f"<div class='song-info'>🎵 Genre: {song_data['genre']}</div>"
            
            # Add album if available
            if song_data.get('album'):
                song_card_html += f"<div class='song-info'>💿 Album: {song_data['album']}</div>"
            
            # Add year if available
            if song_data.get('released_year'):
                song_card_html += f"<div class='song-info'>📅 Year: {song_data['released_year']}</div>"
            
            # Add lyrics if available
            if song_data.get('lyrics'):
                # Show first 100 characters of lyrics
                lyrics_preview = song_data['lyrics'][:100] + "..." if len(song_data['lyrics']) > 100 else song_data['lyrics']
                song_card_html += f"<div class='song-info'>📄 Lyrics: {lyrics_preview}</div>"
            
            song_card_html += "</div>"
            
            st.markdown(song_card_html, unsafe_allow_html=True)
            
            # Use st.audio with lazy loading (preload="none")
            if song_file:
                song_path = f"dataset/copyright/{song_file}"
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
                st.info(f"📊 Index Status: **Active** | Documents: **{stats.get('doc_count', 0)}** | Size: **{stats.get('size_readable', 'N/A')}**")
            else:
                st.warning("⚠️ Elasticsearch index does not exist. Create one to enable search.")
        except Exception as e:
            st.warning(f"Could not retrieve index stats: {str(e)}")
    else:
        st.error("⚠️ Utils module not available. Please ensure all dependencies are installed.")
    
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    # Elasticsearch Index Management Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Create Index", use_container_width=True, disabled=not UTILS_AVAILABLE):
            with st.spinner("Creating Elasticsearch index..."):
                try:
                    if utils.create_song_index():
                        st.success("✅ Elasticsearch index created successfully!")
                    else:
                        st.warning("Index already exists or could not be created.")
                except Exception as e:
                    st.error(f"❌ Error creating index: {str(e)}")
                st.rerun()
    
    with col2:
        if st.button("Load Demo Data", use_container_width=True, disabled=not UTILS_AVAILABLE):
            with st.spinner("Loading demo data into Elasticsearch... This may take several minutes."):
                try:
                    successful, failed = utils.load_demo_data()
                    if successful > 0:
                        st.success(f"✅ Successfully indexed {successful} songs!")
                        if failed > 0:
                            st.warning(f"⚠️ {failed} songs failed to index.")
                    else:
                        st.error("❌ No songs were indexed. Check if the index exists and data is available.")
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
                st.rerun()
    
    with col3:
        if st.button("Delete Index", use_container_width=True, type="secondary", disabled=not UTILS_AVAILABLE):
            with st.spinner("Deleting Elasticsearch index..."):
                try:
                    if utils.delete_song_index():
                        st.success("✅ Elasticsearch index deleted successfully!")
                    else:
                        st.warning("Index does not exist or could not be deleted.")
                except Exception as e:
                    st.error(f"❌ Error deleting index: {str(e)}")
                st.rerun()
    
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
        
        # Music Director - Mandatory
        music_director = st.text_input(
            "Music Director / Composer *",
            placeholder="Enter the music director or composer name"
        )
        
        # Genre - Optional
        genre = st.text_input(
            "Genre",
            placeholder="Enter the genre (e.g., Romance, Folk, Pop)"
        )
        
        # Album - Optional
        album = st.text_input(
            "Album",
            placeholder="Enter the album or movie name"
        )
        
        # Lyrics - Optional (Multiline)
        lyrics = st.text_area(
            "Lyrics",
            placeholder="Enter the song lyrics here...",
            height=150
        )
        
        # Song Input - Mandatory
        song_file = st.file_uploader(
            "Song File *",
            type=['wav']
        )
        
        # Add spacing before button
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("Save Song (Coming Soon)", use_container_width=True, type="primary", disabled=True)
        
        if submit_button:
            # Validate mandatory fields
            if not song_name or not song_name.strip():
                st.error("Song Name is required!")
            elif not music_director or not music_director.strip():
                st.error("Music Director / Composer is required!")
            elif not song_file:
                st.error("Song File is required!")
            else:
                # Check file size (20 MB limit)
                max_size = 20 * 1024 * 1024
                if song_file.size > max_size:
                    st.error(f"File size ({song_file.size / (1024 * 1024):.1f} MB) exceeds the 20 MB limit. Please upload a smaller file.")
                else:
                    try:
                        # Create dataset directory if it doesn't exist
                        dataset_path = Path("dataset")
                        dataset_path.mkdir(exist_ok=True)
                        
                        # Generate filename
                        file_extension = song_file.name.split('.')[-1]
                        safe_song_name = song_name.strip()
                        filename = f"{safe_song_name}.{file_extension}"
                        
                        # Save the audio file
                        file_path = dataset_path / filename
                        with open(file_path, "wb") as f:
                            f.write(song_file.getbuffer())
                        
                        # Update SONG_METADATA dictionary (in-memory only for this session)
                        # In a real app, you'd save this to a database or JSON file
                        metadata = {
                            "title": safe_song_name,
                            "description": f"A beautiful song",
                            "composer": music_director.strip()
                        }
                        
                        # Add optional fields if provided
                        if genre and genre.strip():
                            metadata["genre"] = genre.strip()
                        if album and album.strip():
                            metadata["album"] = album.strip()
                        if lyrics and lyrics.strip():
                            metadata["lyrics"] = lyrics.strip()
                        
                        SONG_METADATA[filename] = metadata
                        
                        st.success(f"✅ Song '{song_name}' has been added successfully!")
                        st.info("Navigate to 'All Songs' to see your newly added song.")
                        
                    except Exception as e:
                        st.error(f"Error saving song: {str(e)}")

# Footer
st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85rem; padding: 1rem 0 0.5rem 0;'>
        Made with ❤️ for forgetful music lovers&nbsp;&nbsp; | &nbsp;&nbsp;Developed by Mohan Ram M | 
        <a href='https://github.com/mmohanram13' target='_blank' style='color: #333; text-decoration: none;'>
            GitHub
        </a> | 
        <a href='https://www.linkedin.com/in/mohan-ram-m/' target='_blank' style='color: #0A66C2; text-decoration: none;'>
            LinkedIn
        </a>
    </div>
""", unsafe_allow_html=True)
