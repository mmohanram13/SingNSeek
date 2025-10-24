"""
Query vetter using Vertex AI Gemini 2.5 Flash Lite for natural language query parsing.
Converts natural language queries into structured Elasticsearch filters.
"""

import os
import json
from typing import Dict, Optional, Any
from pathlib import Path
import yaml

from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv

# Configure logging
from .logging_config import get_logger

logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)


class QueryVetter:
    """
    Uses Vertex AI Gemini 2.5 Flash Lite to parse natural language queries
    and extract structured search parameters for Elasticsearch.
    """
    
    def __init__(self):
        """Initialize the Query Vetter with Vertex AI Gemini."""
        self.model = None
        self._initialize_vertex_ai()
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI with credentials and configuration."""
        try:
            # Get configuration
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            location = os.getenv('VERTEX_AI_REGION', 'us-central1')
            
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT is required in .env file")
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=location)
            
            # Initialize Gemini 2.5 Flash Lite model
            model_name = CONFIG['vertex_ai'].get('gemini_model', 'gemini-2.5-flash-lite')
            self.model = GenerativeModel(model_name)
            
            logger.info(f"Initialized Query Vetter with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Query Vetter: {e}")
            raise
    
    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """
        Parse natural language query and extract structured search parameters.
        
        Args:
            user_query: Natural language query from user
        
        Returns:
            Dictionary with structured query parameters:
            {
                "filters": {
                    "composer": str,
                    "singers": str,
                    "lyricist": str,
                    "genre": str,
                    "album": str,
                    "song_name": str
                },
                "search_text": str,  # Remaining text for general search
                "search_type": str,  # "lyrics", "metadata", "general"
                "use_hybrid": bool   # True if lyrics search
            }
        """
        if not user_query or not user_query.strip():
            return {
                "filters": {},
                "search_text": "",
                "search_type": "general",
                "use_hybrid": False
            }
        
        try:
            # Construct prompt for Gemini
            prompt = self._build_parsing_prompt(user_query)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent parsing
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            )
            
            # Parse the response
            result = self._parse_response(response.text, user_query)
            
            logger.info(f"Parsed query: {user_query}")
            logger.info(f"Result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse query with Gemini: {e}")
            # Fallback: treat entire query as general search text
            return {
                "filters": {},
                "search_text": user_query,
                "search_type": "general",
                "use_hybrid": False
            }
    
    def _build_parsing_prompt(self, user_query: str) -> str:
        """
        Build the prompt for Gemini to parse the query.
        
        Args:
            user_query: User's natural language query
        
        Returns:
            Structured prompt for Gemini
        """
        prompt = f"""You are a query parser for a music search system. Parse the following natural language query and extract structured information.

Available fields:
- composer: The person who composed the music
- singers: The person(s) who sang the song (also known as artist, vocalist, performer)
- lyricist: The person who wrote the lyrics
- genre: The musical genre (e.g., Pop, Rock, Jazz, etc.)
- album: The album name
- song_name: The name/title of the song

User Query: "{user_query}"

Instructions:
1. Identify any specific field mentions (composer, singer/artist, lyricist, genre, album, song name)
2. Extract the value for each identified field
3. Determine if the query is about lyrics (words/content of the song) vs metadata (song info)
4. If lyrics are mentioned explicitly, mark as lyrics search, otherwise mark as metadata or general search
5. Any remaining text not tied to specific fields should be treated as general search text

Return your response in this exact JSON format (no markdown, just raw JSON):
{{
    "filters": {{
        "composer": "value or null",
        "singers": "value or null",
        "lyricist": "value or null",
        "genre": "value or null",
        "album": "value or null",
        "song_name": "value or null"
    }},
    "search_text": "remaining text for general search",
    "search_type": "lyrics|metadata|general",
    "use_hybrid": true|false
}}

Rules:
- Set use_hybrid to true ONLY if search_type is "lyrics"
- If no specific fields are found, put all text in search_text
- Remove null values from filters
- Be case-insensitive but preserve the original case of values
- Normalize field references (e.g., "artist" → "singers", "song" → "song_name")

Example 1:
Query: "song with composer AiCanvas"
Response: {{"filters": {{"composer": "AiCanvas"}}, "search_text": "", "search_type": "metadata", "use_hybrid": false}}

Example 2:
Query: "find lyrics about love"
Response: {{"filters": {{}}, "search_text": "love", "search_type": "lyrics", "use_hybrid": true}}

Example 3:
Query: "pop song by voiceofruthie about apocalypse"
Response: {{"filters": {{"genre": "Pop", "singers": "voiceofruthie"}}, "search_text": "apocalypse", "search_type": "general", "use_hybrid": false}}

Now parse the user query above and return only the JSON response.
"""
        return prompt
    
    def _parse_response(self, response_text: str, original_query: str) -> Dict[str, Any]:
        """
        Parse Gemini's response and convert to structured format.
        
        Args:
            response_text: Response from Gemini
            original_query: Original user query (fallback)
        
        Returns:
            Structured query parameters
        """
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Clean up filters (remove null/empty values)
            filters = {}
            if "filters" in parsed and parsed["filters"]:
                for key, value in parsed["filters"].items():
                    if value and value.lower() not in ["null", "none", ""]:
                        filters[key] = value
            
            # Build result
            result = {
                "filters": filters,
                "search_text": parsed.get("search_text", "").strip(),
                "search_type": parsed.get("search_type", "general"),
                "use_hybrid": parsed.get("use_hybrid", False)
            }
            
            # Validation: if use_hybrid is true, search_type should be lyrics
            if result["use_hybrid"] and result["search_type"] != "lyrics":
                result["search_type"] = "lyrics"
            
            # If no filters and no search_text, use original query as search_text
            if not result["filters"] and not result["search_text"]:
                result["search_text"] = original_query
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            # Fallback
            return {
                "filters": {},
                "search_text": original_query,
                "search_type": "general",
                "use_hybrid": False
            }
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {e}")
            # Fallback
            return {
                "filters": {},
                "search_text": original_query,
                "search_type": "general",
                "use_hybrid": False
            }


# Singleton instance
_query_vetter = None

def get_query_vetter() -> QueryVetter:
    """Get or create the singleton QueryVetter instance."""
    global _query_vetter
    if _query_vetter is None:
        _query_vetter = QueryVetter()
    return _query_vetter
