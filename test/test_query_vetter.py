"""
Test script for the Query Vetter functionality.
Demonstrates how natural language queries are parsed.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Now import (ignore IDE warnings, this works at runtime)
from utils.query_vetter import get_query_vetter  # type: ignore

def test_query_vetter():
    """Test the query vetter with various natural language queries."""
    
    # Initialize query vetter
    print("Initializing Query Vetter...")
    vetter = get_query_vetter()
    print("✓ Query Vetter initialized\n")
    
    # Test queries
    test_queries = [
        "song with composer AiCanvas",
        "find pop songs by voiceofruthie",
        "lyrics about love and apocalypse",
        "songs with lyricist AiCanvas in R&B genre",
        "apocalypse",
        "find songs by singer Rangga Fermata",
        "best thing song",
        "lyrics with the word hollow"
    ]
    
    print("=" * 80)
    print("Testing Query Vetter")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        print("-" * 80)
        
        result = vetter.parse_query(query)
        
        print(f"   Filters: {result['filters']}")
        print(f"   Search Text: \"{result['search_text']}\"")
        print(f"   Search Type: {result['search_type']}")
        print(f"   Use Hybrid: {result['use_hybrid']}")
        print()

if __name__ == "__main__":
    test_query_vetter()
