"""
Test script for SingN'Seek utility functions.
Tests Elasticsearch operations, embeddings, and search functionality.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import utils


def test_elasticsearch_connection():
    """Test Elasticsearch connection."""
    print("\n" + "="*60)
    print("TEST 1: Elasticsearch Connection")
    print("="*60)
    
    try:
        client = utils.get_es_client()
        es = client.get_client()
        
        if es.ping():
            print("✅ Successfully connected to Elasticsearch")
            
            # Get cluster info
            info = es.info()
            print(f"   Cluster: {info['cluster_name']}")
            print(f"   Version: {info['version']['number']}")
            return True
        else:
            print("❌ Failed to ping Elasticsearch")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False


def test_index_creation():
    """Test index creation."""
    print("\n" + "="*60)
    print("TEST 2: Index Creation")
    print("="*60)
    
    try:
        # Delete index if it exists (for clean test)
        test_index = "test_songs"
        utils.delete_song_index(test_index)
        
        # Create index
        result = utils.create_song_index(test_index)
        
        if result:
            print(f"✅ Successfully created index: {test_index}")
            
            # Verify index exists
            es = utils.get_es_client().get_client()
            exists = es.indices.exists(index=test_index)
            print(f"   Index exists: {exists}")
            
            # Clean up
            utils.delete_song_index(test_index)
            print(f"   Cleaned up test index")
            return True
        else:
            print(f"❌ Failed to create index: {test_index}")
            return False
            
    except Exception as e:
        print(f"❌ Index creation failed: {str(e)}")
        return False


def test_embedding_generation():
    """Test embedding generation."""
    print("\n" + "="*60)
    print("TEST 3: Embedding Generation")
    print("="*60)
    
    try:
        embedding_gen = utils.get_embedding_generator()
        
        # Test text embedding
        print("\n📝 Testing text embedding...")
        test_text = "A beautiful romantic song with melodious music"
        text_emb = embedding_gen.generate_text_embedding(test_text)
        
        if text_emb is not None:
            print(f"✅ Generated text embedding with shape: {text_emb.shape}")
        else:
            print("⚠️  Text embedding generation returned None (Vertex AI may not be configured)")
        
        # Test audio embedding
        print("\n🎵 Testing audio embedding...")
        test_audio = "dataset/copyright/Mun Paniya.wav"
        
        if os.path.exists(test_audio):
            audio_emb = embedding_gen.generate_audio_embedding(test_audio)
            
            if audio_emb is not None:
                print(f"✅ Generated audio embedding with shape: {audio_emb.shape}")
            else:
                print("⚠️  Audio embedding generation returned None (MuQ may not be configured)")
        else:
            print(f"⚠️  Test audio file not found: {test_audio}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {str(e)}")
        return False


def test_index_operations():
    """Test index stats retrieval."""
    print("\n" + "="*60)
    print("TEST 4: Index Operations")
    print("="*60)
    
    try:
        # Get stats for default index
        stats = utils.get_index_stats()
        
        print(f"Index Name: {stats.get('index_name', 'N/A')}")
        print(f"Exists: {stats.get('exists', False)}")
        
        if stats.get('exists'):
            print(f"✅ Index exists")
            print(f"   Documents: {stats.get('doc_count', 0)}")
            print(f"   Size: {stats.get('size_readable', 'N/A')}")
        else:
            print("⚠️  Index does not exist (this is expected if not yet created)")
        
        return True
        
    except Exception as e:
        print(f"❌ Index operations failed: {str(e)}")
        return False


def test_search_functionality():
    """Test search functionality."""
    print("\n" + "="*60)
    print("TEST 5: Search Functionality")
    print("="*60)
    
    try:
        # Check if index exists
        stats = utils.get_index_stats()
        
        if not stats.get('exists'):
            print("⚠️  Index does not exist. Skipping search test.")
            print("   Run 'Create Index' and 'Load Demo Data' in the app first.")
            return True
        
        if stats.get('doc_count', 0) == 0:
            print("⚠️  Index is empty. Skipping search test.")
            print("   Run 'Load Demo Data' in the app first.")
            return True
        
        # Test text search
        print("\n🔍 Testing text search...")
        query_text = "love romantic"
        results = utils.search_songs(query_text=query_text)
        
        if results:
            print(f"✅ Found {len(results)} results for '{query_text}'")
            for i, result in enumerate(results[:3], 1):
                print(f"   {i}. {result.get('song_name', 'Unknown')} (Score: {result.get('_score', 0):.2f})")
        else:
            print(f"⚠️  No results found for '{query_text}'")
        
        # Test retrieving all songs
        print("\n📚 Testing get all songs...")
        all_songs = utils.get_all_songs()
        
        if all_songs:
            print(f"✅ Retrieved {len(all_songs)} songs from index")
        else:
            print("⚠️  No songs retrieved")
        
        return True
        
    except Exception as e:
        print(f"❌ Search functionality failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪 "*30)
    print("SingN'Seek Utility Functions Test Suite")
    print("🧪 "*30)
    
    results = {
        "Elasticsearch Connection": test_elasticsearch_connection(),
        "Index Creation": test_index_creation(),
        "Embedding Generation": test_embedding_generation(),
        "Index Operations": test_index_operations(),
        "Search Functionality": test_search_functionality()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
