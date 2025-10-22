#!/usr/bin/env python3
"""
Environment checker for SingN'Seek
Verifies that all dependencies and configurations are properly set up
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (3.9+ required)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'streamlit': 'Streamlit',
        'elasticsearch': 'Elasticsearch',
        'torch': 'PyTorch',
        'librosa': 'Librosa',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'yaml': 'PyYAML (yaml)',
        'dotenv': 'python-dotenv (dotenv)'
    }
    
    missing = []
    installed = []
    
    for module, name in required.items():
        try:
            __import__(module)
            installed.append(name)
        except ImportError:
            missing.append(name)
    
    for pkg in installed:
        print(f"✅ {pkg}")
    
    for pkg in missing:
        print(f"❌ {pkg} (not installed)")
    
    return len(missing) == 0


def check_optional_dependencies():
    """Check optional dependencies."""
    optional = {
        'google.cloud.aiplatform': 'Google Cloud AI Platform',
        'muq': 'MuQ Audio Model'
    }
    
    print("\n📦 Optional Dependencies:")
    
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} (not installed - functionality will be limited)")


def check_config_files():
    """Check if configuration files exist."""
    configs = {
        'config/elastic_config.yaml': 'Elasticsearch configuration',
        '.env': 'Environment variables (optional for Vertex AI)'
    }
    
    all_exist = True
    
    for path, description in configs.items():
        if os.path.exists(path):
            print(f"✅ {description} ({path})")
        else:
            print(f"⚠️  {description} ({path}) - not found")
            if path == 'config/elastic_config.yaml':
                all_exist = False
    
    return all_exist


def check_dataset():
    """Check if dataset exists."""
    dataset_path = Path("dataset/copyright")
    csv_path = dataset_path / "dataset_meta.csv"
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset directory not found: {dataset_path}")
        return False
    
    if not csv_path.exists():
        print(f"⚠️  Dataset CSV not found: {csv_path}")
        return False
    
    # Count audio files
    audio_files = list(dataset_path.glob("*.wav"))
    
    print(f"✅ Dataset directory exists")
    print(f"✅ Dataset CSV exists")
    print(f"ℹ️  Found {len(audio_files)} audio files")
    
    return True


def check_elasticsearch():
    """Check Elasticsearch connection."""
    try:
        import utils
        
        es_client = utils.get_es_client()
        if es_client.get_client().ping():
            print("✅ Elasticsearch connection successful")
            
            # Check index
            stats = utils.get_index_stats()
            if stats.get('exists'):
                print(f"✅ Index exists with {stats.get('doc_count', 0)} documents")
            else:
                print("⚠️  Index not created yet (use 'Manage' tab to create)")
            
            return True
        else:
            print("❌ Cannot connect to Elasticsearch")
            return False
            
    except Exception as e:
        print(f"❌ Elasticsearch error: {str(e)}")
        return False


def check_vertex_ai():
    """Check Vertex AI configuration."""
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not project_id:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set in .env")
        return False
    
    if not credentials:
        print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set in .env")
        return False
    
    if not os.path.exists(credentials):
        print(f"⚠️  Credentials file not found: {credentials}")
        return False
    
    print(f"✅ Vertex AI configured (Project: {project_id})")
    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("SingN'Seek Environment Checker".center(60))
    print("=" * 60)
    
    results = {}
    
    # Python version
    print("\n🐍 Python Version:")
    results['Python'] = check_python_version()
    
    # Required dependencies
    print("\n📦 Required Dependencies:")
    results['Dependencies'] = check_dependencies()
    
    # Optional dependencies
    check_optional_dependencies()
    
    # Configuration files
    print("\n⚙️  Configuration Files:")
    results['Config'] = check_config_files()
    
    # Dataset
    print("\n📁 Dataset:")
    results['Dataset'] = check_dataset()
    
    # Elasticsearch
    print("\n🔍 Elasticsearch:")
    results['Elasticsearch'] = check_elasticsearch()
    
    # Vertex AI
    print("\n🤖 Vertex AI:")
    results['Vertex AI'] = check_vertex_ai()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary".center(60))
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}")
    
    print("=" * 60)
    print(f"Status: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to use SingN'Seek.")
        print("\nRun: streamlit run main.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")
        print("\nPlease fix the issues above before running the app.")
        print("For help, see QUICKSTART.md or README_NEW.md")
        return 1


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    sys.exit(main())
