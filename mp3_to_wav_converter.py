"""
MP3 to WAV Converter
Converts MP3 files to WAV format using librosa and soundfile libraries.
Compatible with Python 3.13+
"""

import os
import sys
from pathlib import Path
import librosa
import soundfile as sf


def convert_mp3_to_wav(mp3_file_path, output_dir=None):
    """
    Convert a single MP3 file to WAV format.
    
    Args:
        mp3_file_path (str): Path to the MP3 file
        output_dir (str, optional): Directory to save the WAV file. 
                                   If None, saves in the same directory as input file.
    
    Returns:
        str: Path to the converted WAV file
    """
    try:
        # Load the MP3 file using librosa
        audio, sample_rate = librosa.load(mp3_file_path, sr=None, mono=False)
        
        # Determine output path
        mp3_path = Path(mp3_file_path)
        if output_dir:
            output_path = Path(output_dir) / f"{mp3_path.stem}.wav"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_path = mp3_path.with_suffix('.wav')
        
        # Export as WAV using soundfile
        sf.write(output_path, audio.T if audio.ndim > 1 else audio, sample_rate)
        print(f"✓ Converted: {mp3_path.name} -> {output_path.name}")
        return str(output_path)
    
    except Exception as e:
        print(f"✗ Error converting {mp3_file_path}: {str(e)}")
        return None


def convert_directory(input_dir, output_dir=None, recursive=False):
    """
    Convert all MP3 files in a directory to WAV format.
    
    Args:
        input_dir (str): Directory containing MP3 files
        output_dir (str, optional): Directory to save WAV files
        recursive (bool): Whether to search subdirectories
    
    Returns:
        tuple: (success_count, failed_count)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return 0, 0
    
    # Find all MP3 files
    if recursive:
        mp3_files = list(input_path.rglob('*.mp3'))
    else:
        mp3_files = list(input_path.glob('*.mp3'))
    
    if not mp3_files:
        print(f"No MP3 files found in '{input_dir}'")
        return 0, 0
    
    print(f"\nFound {len(mp3_files)} MP3 file(s) to convert\n")
    
    success_count = 0
    failed_count = 0
    
    for mp3_file in mp3_files:
        result = convert_mp3_to_wav(str(mp3_file), output_dir)
        if result:
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Success: {success_count} | Failed: {failed_count}")
    print(f"{'='*50}\n")
    
    return success_count, failed_count


def main():
    """Main function to handle command-line usage."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert a single file:")
        print("    python mp3_to_wav_converter.py <mp3_file> [output_directory]")
        print("\n  Convert all files in a directory:")
        print("    python mp3_to_wav_converter.py --dir <input_directory> [output_directory] [--recursive]")
        print("\nExamples:")
        print("  python mp3_to_wav_converter.py song.mp3")
        print("  python mp3_to_wav_converter.py song.mp3 ./converted")
        print("  python mp3_to_wav_converter.py --dir ./music ./converted")
        print("  python mp3_to_wav_converter.py --dir ./music ./converted --recursive")
        sys.exit(1)
    
    # Directory mode
    if sys.argv[1] == '--dir':
        if len(sys.argv) < 3:
            print("Error: Please specify input directory")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '--recursive' else None
        recursive = '--recursive' in sys.argv
        
        convert_directory(input_dir, output_dir, recursive)
    
    # Single file mode
    else:
        mp3_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(mp3_file):
            print(f"Error: File '{mp3_file}' does not exist.")
            sys.exit(1)
        
        if not mp3_file.lower().endswith('.mp3'):
            print(f"Error: File must be an MP3 file.")
            sys.exit(1)
        
        convert_mp3_to_wav(mp3_file, output_dir)


if __name__ == "__main__":
    main()
