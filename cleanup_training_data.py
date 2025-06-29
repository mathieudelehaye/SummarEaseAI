#!/usr/bin/env python3
"""
Training Data Cleanup Utility
Clean up temporary Wikipedia training data files
"""

import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_training_data(confirm: bool = True):
    """Clean up temporary training data files"""
    
    # Files to clean up
    cleanup_files = [
        "music_training_sample.json",
        "music_training_sample_flat.json", 
        "enhanced_training_data.csv",
        "training_history.png"
    ]
    
    # Directories to check for temporary files
    temp_dirs = [
        "training_data",
        "tensorflow_models",
        "."
    ]
    
    # Find all files to clean
    files_to_remove = []
    total_size = 0
    
    # Check specific files
    for file_path in cleanup_files:
        path = Path(file_path)
        if path.exists():
            file_size = path.stat().st_size
            files_to_remove.append((path, file_size))
            total_size += file_size
    
    # Check for additional temporary files
    temp_patterns = ["*.tmp", "*_temp.*", "*_cache.*", "temp_*"]
    
    for temp_dir in temp_dirs:
        dir_path = Path(temp_dir)
        if dir_path.exists():
            for pattern in temp_patterns:
                for temp_file in dir_path.glob(pattern):
                    if temp_file.is_file():
                        file_size = temp_file.stat().st_size
                        files_to_remove.append((temp_file, file_size))
                        total_size += file_size
    
    if not files_to_remove:
        logger.info("✅ No temporary training data files found to clean up")
        return
    
    # Show what will be removed
    logger.info("🧹 Found temporary training data files:")
    for file_path, file_size in files_to_remove:
        logger.info(f"   📄 {file_path} ({file_size/1024:.1f} KB)")
    
    logger.info(f"💾 Total size: {total_size/1024:.1f} KB ({total_size/1024/1024:.2f} MB)")
    
    # Confirm deletion
    if confirm:
        response = input(f"\n❓ Remove {len(files_to_remove)} files? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("❌ Cleanup cancelled")
            return
    
    # Remove files
    removed_count = 0
    removed_size = 0
    
    for file_path, file_size in files_to_remove:
        try:
            file_path.unlink()
            removed_count += 1
            removed_size += file_size
            logger.info(f"   🗑️  Removed: {file_path}")
        except Exception as e:
            logger.error(f"   ❌ Failed to remove {file_path}: {e}")
    
    logger.info(f"✅ Cleanup complete: {removed_count} files removed, {removed_size/1024:.1f} KB freed")

def show_training_data_info():
    """Show information about current training data files"""
    logger.info("📊 Current Training Data Files:")
    
    # Check for training data files
    training_files = [
        "music_training_sample.json",
        "music_training_sample_flat.json",
        "enhanced_training_data.csv",
        "training_data/comprehensive_wikipedia_data.json",
        "training_data/training_summary.json",
        "tensorflow_models/bert_gpu_models/metadata.json"
    ]
    
    total_size = 0
    found_files = 0
    
    for file_path in training_files:
        path = Path(file_path)
        if path.exists():
            file_size = path.stat().st_size
            modified_time = path.stat().st_mtime
            from datetime import datetime
            mod_date = datetime.fromtimestamp(modified_time).strftime("%Y-%m-%d %H:%M")
            
            logger.info(f"   📄 {file_path}")
            logger.info(f"      Size: {file_size/1024:.1f} KB")
            logger.info(f"      Modified: {mod_date}")
            
            total_size += file_size
            found_files += 1
    
    if found_files == 0:
        logger.info("   No training data files found")
    else:
        logger.info(f"📊 Total: {found_files} files, {total_size/1024:.1f} KB ({total_size/1024/1024:.2f} MB)")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Data Cleanup Utility")
    parser.add_argument("--info", action="store_true", help="Show training data file information")
    parser.add_argument("--clean", action="store_true", help="Clean up temporary files")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if args.info:
        show_training_data_info()
    elif args.clean:
        cleanup_training_data(confirm=not args.force)
    else:
        # Interactive mode
        logger.info("🧹 Training Data Cleanup Utility")
        logger.info("=" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Show training data info")
            print("2. Clean up temporary files")
            print("3. Exit")
            
            choice = input("\nChoose option (1-3): ").strip()
            
            if choice == "1":
                show_training_data_info()
            elif choice == "2":
                cleanup_training_data()
            elif choice == "3":
                logger.info("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 