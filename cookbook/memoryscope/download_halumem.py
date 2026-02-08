"""
Download HaluMem dataset from Hugging Face
"""
import os
from datasets import load_dataset
import json

def download_halumem():
    """Download both HaluMem-Medium and HaluMem-Long datasets"""
    
    # Create directories
    base_dir = "datasets/halumem"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/medium", exist_ok=True)
    os.makedirs(f"{base_dir}/long", exist_ok=True)
    
    print("=" * 60)
    print("Downloading HaluMem Dataset from Hugging Face")
    print("=" * 60)
    
    # Download HaluMem dataset (contains both Medium and Long versions)
    print("\n[1/1] Downloading HaluMem dataset (default split)...")
    print("   - Expected: 20 users with both Medium and Long versions")
    try:
        dataset = load_dataset("IAAR-Shanghai/HaluMem")
        
        print(f"   - Dataset keys: {dataset.keys()}")
        train_dataset = dataset['train']
        print(f"   - Total entries in train split: {len(train_dataset)}")
        
        # Save the full dataset
        dataset.save_to_disk(f"{base_dir}/full")
        
        # Check structure of first entry to understand format
        if len(train_dataset) > 0:
            first_entry = train_dataset[0]
            print(f"\n   Dataset structure (first entry):")
            print(f"   - Keys: {first_entry.keys()}")
            if 'sessions' in first_entry:
                print(f"   - Number of sessions: {len(first_entry['sessions'])}")
        
        # Save summary
        stats = {
            "name": "HaluMem",
            "total_users": len(train_dataset),
            "download_status": "success",
            "split": "train",
            "saved_location": f"{base_dir}/full"
        }
        
        with open(f"{base_dir}/full/stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n   ✓ HaluMem downloaded successfully!")
        print(f"   - Saved to: {base_dir}/full")
        print(f"   - Total users: {len(train_dataset)}")
        
        medium_stats = {"download_status": "success"}
        long_stats = {"download_status": "success"}
        
    except Exception as e:
        print(f"   ✗ Error downloading HaluMem: {e}")
        import traceback
        traceback.print_exc()
        medium_stats = {"download_status": "failed", "error": str(e)}
        long_stats = {"download_status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Download Status: {medium_stats['download_status']}")
    print(f"\nDataset location: {base_dir}/full/")
    print("\nNext steps:")
    print("1. Run adapter to convert to MemoryScope format")
    print("2. Adapter will separate Medium and Long sessions based on session count")
    print("3. Run evaluation on converted datasets")
    print("=" * 60)

if __name__ == "__main__":
    download_halumem()
