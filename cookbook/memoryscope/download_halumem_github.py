"""
Download HaluMem dataset from GitHub repository
The Hugging Face dataset has schema issues, so we'll use the GitHub releases/data directly
"""
import os
import json
import requests
from tqdm import tqdm

def download_file(url: str, dest_path: str):
    """Download a file with progress bar"""
    print(f"Downloading from {url}")
    print(f"  -> {dest_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"  ✓ Downloaded successfully")

def download_halumem_github():
    """Download HaluMem dataset from GitHub"""
    
    print("=" * 60)
    print("Downloading HaluMem Dataset from GitHub")
    print("=" * 60)
    
    base_dir = "datasets/halumem"
    os.makedirs(base_dir, exist_ok=True)
    
    # GitHub raw content URLs for the dataset
    # We'll need to check the actual GitHub repo structure
    github_repo = "MemTensor/HaluMem"
    branch = "main"
    
    print(f"\nChecking GitHub repository: {github_repo}")
    print("Looking for data files...")
    
    # Try to get the repository structure
    api_url = f"https://api.github.com/repos/{github_repo}/contents/data"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        
        files = response.json()
        
        print(f"\nFound {len(files)} files in data/ directory:")
        for file_info in files:
            print(f"  - {file_info['name']} ({file_info['size']} bytes)")
        
        # Download each data file
        for file_info in files:
            if file_info['name'].endswith('.json') or file_info['name'].endswith('.jsonl'):
                download_url = file_info['download_url']
                dest_path = os.path.join(base_dir, file_info['name'])
                
                print(f"\nDownloading {file_info['name']}...")
                download_file(download_url, dest_path)
        
        # Save stats
        stats = {
            "name": "HaluMem",
            "source": "GitHub",
            "repo": github_repo,
            "files_downloaded": [f['name'] for f in files if f['name'].endswith(('.json', '.jsonl'))],
            "download_status": "success"
        }
        
        with open(f"{base_dir}/stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Download Summary")
        print("=" * 60)
        print(f"✓ Successfully downloaded {len(stats['files_downloaded'])} files")
        print(f"✓ Saved to: {base_dir}/")
        print("\nFiles:")
        for fname in stats['files_downloaded']:
            fpath = os.path.join(base_dir, fname)
            fsize = os.path.getsize(fpath) / (1024 * 1024)  # MB
            print(f"  - {fname}: {fsize:.2f} MB")
        print("=" * 60)
        
        return True
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("\n⚠ Data directory not found in GitHub repo")
            print("Trying alternative approach: cloning the repository...")
            return download_via_git_clone()
        else:
            raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_via_git_clone():
    """Clone the HaluMem repository to get the data"""
    import subprocess
    
    print("\nCloning HaluMem repository...")
    
    base_dir = "datasets/halumem"
    repo_dir = f"{base_dir}/HaluMem_repo"
    
    if os.path.exists(repo_dir):
        print(f"  Repository already exists at {repo_dir}")
        print("  Pulling latest changes...")
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
    else:
        print(f"  Cloning to {repo_dir}...")
        subprocess.run([
            "git", "clone",
            "https://github.com/MemTensor/HaluMem.git",
            repo_dir
        ], check=True)
    
    print("  ✓ Repository cloned successfully")
    
    # Check for data files
    data_dir = f"{repo_dir}/data"
    if os.path.exists(data_dir):
        print(f"\n  Found data directory: {data_dir}")
        files = os.listdir(data_dir)
        print(f"  Files: {files}")
        
        stats = {
            "name": "HaluMem",
            "source": "GitHub Clone",
            "repo_path": repo_dir,
            "data_path": data_dir,
            "files": files,
            "download_status": "success"
        }
        
        with open(f"{base_dir}/stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Download Summary")
        print("=" * 60)
        print(f"✓ Repository cloned to: {repo_dir}")
        print(f"✓ Data available at: {data_dir}")
        print("\nNext steps:")
        print("1. Check the data directory structure")
        print("2. Adapt the conversion script to read from this location")
        print("=" * 60)
        
        return True
    else:
        print(f"\n  ⚠ No data directory found in repository")
        print("  The dataset might need to be downloaded separately")
        print("  Check the repository README for instructions")
        return False

if __name__ == "__main__":
    success = download_halumem_github()
    
    if not success:
        print("\n⚠ Download failed. Please check:")
        print("1. Internet connection")
        print("2. GitHub repository structure")
        print("3. Repository README for manual download instructions")
