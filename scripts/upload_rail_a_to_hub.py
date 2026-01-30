"""
One-time script to upload Sentinel-Rail-A to Hugging Face Hub.

Prerequisites:
1. Install: pip install huggingface_hub
2. Login: huggingface-cli login
3. Ensure model is trained and saved in models/rail_a_v3/final/

Usage:
    python scripts/upload_rail_a_to_hub.py
"""

import os

from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_DIR = "models/rail_a_v3/final"
REPO_NAME = "abdulmunimjemal/sentinel-rail-a"
REPO_TYPE = "model"


def prepare_upload():
    """Prepare files for upload."""
    print("=" * 60)
    print("PREPARING SENTINEL-RAIL-A FOR HUGGING FACE HUB")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    # List files to upload
    files = os.listdir(MODEL_DIR)
    print(f"\nFiles to upload from {MODEL_DIR}:")
    for f in files:
        size = os.path.getsize(os.path.join(MODEL_DIR, f)) / (1024**2)
        print(f"  - {f} ({size:.2f} MB)")

    # Verify critical files exist
    required_files = ["README.md", "classifier.pt", "config.json"]
    missing = [f for f in required_files if f not in files]
    if missing:
        print(f"\n⚠️  Warning: Missing recommended files: {missing}")

    return True


def upload_to_hub():
    """Upload model to Hugging Face Hub."""
    api = HfApi()

    # Create repository (if it doesn't exist)
    print(f"\n📦 Creating repository: {REPO_NAME}")
    try:
        create_repo(repo_id=REPO_NAME, repo_type=REPO_TYPE, private=False, exist_ok=True)
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")

    # Upload all files from the model directory
    print(f"\n⬆️  Uploading files to {REPO_NAME}...")
    try:
        api.upload_folder(
            folder_path=MODEL_DIR,
            repo_id=REPO_NAME,
            repo_type=REPO_TYPE,
            commit_message="Upload Sentinel-Rail-A: Prompt Injection & Jailbreak Detector",
        )
        print("✅ Upload complete!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

    # Print success message
    print("\n" + "=" * 60)
    print("🎉 MODEL SUCCESSFULLY PUBLISHED!")
    print("=" * 60)
    print("\n📍 View your model at:")
    print(f"   https://huggingface.co/{REPO_NAME}")
    print("\n📥 Users can now load it with:")
    print(f'   from_pretrained("{REPO_NAME}")')

    return True


def main():
    print("\n🚀 Sentinel-Rail-A Upload Script\n")

    # Prepare
    if not prepare_upload():
        print("❌ Preparation failed. Aborting.")
        return

    # Confirm
    response = input("\n⚠️  This will upload the model to Hugging Face Hub. Continue? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("❌ Upload cancelled.")
        return

    # Upload
    success = upload_to_hub()

    if success:
        print("\n✅ All done! Your model is now public on Hugging Face Hub.")
    else:
        print("\n❌ Upload failed. Check the error messages above.")


if __name__ == "__main__":
    main()
