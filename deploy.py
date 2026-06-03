import os
from huggingface_hub import HfApi, login

def main():
    print("============================================")
    print("  Cat vs Dog -- Hugging Face Auto-Deployer  ")
    print("============================================\n")
    
    token = input("1. Enter your Hugging Face Access Token (must have 'write' permission):\n> ").strip()
    if not token:
        print("[!] Token is required! You can generate one at https://huggingface.co/settings/tokens")
        return

    repo_id = input("\n2. Enter your Space Repo ID (e.g., your_username/cat-or-dog):\n> ").strip()
    if not repo_id:
        print("[!] Repo ID is required!")
        return

    print("\n[*] Logging in...")
    try:
        login(token=token)
    except Exception as e:
        print(f"[!] Login failed: {e}")
        return

    api = HfApi()
    
    print(f"\n[*] Uploading your project to {repo_id}...")
    print("    This will skip the huge 'data' folder and virtual environments.")
    print("    This may take a minute depending on your internet connection...\n")
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=[
                "app/**",
                "model/**",
                "weights/**",
                "Dockerfile",
                "requirements.txt"
            ],
        )
        print("\n[OK] Deployment uploaded successfully!")
        print(f"     Check your live space here: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"\n[!] Upload failed: {e}")

if __name__ == "__main__":
    main()
