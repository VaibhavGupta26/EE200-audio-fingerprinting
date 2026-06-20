import os
import zipfile

def create_submission_zip(zip_name="submission.zip"):
    # Files and folders to include
    include_paths = [
        "app/app.py",
        "src/spectrogram.py",
        "src/fingerprint.py",
        "src/matcher.py",
        "src/robustness.py",
        "generate_report.py",
        "requirements.txt",
        "README.md",
        "database.pkl",
        "report/Q3_report.pdf"
    ]
    
    print(f"Creating submission zip: {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if os.path.exists(path):
                print(f"Adding: {path}")
                zipf.write(path)
            else:
                print(f"Warning: {path} not found, skipping.")
                
        # Also create empty data/queries and data/songs folders structure in the zip
        # by writing a dummy file or folder entry
        zipf.writestr("data/songs/.gitkeep", "")
        zipf.writestr("data/queries/.gitkeep", "")
        print("Adding: empty data/songs/ and data/queries/ structures")
        
    print(f"Zip archive '{zip_name}' created successfully!")
    # Print size
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"Zip size: {size_mb:.2f} MB")

if __name__ == "__main__":
    create_submission_zip()
