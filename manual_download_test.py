#!/usr/bin/env python
"""
Manual test and demo script for the download functionality.

This script demonstrates:
1. Uploading a test artifact
2. Downloading the full package
3. Downloading specific parts from the archive
4. Verifying content integrity
"""

import requests
import zipfile
import os
import sys
import tempfile
from io import BytesIO
import json

BASE_URL = "http://localhost:8000"
AUTH_HEADER = "X-Authorization"
AUTH_TOKEN = "test-token"


def create_sample_zip():
    """Create a sample zip file for testing."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.writestr("README.md", "# Sample Model Package\n\nThis is a test model package.")
        zf.writestr("model.json", '{"name": "test-model", "version": "1.0"}')
        zf.writestr("data/train.csv", "id,feature1,feature2\n1,0.5,0.3\n2,0.7,0.2\n")
        zf.writestr("data/test.csv", "id,feature1,feature2\n3,0.6,0.4\n")
        zf.writestr("weights/model.bin", b"\x89\x50\x4e\x47" + b"\x00" * 100)  # Fake binary
    zip_buffer.seek(0)
    return zip_buffer


def test_upload():
    """Test uploading an artifact."""
    print("\n" + "=" * 60)
    print("TEST 1: Upload Artifact")
    print("=" * 60)
    
    try:
        zip_file = create_sample_zip()
        files = {"file": ("model.zip", zip_file, "application/zip")}
        response = requests.post(
            f"{BASE_URL}/upload",
            files=files
        )
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return None
        
        artifact = response.json()
        artifact_id = artifact["id"]
        print(f"✅ Upload successful!")
        print(f"   Artifact ID: {artifact_id}")
        print(f"   Filename: {artifact['filename']}")
        return artifact_id
    
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        return None


def test_download_full_package(artifact_id):
    """Test downloading the full package."""
    print("\n" + "=" * 60)
    print("TEST 2: Download Full Package")
    print("=" * 60)
    
    try:
        headers = {AUTH_HEADER: AUTH_TOKEN}
        response = requests.get(
            f"{BASE_URL}/download/{artifact_id}",
            headers=headers,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"❌ Download failed: {response.status_code}")
            print(response.text)
            return False
        
        # Save and verify the zip
        content = response.content
        zip_buffer = BytesIO(content)
        
        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            files = zf.namelist()
            print(f"✅ Download successful!")
            print(f"   Content-Type: {response.headers.get('content-type')}")
            print(f"   Content-Length: {len(content)} bytes")
            print(f"   Files in archive: {files}")
            
            # Verify specific file content
            readme = zf.read("README.md").decode()
            print(f"   README.md content: {readme[:50]}...")
        
        return True
    
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False


def test_download_specific_file(artifact_id):
    """Test downloading a specific file from the archive."""
    print("\n" + "=" * 60)
    print("TEST 3: Download Specific File (data/train.csv)")
    print("=" * 60)
    
    try:
        headers = {AUTH_HEADER: AUTH_TOKEN}
        response = requests.get(
            f"{BASE_URL}/download/{artifact_id}?part=data/train.csv",
            headers=headers,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"❌ Download failed: {response.status_code}")
            print(response.text)
            return False
        
        content = response.content.decode()
        print(f"✅ Download successful!")
        print(f"   Content-Type: {response.headers.get('content-type')}")
        print(f"   Content:\n{content}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False


def test_auth_required(artifact_id):
    """Test that authentication is required."""
    print("\n" + "=" * 60)
    print("TEST 4: Authorization Check (should fail without header)")
    print("=" * 60)
    
    try:
        # Try without auth header
        response = requests.get(f"{BASE_URL}/download/{artifact_id}")
        
        if response.status_code == 401:
            print(f"✅ Authorization correctly enforced!")
            print(f"   Error: {response.json()['detail']}")
            return True
        else:
            print(f"❌ Expected 401, got {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Error during auth test: {e}")
        return False


def test_missing_artifact():
    """Test downloading a non-existent artifact."""
    print("\n" + "=" * 60)
    print("TEST 5: Missing Artifact (should return 404)")
    print("=" * 60)
    
    try:
        headers = {AUTH_HEADER: AUTH_TOKEN}
        response = requests.get(
            f"{BASE_URL}/download/nonexistent-artifact-id",
            headers=headers
        )
        
        if response.status_code == 404:
            print(f"✅ 404 correctly returned for missing artifact!")
            print(f"   Error: {response.json()['detail']}")
            return True
        else:
            print(f"❌ Expected 404, got {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False


def list_artifacts():
    """List all artifacts."""
    print("\n" + "=" * 60)
    print("Available Artifacts")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/artifacts")
        
        if response.status_code != 200:
            print(f"❌ Failed to list artifacts: {response.status_code}")
            return
        
        data = response.json()
        print(f"Total artifacts: {data['total']}")
        for artifact in data.get("items", []):
            print(f"  - {artifact['id']}: {artifact['filename']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("Download Functionality Test Suite")
    print("🚀 " * 20)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"❌ Server not responding at {BASE_URL}")
            print("   Make sure the API is running: uvicorn api.main:app --reload")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print(f"   Error: {e}")
        print("   Make sure the API is running: uvicorn api.main:app --reload")
        sys.exit(1)
    
    # Run tests
    artifact_id = test_upload()
    if not artifact_id:
        sys.exit(1)
    
    list_artifacts()
    
    test_download_full_package(artifact_id)
    test_download_specific_file(artifact_id)
    test_auth_required(artifact_id)
    test_missing_artifact()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    print("\nQuick Start:")
    print("1. Start the API server:")
    print("   uvicorn api.main:app --reload --port 8000")
    print("\n2. Run this test script:")
    print("   python manual_download_test.py")
    print("\n3. Curl examples:")
    print(f"   # Download full package")
    print(f"   curl -H 'X-Authorization: token' http://localhost:8000/download/{{artifact_id}} -o model.zip")
    print(f"\n   # Download specific part")
    print(f"   curl -H 'X-Authorization: token' 'http://localhost:8000/download/{{artifact_id}}?part=data/train.csv'")


if __name__ == "__main__":
    main()
