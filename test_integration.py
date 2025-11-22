import requests
import time
import sys
import os

# Configuration
BACKEND_URL = "http://localhost:8080"
DICOM_PATH = "MLService/data/Baranova A/00000172/20190312/2.16.840.114421.82596.9605717572.9637253572/0551.dcm"

def main():
    print(f"🚀 Starting Integration Test")
    print(f"   Backend: {BACKEND_URL}")
    print(f"   DICOM: {DICOM_PATH}")
    
    if not os.path.exists(DICOM_PATH):
        print(f"❌ DICOM file not found at {DICOM_PATH}")
        sys.exit(1)

    # 1. Upload File
    print("\n📤 Uploading DICOM...")
    url = f"{BACKEND_URL}/api/dicom/upload"
    
    try:
        with open(DICOM_PATH, 'rb') as f:
            files = {'data': ('test.dcm', f, 'application/dicom')}
            # Note: Vapor multipart might expect 'filename' field or inside Content-Disposition
            # Let's try standard requests multipart
            # Vapor Controller expects: struct FileUpload: Content { var filename: String; var data: ByteBuffer }
            # Usually this maps to multipart form fields.
            # Let's check how FileUpload is decoded. usually req.content.decode(FileUpload.self) expects JSON or Multipart.
            # If Multipart, it looks for fields 'filename' and 'data'.
            
            payload = {
                'filename': 'test.dcm'
            }
            # 'data' is the file content.
            # Requests handles this if we pass 'data' as file? 
            # Or separate fields.
            
            # Correct way for Vapor Content decode of struct with ByteBuffer:
            # It expects multipart with parts named 'filename' and 'data'.
            
            files = {
                'data': ('test.dcm', f, 'application/octet-stream')
            }
            data = {
                'filename': 'test.dcm'
            }
            
            response = requests.post(url, files=files, data=data)
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
        
    if response.status_code != 200:
        print(f"❌ Upload failed with status {response.status_code}")
        print(response.text)
        sys.exit(1)
        
    result = response.json()
    task_id = result['taskId']
    print(f"✅ Upload successful! Task ID: {task_id}")
    
    # 2. Poll Status
    print("\n🔄 Polling status...")
    status_url = f"{BACKEND_URL}/api/analysis/{task_id}/status"
    
    max_retries = 30
    for i in range(max_retries):
        try:
            resp = requests.get(status_url)
            status_data = resp.json()
            status = status_data['status']
            print(f"   Attempt {i+1}: {status}")
            
            if status == 'completed':
                print("✅ Task completed!")
                break
            elif status == 'failed':
                print(f"❌ Task failed: {status_data.get('errorMessage')}")
                sys.exit(1)
                
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Polling error: {e}")
            time.sleep(1)
            
    else:
        print("❌ Timeout waiting for task completion")
        sys.exit(1)
        
    # 3. Get Result
    print("\n📥 Fetching results...")
    result_url = f"{BACKEND_URL}/api/analysis/{task_id}"
    resp = requests.get(result_url)
    data = resp.json()
    
    print("✅ Result received:")
    # Print summary
    print(f"   Diagnosis: {data.get('diagnosis')}")
    
    # Check if slices/masks present (they are JSON strings in the response structure?)
    # Backend AnalysisResponse defines them as String?.
    # MLServiceClient encodes them to String.
    
    if data.get('masksData'):
        print("   Masks Data: Present")
    else:
        print("   Masks Data: Missing")
        
    print("\n🎉 Integration Test Passed!")

if __name__ == "__main__":
    main()

