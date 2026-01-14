# TMJ Classification Tool

Web-based tool for annotating TMJ DICOM studies with joint condition classifications.

## Features

- 🔍 **DICOM Scanning**: Automatically scan patient directories and find all studies
- 🧹 **File Cleaning**: Remove junk files (.exe, .dll, images, etc.) from patient folders
- 🖼️ **3-Plane Visualization**: View DICOM studies in Axial, Sagittal, and Coronal planes
- 🏷️ **Flexible Tagging**: Classify left and right TMJ joints with customizable tags
- 💾 **Persistent Storage**: Annotations saved to JSON, resume anytime
- ⌨️ **Keyboard Shortcuts**: Fast navigation and tagging with hotkeys
- 🎨 **Dark Theme**: Eye-friendly interface optimized for medical imaging

## Installation

### Prerequisites

- Python 3.8+
- All dependencies from MLService/requirements.txt

### Setup

The tool uses existing dependencies from the MLService. No additional installation needed.

## Usage

### 1. Start the Server

```bash
cd MLService/tools/tmj_classification_tool
./start.sh
```

Or manually:

```bash
cd MLService/tools/tmj_classification_tool
python3 app.py
```

The server will start at http://localhost:8000

### 2. Scan Patients

1. Open http://localhost:8000 in your browser
2. Enter the path to your patients directory
3. Click "Сканировать" to find all DICOM studies

Expected directory structure:
```
patients_dir/
├── Patient Name 1/
│   ├── patient_id/
│   │   ├── date/
│   │   │   ├── series_uid/
│   │   │   │   ├── 0001.dcm
│   │   │   │   ├── 0002.dcm
│   │   │   │   └── ...
```

### 3. Clean Files (Optional)

Click "Предпросмотр (Dry Run)" to see what files would be removed, or "Очистить файлы" to actually remove junk files.

### 4. Annotate Studies

1. Click "Открыть" on any study
2. Use sliders to navigate through slices in 3 planes
3. Select tags for left and right joints
4. Click "Сохранить" or use Ctrl+S

### Keyboard Shortcuts

- **←/→**: Navigate through axial slices
- **1-9**: Select tag for left joint
- **Shift+1-9**: Select tag for right joint
- **Ctrl+S**: Save annotation
- **Ctrl+Enter**: Save and go to next study

## Output

Annotations are saved to:
```
MLService/data/tmj_classifications.json
```

Format:
```json
{
  "version": "1.0",
  "available_tags": ["normal", "osteoarthritis", "dislocation", "ankylosis"],
  "annotations": [
    {
      "patient_id": "11215",
      "study_id": "11215_20250425_2.16.840",
      "study_path": "/path/to/study",
      "left_joint_tag": "normal",
      "right_joint_tag": "osteoarthritis",
      "annotated_at": "2025-01-14T12:00:00",
      "annotated_by": "user"
    }
  ]
}
```

## Architecture

```
tmj_classification_tool/
├── app.py                      # FastAPI application
├── services/
│   ├── file_cleaner.py         # Removes junk files
│   ├── dicom_loader.py         # Scans and loads DICOM studies
│   └── annotation_manager.py   # Manages JSON annotations
├── utils/
│   └── slice_extractor.py      # Extracts slices in 3 planes
├── templates/
│   ├── index.html              # Main page
│   └── annotate.html           # Annotation page
└── static/
    ├── style.css               # Dark theme styles
    └── app.js                  # Viewer and annotation logic
```

## API Endpoints

- `POST /api/scan_patients` - Scan directory for studies
- `POST /api/clean_files` - Clean junk files
- `GET /api/studies` - Get list of studies
- `GET /api/study/{patient_id}/{study_id}` - Load specific study
- `GET /api/slice/{patient_id}/{study_id}/{plane}/{index}` - Get slice image
- `POST /api/annotate` - Save annotation
- `GET /api/annotations` - Get all annotations
- `GET /api/tags` - Get available tags
- `POST /api/add_tag` - Add new tag

## Tips

1. **Start with Dry Run**: Always preview file cleaning before actually deleting
2. **Use Keyboard Shortcuts**: Much faster than clicking
3. **Add Custom Tags**: Click "+ Добавить категорию" to add disease-specific tags
4. **Resume Anytime**: Already annotated studies are marked and can be skipped

## Troubleshooting

### DICOM Files Not Found

Make sure your directory structure matches the expected format. The tool looks for `.dcm` files in nested folders.

### Slices Not Loading

Check the browser console (F12) for errors. Make sure the study was loaded successfully.

### Port Already in Use

If port 8000 is busy, edit `app.py` and change the port:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

## License

Part of the TMJ Detection project.
