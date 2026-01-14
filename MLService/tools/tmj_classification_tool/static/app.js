/**
 * TMJ Classification Tool - Main JavaScript
 * Handles DICOM visualization and annotation
 */

// Global state
let studyData = null;
let volumeDimensions = {
    axial: { max: 0, current: 0 },
    sagittal: { max: 0, current: 0 },
    coronal: { max: 0, current: 0 }
};
let selectedTags = {
    left: null,
    right: null
};
let availableTags = [];
let currentPatientId = null;
let currentStudyId = null;

// Canvas contexts
const canvases = {
    axial: null,
    sagittal: null,
    coronal: null
};

/**
 * Initialize viewer for a study
 */
async function initializeViewer(patientId, studyId) {
    currentPatientId = patientId;
    currentStudyId = studyId;

    // Initialize canvases
    canvases.axial = document.getElementById('axialCanvas').getContext('2d');
    canvases.sagittal = document.getElementById('sagittalCanvas').getContext('2d');
    canvases.coronal = document.getElementById('coronalCanvas').getContext('2d');

    // Load tags first
    await loadTags();

    // Then load existing annotation if present (after tags are loaded)
    await loadExistingAnnotation(studyId);

    // Load study
    await loadStudy(patientId, studyId);

    // Setup event listeners
    setupEventListeners();

    // Setup keyboard shortcuts
    setupKeyboardShortcuts();
}

/**
 * Load study data
 */
async function loadStudy(patientId, studyId) {
    try {
        const response = await fetch(`/api/study/${patientId}/${studyId}`);
        const data = await response.json();

        if (data.success) {
            studyData = data.study;
            
            // Update header info
            document.getElementById('studyInfo').textContent = 
                `${studyData.patient_name} | ${studyData.study_date} | ${studyData.num_files} файлов`;

            // Set volume dimensions
            const shape = studyData.volume_shape;
            volumeDimensions.axial.max = shape[0] - 1;
            volumeDimensions.sagittal.max = shape[2] - 1;
            volumeDimensions.coronal.max = shape[1] - 1;

            // Set slider max values
            document.getElementById('axialSlider').max = volumeDimensions.axial.max;
            document.getElementById('sagittalSlider').max = volumeDimensions.sagittal.max;
            document.getElementById('coronalSlider').max = volumeDimensions.coronal.max;

            // Set initial positions (middle slices)
            volumeDimensions.axial.current = Math.floor(volumeDimensions.axial.max / 2);
            volumeDimensions.sagittal.current = Math.floor(volumeDimensions.sagittal.max / 2);
            volumeDimensions.coronal.current = Math.floor(volumeDimensions.coronal.max / 2);

            document.getElementById('axialSlider').value = volumeDimensions.axial.current;
            document.getElementById('sagittalSlider').value = volumeDimensions.sagittal.current;
            document.getElementById('coronalSlider').value = volumeDimensions.coronal.current;

            // Load initial slices
            await loadSlice('axial', volumeDimensions.axial.current);
            await loadSlice('sagittal', volumeDimensions.sagittal.current);
            await loadSlice('coronal', volumeDimensions.coronal.current);

        } else {
            alert('Ошибка загрузки исследования: ' + data.detail);
        }
    } catch (error) {
        alert('Ошибка загрузки исследования: ' + error.message);
    }
}

/**
 * Load a specific slice
 */
async function loadSlice(plane, index) {
    const loadingDiv = document.getElementById(`${plane}Loading`);
    loadingDiv.style.display = 'block';

    try {
        const response = await fetch(`/api/slice/${currentPatientId}/${currentStudyId}/${plane}/${index}`);
        const data = await response.json();

        if (data.success) {
            // Load image
            const img = new Image();
            img.onload = () => {
                const canvas = document.getElementById(`${plane}Canvas`);
                const ctx = canvases[plane];

                // Resize canvas to match image
                canvas.width = img.width;
                canvas.height = img.height;

                // Draw image
                ctx.drawImage(img, 0, 0);

                // Update info
                const maxIndex = volumeDimensions[plane].max;
                document.getElementById(`${plane}Info`).textContent = 
                    `Срез: ${index + 1} / ${maxIndex + 1}`;

                volumeDimensions[plane].current = index;
                loadingDiv.style.display = 'none';
            };
            img.src = data.slice;
        } else {
            console.error(`Error loading ${plane} slice:`, data.detail);
            loadingDiv.textContent = 'Ошибка загрузки';
        }
    } catch (error) {
        console.error(`Error loading ${plane} slice:`, error);
        loadingDiv.textContent = 'Ошибка загрузки';
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Axial slider
    document.getElementById('axialSlider').addEventListener('input', (e) => {
        loadSlice('axial', parseInt(e.target.value));
    });

    // Sagittal slider
    document.getElementById('sagittalSlider').addEventListener('input', (e) => {
        loadSlice('sagittal', parseInt(e.target.value));
    });

    // Coronal slider
    document.getElementById('coronalSlider').addEventListener('input', (e) => {
        loadSlice('coronal', parseInt(e.target.value));
    });
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Arrow keys for navigation (without Ctrl - for slice navigation)
        if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && !e.ctrlKey) {
            e.preventDefault();
            const delta = e.key === 'ArrowRight' ? 1 : -1;
            
            // Navigate axial slices by default
            const slider = document.getElementById('axialSlider');
            const newValue = Math.max(0, Math.min(volumeDimensions.axial.max, 
                volumeDimensions.axial.current + delta));
            slider.value = newValue;
            loadSlice('axial', newValue);
        }

        // Ctrl+Arrow keys for study navigation
        if (e.ctrlKey && e.key === 'ArrowRight') {
            e.preventDefault();
            goToNext();
        }

        if (e.ctrlKey && e.key === 'ArrowLeft') {
            e.preventDefault();
            goToPrevious();
        }

        // Ctrl+S to save
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            saveAnnotation();
        }

        // Ctrl+Enter to go to next (alternative)
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            goToNext();
        }

        // Number keys for quick tag selection
        if (e.key >= '1' && e.key <= '9') {
            const tagIndex = parseInt(e.key) - 1;
            if (tagIndex < availableTags.length) {
                // If Shift pressed, select for right joint, else left
                if (e.shiftKey) {
                    selectTag('right', availableTags[tagIndex]);
                } else {
                    selectTag('left', availableTags[tagIndex]);
                }
            }
        }
    });
}

/**
 * Load available tags
 */
async function loadTags() {
    try {
        const response = await fetch('/api/tags');
        const data = await response.json();

        if (data.success) {
            availableTags = data.tags;
            renderTags();
        }
    } catch (error) {
        console.error('Error loading tags:', error);
    }
}

/**
 * Load existing annotation for current study
 */
async function loadExistingAnnotation(studyId) {
    try {
        const response = await fetch(`/api/annotation/${studyId}`);
        const data = await response.json();

        if (data.success && data.annotation) {
            const annotation = data.annotation;
            
            // Pre-select tags (tags should already be rendered at this point)
            if (annotation.left_joint_tag) {
                selectTag('left', annotation.left_joint_tag);
            }
            if (annotation.right_joint_tag) {
                selectTag('right', annotation.right_joint_tag);
            }
        }
    } catch (error) {
        console.error('Error loading existing annotation:', error);
    }
}

/**
 * Render tags in UI
 */
function renderTags() {
    const leftContainer = document.getElementById('leftJointTags');
    const rightContainer = document.getElementById('rightJointTags');

    let html = '';
    availableTags.forEach((tag, index) => {
        html += `<button class="tag-btn" onclick="selectTag('left', '${tag}')" data-tag="${tag}">
            <span class="tag-number">${index + 1}</span> ${tag}
        </button>`;
    });
    leftContainer.innerHTML = html;

    html = '';
    availableTags.forEach((tag, index) => {
        html += `<button class="tag-btn" onclick="selectTag('right', '${tag}')" data-tag="${tag}">
            <span class="tag-number">⇧${index + 1}</span> ${tag}
        </button>`;
    });
    rightContainer.innerHTML = html;
}

/**
 * Select a tag for a joint
 */
function selectTag(joint, tag) {
    selectedTags[joint] = tag;

    // Update UI
    const selectedDiv = document.getElementById(`${joint}SelectedTag`);
    selectedDiv.textContent = tag;
    selectedDiv.className = 'selected-tag active';

    // Highlight selected button
    const container = document.getElementById(`${joint}JointTags`);
    const buttons = container.querySelectorAll('.tag-btn');
    buttons.forEach(btn => {
        if (btn.dataset.tag === tag) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });

    // Enable save button if both tags selected
    updateSaveButton();
}

/**
 * Update save button state
 */
function updateSaveButton() {
    const saveBtn = document.getElementById('saveBtn');
    if (selectedTags.left && selectedTags.right) {
        saveBtn.disabled = false;
        saveBtn.classList.add('ready');
    } else {
        saveBtn.disabled = true;
        saveBtn.classList.remove('ready');
    }
}

/**
 * Add a new tag
 */
async function addNewTag() {
    const input = document.getElementById('newTagInput');
    const tagName = input.value.trim();

    if (!tagName) {
        alert('Введите название категории');
        return;
    }

    try {
        const response = await fetch('/api/add_tag', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({tag_name: tagName})
        });

        const data = await response.json();

        if (data.success) {
            input.value = '';
            await loadTags();
            alert(`Категория "${tagName}" добавлена`);
        } else {
            alert('Ошибка добавления категории: ' + data.detail);
        }
    } catch (error) {
        alert('Ошибка добавления категории: ' + error.message);
    }
}

/**
 * Save annotation
 */
async function saveAnnotation() {
    if (!selectedTags.left || !selectedTags.right) {
        alert('Выберите теги для обоих суставов');
        return;
    }

    const statusDiv = document.getElementById('saveStatus');
    statusDiv.innerHTML = '<span class="loading">⏳ Сохранение...</span>';

    try {
        const response = await fetch('/api/annotate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                patient_id: currentPatientId,
                study_id: currentStudyId,
                left_joint_tag: selectedTags.left,
                right_joint_tag: selectedTags.right
            })
        });

        const data = await response.json();

        if (data.success) {
            statusDiv.innerHTML = '<span class="success">✅ Сохранено!</span>';
            setTimeout(() => {
                statusDiv.innerHTML = '';
            }, 3000);
        } else {
            statusDiv.innerHTML = `<span class="error">❌ Ошибка: ${data.detail}</span>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<span class="error">❌ Ошибка: ${error.message}</span>`;
    }
}

/**
 * Go to next study
 */
async function goToNext() {
    const statusDiv = document.getElementById('saveStatus');
    statusDiv.innerHTML = '<span class="loading">⏳ Загрузка следующего...</span>';

    try {
        // Get next study after current one
        const nextResponse = await fetch(`/api/next_study/${currentStudyId}`);
        const nextData = await nextResponse.json();

        if (nextData.success && nextData.study) {
            // Redirect to next study
            const nextStudy = nextData.study;
            window.location.href = `/annotate/${nextStudy.patient_id}/${nextStudy.study_id}`;
        } else {
            // No more studies
            statusDiv.innerHTML = '<span class="error">❌ Нет доступных исследований</span>';
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
        }
    } catch (error) {
        statusDiv.innerHTML = `<span class="error">❌ Ошибка: ${error.message}</span>`;
    }
}

/**
 * Go to previous study
 */
async function goToPrevious() {
    const statusDiv = document.getElementById('saveStatus');
    statusDiv.innerHTML = '<span class="loading">⏳ Загрузка предыдущего...</span>';

    try {
        // Get previous study before current one
        const prevResponse = await fetch(`/api/previous_study/${currentStudyId}`);
        const prevData = await prevResponse.json();

        if (prevData.success && prevData.study) {
            // Redirect to previous study
            const prevStudy = prevData.study;
            window.location.href = `/annotate/${prevStudy.patient_id}/${prevStudy.study_id}`;
        } else {
            // No more studies
            statusDiv.innerHTML = '<span class="error">❌ Нет доступных исследований</span>';
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
        }
    } catch (error) {
        statusDiv.innerHTML = `<span class="error">❌ Ошибка: ${error.message}</span>`;
    }
}
