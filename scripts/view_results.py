import os
import argparse
import json
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.serving import run_simple
from LLBMA.resources.BMAassumptions import cellnames_dict, differential_group_dict_display

app = Flask(__name__)

# Global variable to store the result folder path
RESULT_FOLDER = None
# Path to the logo
LOGO_PATH = '/home/neo/Documents/neo/LL-BMA-v5/LLBMA/resources/logo_76.png'

def get_image_paths():
    """
    Scan the result folder structure and return organized image paths
    """
    result = {
        'regions': {
            'unannotated': [],
            'annotated': []
        },
        'cells': {}
    }
    
    # Get region images (unannotated)
    unannotated_dir = os.path.join(RESULT_FOLDER, 'selected_focus_regions', 'high_mag_unannotated')
    if os.path.exists(unannotated_dir):
        for file in sorted(os.listdir(unannotated_dir)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                result['regions']['unannotated'].append(file)
    
    # Get region images (annotated)
    annotated_dir = os.path.join(RESULT_FOLDER, 'selected_focus_regions', 'high_mag_annotated')
    if os.path.exists(annotated_dir):
        for file in sorted(os.listdir(annotated_dir)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                result['regions']['annotated'].append(file)
    
    # Get cell images from subdirectories
    cells_dir = os.path.join(RESULT_FOLDER, 'selected_cells')
    if os.path.exists(cells_dir):
        for subdir in sorted(os.listdir(cells_dir)):
            subdir_path = os.path.join(cells_dir, subdir)
            if os.path.isdir(subdir_path):
                cell_class = f"{subdir} - {cellnames_dict.get(subdir, 'Unknown')}"
                result['cells'][cell_class] = []
                for file in sorted(os.listdir(subdir_path)):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        result['cells'][cell_class].append(file)
    
    return result

def get_differential_counts():
    """
    Calculate differential counts based on the cell types found in the result folder
    """
    differential_counts = {
        "total": 0,
        "total_for_differential": 0,
        "regions_count": 0,
        "groups": {}
    }
    
    # Initialize all groups with zero counts
    for group_name in differential_group_dict_display.keys():
        differential_counts["groups"][group_name] = 0
    
    # Count regions
    regions_dir = os.path.join(RESULT_FOLDER, 'selected_focus_regions', 'high_mag_unannotated')
    if os.path.exists(regions_dir):
        differential_counts["regions_count"] = len([f for f in os.listdir(regions_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    
    # Count cells in each directory
    cells_dir = os.path.join(RESULT_FOLDER, 'selected_cells')
    if os.path.exists(cells_dir):
        for subdir in os.listdir(cells_dir):
            subdir_path = os.path.join(cells_dir, subdir)
            if os.path.isdir(subdir_path):
                # Count images in this cell type directory
                cell_count = len([f for f in os.listdir(subdir_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
                
                # Add to total count
                differential_counts["total"] += cell_count
                
                # Find which group this cell type belongs to
                for group_name, cell_types in differential_group_dict_display.items():
                    if subdir in cell_types:
                        differential_counts["groups"][group_name] += cell_count
                        # Add to total for differential if not "Skipped Cells & Artifacts"
                        if group_name != "Skipped Cells & Artifacts":
                            differential_counts["total_for_differential"] += cell_count
                        break
    
    # Calculate percentages
    differential_percentages = {"groups": {}}
    total_for_differential = differential_counts["total_for_differential"]
    
    for group_name, count in differential_counts["groups"].items():
        if group_name == "Skipped Cells & Artifacts":
            differential_percentages["groups"][group_name] = {
                "count": count,
                "percentage": "NA"
            }
        elif total_for_differential > 0:
            percentage = (count / total_for_differential) * 100
            differential_percentages["groups"][group_name] = {
                "count": count,
                "percentage": round(percentage, 1)
            }
        else:
            differential_percentages["groups"][group_name] = {
                "count": count,
                "percentage": 0
            }
    
    # Add total counts to the result
    differential_percentages["total"] = differential_counts["total"]
    differential_percentages["total_for_differential"] = differential_counts["total_for_differential"]
    differential_percentages["regions_count"] = differential_counts["regions_count"]
    
    return differential_percentages

@app.route('/')
def index():
    if not RESULT_FOLDER or not os.path.exists(RESULT_FOLDER):
        return "Error: Result folder not found or not set."
    
    image_paths = get_image_paths()
    differential_data = get_differential_counts()
    return render_template('index.html', 
                          image_paths=image_paths, 
                          result_folder=RESULT_FOLDER,
                          differential_data=differential_data)

@app.route('/logo')
def get_logo():
    """Serve the logo file"""
    if os.path.exists(LOGO_PATH):
        directory, filename = os.path.split(LOGO_PATH)
        return send_from_directory(directory, filename)
    return ""

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon (using the logo)"""
    if os.path.exists(LOGO_PATH):
        directory, filename = os.path.split(LOGO_PATH)
        return send_from_directory(directory, filename)
    return ""

# Route for serving images will be defined in main() function

def create_templates():
    """Create the templates directory and index.html file"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>LeukoLocator - Image Viewer</title>
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
        
        :root {
            --primary: #6741B2;
            --primary-dark: #270F7E;
            --accent: #BE98B3;
            --secondary: #7574C4;
            --light-bg: #D8D7E5;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Nunito', 'Avenir', 'Helvetica Neue', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f8fa;
            color: #333;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        
        .header {
            background-color: rgba(216, 215, 229, 0.5); /* Changed to use rgba for transparency */
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: white;
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            position: sticky; /* Make header sticky */
            top: 0; /* Stick to top */
            z-index: 100; /* Ensure header stays above other content */
        }
        
        .logo {
            width: 50px;
            height: 50px;
            margin-right: 15px;
        }
        
        .app-title {
            font-size: 24px;
            font-weight: 700;
            margin: 0;
        }
        
        .main-wrapper {
            display: flex;
            flex: 1;
        }
        
        .sidebar {
            width: 300px;
            background-color: white;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
            padding: 20px;
            overflow-y: auto;
            transition: all 0.3s ease;
            position: sticky;
            top: 80px; /* Below the header */
            height: calc(100vh - 80px);
            z-index: 90;
        }
        
        .sidebar-collapsed {
            width: 40px;
            padding: 20px 10px;
            overflow: hidden;
        }
        
        .sidebar-collapsed h2, 
        .sidebar-collapsed .summary-stats, 
        .sidebar-collapsed .differential-container {
            opacity: 0;
            visibility: hidden;
        }
        
        .sidebar-toggle {
            position: absolute;
            right: 10px;
            top: 20px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 4px;
            width: 30px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 95;
            transition: transform 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        .sidebar-collapsed .sidebar-toggle {
            transform: rotate(180deg);
            left: 5px;
        }
        
        .container {
            flex: 1;
            padding: 20px;
            transition: all 0.3s ease;
            width: calc(100% - 300px);
        }
        
        .container-expanded {
            width: calc(100% - 40px);
        }
        
        h1, h2, h3 {
            color: var(--primary-dark);
        }
        
        .image-container {
            margin-bottom: 40px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .image-gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .image-item {
            margin-bottom: 20px;
            max-width: 300px;
            cursor: pointer;
        }
        
        /* Different styles for regions vs cells */
        .region-image {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .cell-image {
            width: 96px;
            height: 96px;
            object-fit: cover; /* This prevents distortion */
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        /* Cell items should be smaller */
        .cell-item {
            margin-bottom: 20px;
            max-width: 100px; /* Give some space for the name */
        }
        
        button {
            padding: 8px 15px;
            background-color: var(--secondary);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
            font-family: 'Nunito', 'Avenir', sans-serif;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        
        button:hover {
            background-color: var(--primary);
        }
        
        .control-panel {
            background-color: var(--light-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .section-title {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--light-bg);
            padding-bottom: 10px;
        }
        
        .section-title h2 {
            margin: 0;
        }
        
        /* Loading overlay styles */
        #loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(39, 15, 126, 0.9);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .loading-text {
            margin-bottom: 20px;
            font-size: 24px;
            color: white;
            font-weight: 600;
        }
        
        .logo-container {
            margin-bottom: 30px;
        }
        
        .spinning-logo {
            width: 100px;
            height: 100px;
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-container {
            width: 70%;
            max-width: 500px;
            background-color: var(--light-bg);
            border-radius: 5px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 30px;
            width: 0%;
            background-color: var(--accent);
            text-align: center;
            line-height: 30px;
            color: white;
            font-weight: 600;
            transition: width 0.3s;
        }
        
        #main-content {
            display: none;
        }
        
        .toggle-all-btn {
            background-color: var(--primary);
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            border-radius: 4px;
            font-weight: 600;
        }
        
        .toggle-all-btn:hover {
            background-color: var(--primary-dark);
        }
        
        /* Differential count styles */
        .differential-container {
            margin-bottom: 20px;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }
        
        .differential-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .differential-item:last-child {
            border-bottom: none;
        }
        
        .differential-name {
            font-weight: 600;
        }
        
        .differential-value {
            display: flex;
            align-items: center;
        }
        
        .differential-count {
            margin-right: 10px;
            color: var(--primary-dark);
            font-weight: 600;
        }
        
        .differential-percentage {
            color: var(--secondary);
        }
        
        .bar-container {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .bar {
            height: 100%;
            background-color: var(--primary);
            border-radius: 10px;
        }
        
        .summary-stats {
            background-color: var(--light-bg);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }
        
        .summary-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
        }
        
        .summary-label {
            font-weight: 600;
        }
        
        .summary-value {
            color: var(--primary-dark);
            font-weight: 700;
        }
        
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
        }
        
        .modal-content {
            background-color: #fefefe;
            margin: 15% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 600px;
            border-radius: 8px;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
        }
        
        .path-display {
            margin: 20px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            word-break: break-all;
            font-family: monospace;
        }
        
        .copy-btn {
            background-color: var(--primary);
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        
        .copy-btn:hover {
            background-color: var(--primary-dark);
        }
        
        .copy-message {
            color: green;
            margin-top: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <!-- Loading overlay -->
    <div id="loading-overlay">
        <div class="logo-container">
            <img src="/logo" class="spinning-logo" alt="LeukoLocator Logo">
        </div>
        <div class="loading-text">Loading LeukoLocator...</div>
        <div class="progress-container">
            <div id="progress-bar" class="progress-bar">0%</div>
        </div>
    </div>

    <!-- Path display modal -->
    <div id="path-modal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>Image Path</h2>
            <div id="path-display" class="path-display"></div>
            <button id="copy-path-btn" class="copy-btn">Copy to Clipboard</button>
            <div id="copy-message" class="copy-message">Path copied to clipboard!</div>
        </div>
    </div>

    <!-- Main content (hidden initially) -->
    <div id="main-content">
        <header class="header">
            <img src="/logo" alt="LeukoLocator Logo" class="logo">
            <h1 class="app-title">LeukoLocator</h1>
        </header>
        
        <div class="main-wrapper">
            <!-- Sidebar with differential counts -->
            <div id="sidebar" class="sidebar">
                <button id="sidebar-toggle" class="sidebar-toggle">◀</button>
                <h2>Analysis Summary</h2>
                
                <!-- Summary statistics -->
                <div class="summary-stats">
                    <div class="summary-item">
                        <span class="summary-label">Total Regions:</span>
                        <span class="summary-value">{{ differential_data.regions_count }}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Cells:</span>
                        <span class="summary-value">{{ differential_data.total }}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Cells in Differential:</span>
                        <span class="summary-value">{{ differential_data.total_for_differential }}</span>
                    </div>
                </div>
                
                <h2>Differential Counts</h2>
                
                <div class="differential-container">
                    {% for group_name, data in differential_data.groups.items() %}
                    <div class="differential-item">
                        <div class="differential-name">{{ group_name|capitalize }}</div>
                        <div class="differential-value">
                            <span class="differential-count">{{ data.count }}</span>
                            <span class="differential-percentage">
                                {% if data.percentage == "Not included" %}
                                    ({{ data.percentage }})
                                {% else %}
                                    ({{ data.percentage }}%)
                                {% endif %}
                            </span>
                        </div>
                    </div>
                    {% if data.percentage != "Not included" %}
                    <div class="bar-container">
                        <div class="bar" style="width: {{ data.percentage }}%;"></div>
                    </div>
                    {% endif %}
                    {% endfor %}
                </div>
            </div>
            
            <!-- Main content container -->
            <div id="content-container" class="container">
                <h1>Slide Analysis Results</h1>
                
                <!-- Display Regions Section -->
                <div class="image-container">
                    <div class="section-title">
                        <h2>Focus Regions</h2>
                    </div>
                    
                    <div class="control-panel">
                        <h3 style="margin-top: 0;">Annotation Controls:</h3>
                        <p><strong>Keyboard Shortcut:</strong> Press 'A' key to toggle between annotated and unannotated versions of all images</p>
                        <p><strong>Individual Images:</strong> Use the toggle button below each image to switch that specific image</p>
                        <p><strong>View Path:</strong> Click on any image to view and copy its full path</p>
                        <button id="toggle-all-btn" class="toggle-all-btn">Toggle All Annotations</button>
                    </div>
                    
                    <div class="image-gallery">
                        {% for image in image_paths.regions.unannotated %}
                            {% if image in image_paths.regions.annotated %}
                            <div class="image-item" id="region-{{ loop.index }}">
                                <img src="/get_image/selected_focus_regions/high_mag_unannotated/{{ image }}" 
                                     data-unannotated="/get_image/selected_focus_regions/high_mag_unannotated/{{ image }}"
                                     data-annotated="/get_image/selected_focus_regions/high_mag_annotated/{{ image }}"
                                     data-state="unannotated"
                                     data-path="{{ result_folder }}/selected_focus_regions/high_mag_unannotated/{{ image }}"
                                     data-annotated-path="{{ result_folder }}/selected_focus_regions/high_mag_annotated/{{ image }}"
                                     class="preload-image region-image"
                                     onclick="showImagePath(this)">
                                <button onclick="toggleAnnotation('region-{{ loop.index }}')">Toggle Annotation</button>
                            </div>
                            {% endif %}
                        {% endfor %}
                    </div>
                </div>
                
                <!-- Display Cells Section -->
                {% for cell_type, images in image_paths.cells.items() %}
                <div class="image-container">
                    <div class="section-title">
                        <h2>{{ cell_type }}</h2>
                    </div>
                    <div class="image-gallery">
                        {% for image in images %}
                        <div class="image-item cell-item">
                            <img src="/get_image/selected_cells/{{ cell_type.split(' - ')[0] }}/{{ image }}" 
                                 class="preload-image cell-image"
                                 data-path="{{ result_folder }}/selected_cells/{{ cell_type.split(' - ')[0] }}/{{ image }}"
                                 onclick="showImagePath(this)">
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Get all images that need to be preloaded
            const imagesToPreload = document.querySelectorAll('.preload-image');
            const totalImages = imagesToPreload.length;
            const progressBar = document.getElementById('progress-bar');
            const loadingOverlay = document.getElementById('loading-overlay');
            const mainContent = document.getElementById('main-content');
            
            let loadedImages = 0;
            
            // No images to preload case
            if (totalImages === 0) {
                progressBar.style.width = '100%';
                progressBar.textContent = '100%';
                
                // Hide loading overlay and show content after a short delay
                setTimeout(function() {
                    loadingOverlay.style.display = 'none';
                    mainContent.style.display = 'block';
                }, 500);
                return;
            }
            
            // Preload each image
            imagesToPreload.forEach(function(img) {
                // Create a new image object to preload
                const preloadImg = new Image();
                
                preloadImg.onload = function() {
                    loadedImages++;
                    
                    // Update progress bar
                    const percentComplete = Math.round((loadedImages / totalImages) * 100);
                    progressBar.style.width = percentComplete + '%';
                    progressBar.textContent = percentComplete + '%';
                    
                    // If all images are loaded
                    if (loadedImages === totalImages) {
                        // Hide loading overlay and show content after a short delay
                        setTimeout(function() {
                            loadingOverlay.style.display = 'none';
                            mainContent.style.display = 'block';
                        }, 500);
                    }
                };
                
                preloadImg.onerror = function() {
                    loadedImages++;
                    console.error('Failed to load image:', img.src);
                    
                    // Update progress bar even on error
                    const percentComplete = Math.round((loadedImages / totalImages) * 100);
                    progressBar.style.width = percentComplete + '%';
                    progressBar.textContent = percentComplete + '%';
                    
                    // If all images are loaded (or failed)
                    if (loadedImages === totalImages) {
                        // Hide loading overlay and show content after a short delay
                        setTimeout(function() {
                            loadingOverlay.style.display = 'none';
                            mainContent.style.display = 'block';
                        }, 500);
                    }
                };
                
                // Start loading the image
                preloadImg.src = img.src;
                
                // For annotated images, also preload the annotated version
                if (img.hasAttribute('data-annotated')) {
                    const preloadAnnotated = new Image();
                    preloadAnnotated.src = img.getAttribute('data-annotated');
                }
            });
            
            // Set up modal close button
            document.querySelector('.close').addEventListener('click', function() {
                document.getElementById('path-modal').style.display = 'none';
            });
            
            // Close modal when clicking outside of it
            window.addEventListener('click', function(event) {
                const modal = document.getElementById('path-modal');
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
            
            // Set up copy button
            document.getElementById('copy-path-btn').addEventListener('click', function() {
                const pathText = document.getElementById('path-display').textContent;
                navigator.clipboard.writeText(pathText).then(function() {
                    const copyMessage = document.getElementById('copy-message');
                    copyMessage.style.display = 'block';
                    setTimeout(function() {
                        copyMessage.style.display = 'none';
                    }, 2000);
                });
            });
        });

        // Function to toggle between annotated and unannotated images
        function toggleAnnotation(itemId) {
            const imgElement = document.querySelector(`#${itemId} img`);
            const currentState = imgElement.getAttribute('data-state');
            
            if (currentState === 'unannotated') {
                imgElement.src = imgElement.getAttribute('data-annotated');
                imgElement.setAttribute('data-state', 'annotated');
            } else {
                imgElement.src = imgElement.getAttribute('data-unannotated');
                imgElement.setAttribute('data-state', 'unannotated');
            }
        }
        
        // Function to show image path in modal
        function showImagePath(imgElement) {
            event.stopPropagation(); // Prevent triggering other click events
            
            const modal = document.getElementById('path-modal');
            const pathDisplay = document.getElementById('path-display');
            
            // Get the appropriate path based on current state
            let path;
            if (imgElement.hasAttribute('data-state') && imgElement.getAttribute('data-state') === 'annotated') {
                path = imgElement.getAttribute('data-annotated-path');
            } else {
                path = imgElement.getAttribute('data-path');
            }
            
            // Set the path text and display the modal
            pathDisplay.textContent = path;
            modal.style.display = 'block';
        }
        
        // Add keyboard shortcut (A) to toggle annotation for the currently visible images
        document.addEventListener('keydown', function(event) {
            if (event.key.toLowerCase() === 'a') {
                const visibleImages = document.querySelectorAll('.image-item img[data-state]');
                visibleImages.forEach(img => {
                    const itemId = img.closest('.image-item').id;
                    toggleAnnotation(itemId);
                });
            }
        });
        
        // Toggle All button functionality
        document.getElementById('toggle-all-btn').addEventListener('click', function() {
            const visibleImages = document.querySelectorAll('.image-item img[data-state]');
            
            // Check if all images are already annotated
            let allAnnotated = true;
            visibleImages.forEach(img => {
                if (img.getAttribute('data-state') !== 'annotated') {
                    allAnnotated = false;
                }
            });
            
            // If all are annotated, turn off all annotations
            // If some or none are annotated, turn on all annotations
            visibleImages.forEach(img => {
                const itemId = img.closest('.image-item').id;
                const currentState = img.getAttribute('data-state');
                
                if (allAnnotated) {
                    // Turn off annotations if all are annotated
                    if (currentState !== 'unannotated') {
                        toggleAnnotation(itemId);
                    }
                } else {
                    // Turn on annotations if some are not annotated
                    if (currentState !== 'annotated') {
                        toggleAnnotation(itemId);
                    }
                }
            });
        });
        
        // Toggle sidebar
        document.getElementById('sidebar-toggle').addEventListener('click', function() {
            const sidebar = document.getElementById('sidebar');
            const container = document.getElementById('content-container');
            const toggleButton = document.getElementById('sidebar-toggle');
            
            if (sidebar.classList.contains('sidebar-collapsed')) {
                sidebar.classList.remove('sidebar-collapsed');
                container.classList.remove('container-expanded');
                toggleButton.textContent = '◀';
            } else {
                sidebar.classList.add('sidebar-collapsed');
                container.classList.add('container-expanded');
                toggleButton.textContent = '▶';
            }
        });
    </script>
</body>
</html>
""")

def main():
    parser = argparse.ArgumentParser(description='Flask app to display cell and region images.')
    parser.add_argument('result_folder', type=str, help='Path to the result folder')
    args = parser.parse_args()
    
    global RESULT_FOLDER
    RESULT_FOLDER = os.path.abspath(args.result_folder)
    
    if not os.path.exists(RESULT_FOLDER):
        print(f"Error: Result folder '{RESULT_FOLDER}' does not exist.")
        return
    
    # Check for required subfolders
    required_dirs = [
        os.path.join(RESULT_FOLDER, 'selected_cells'),
        os.path.join(RESULT_FOLDER, 'selected_focus_regions')
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Warning: Required directory '{directory}' does not exist.")
    
    # Create templates directory and files
    create_templates()
    
    # Fix for serving static files
    from flask import send_from_directory
    
    # Define the route for serving images
    @app.route('/get_image/<path:image_path>')
    def get_image(image_path):
        directory, filename = os.path.split(image_path)
        return send_from_directory(os.path.join(RESULT_FOLDER, directory), filename)
    
    # Run the app
    print(f"Starting server... Navigate to http://127.0.0.1:5000/ in your browser")
    app.run(debug=True, host='127.0.0.1', port=5000)

if __name__ == '__main__':
    main()