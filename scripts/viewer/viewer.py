import os
import io
import h5py
from flask import Flask, send_file, request, render_template_string
from LLBMA.tiling.dzsave_h5 import retrieve_tile_h5

app = Flask(__name__)


# Function to get dimensions from H5 file
def get_dimensions(h5_path):
    with h5py.File(h5_path, "r") as f:
        height = int(f["level_0_height"][()])
        width = int(f["level_0_width"][()])
    return width, height


@app.route("/tile_api", methods=["GET"])
def tile_api():
    slide = request.args.get("slide")
    level = int(request.args.get("level"))
    row = int(request.args.get("x"))  # Note: x corresponds to row
    col = int(request.args.get("y"))  # Note: y corresponds to col

    tile = retrieve_tile_h5(slide, level, row, col)  # Retrieve the JPEG image file
    img_io = io.BytesIO()
    tile.save(img_io, format="JPEG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


@app.route("/", methods=["GET"])
def index():

    root_dir = "/media/hdd3/neo/results_dir"

    subdirs = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]

    slide_h5_paths = []
    for subdir in subdirs:
        for root, dirs, files in os.walk(subdir):
            if "slide.h5" in files:
                slide_h5_paths.append(os.path.join(root, "slide.h5"))

    # Sort slide paths alphabetically
    slide_h5_paths.sort()

    slide_options = "".join(
        [f'<option value="{slide}"> {slide}</option>' for slide in slide_h5_paths]
    )

    template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title> H5 Slide Viewer</title>
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/openseadragon.min.js"></script>
        <style>
            /* General Styles */
            body {{
                font-family: 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #0a1e34;
                color: #d3e3f1;
                transition: background-color 0.5s, color 0.5s;
                display: flex;
                flex-direction: column;
                align-items: center;
                overflow-x: hidden;
            }}

            /* Header Style */
            .header {{
                text-align: center;
                padding: 20px;
                background: linear-gradient(to right, #162b44, #0a1e34);
                color: #00eaff;
                font-family: 'Orbitron', sans-serif;
                font-size: 28px;
                letter-spacing: 2px;
                border-bottom: 3px solid #0e6ba8;
                width: 100%;
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.6);
                transition: background 0.5s, box-shadow 0.5s;
            }}

            .header:hover {{
                background: linear-gradient(to right, #0e6ba8, #162b44);
                box-shadow: 0 12px 20px rgba(0, 0, 0, 0.8);
            }}

            /* Theme Toggle Button */
            .theme-toggle {{
                position: fixed;
                top: 15px;
                right: 20px;
                background: #003b66;
                border: 2px solid #00eaff;
                border-radius: 30px;
                color: #d3e3f1;
                padding: 10px 25px;
                cursor: pointer;
                font-size: 14px;
                font-family: 'Roboto', sans-serif;
                transition: background-color 0.3s, transform 0.2s, color 0.3s;
                z-index: 10;
            }}

            .theme-toggle:hover {{
                background: #00eaff;
                color: #003b66;
                transform: scale(1.05);
            }}

            /* Dropdown Style */
            select {{
                margin: 40px auto;
                display: block;
                padding: 12px;
                font-size: 16px;
                font-family: 'Roboto', sans-serif;
                border-radius: 10px;
                border: 2px solid #0e6ba8;
                background: #0a1e34;
                color: #d3e3f1;
                width: 90%;
                max-width: 500px;
                text-align: center;
                transition: border-color 0.3s, box-shadow 0.3s;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            }}

            select:hover {{
                border-color: #00eaff;
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.7);
            }}

            /* OpenSeadragon Viewer */
            #openseadragon1 {{
                width: 90%;
                max-width: 1200px;
                height: 600px;
                border: 5px solid #0e6ba8;
                border-radius: 15px;
                margin-top: 30px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.7);
                opacity: 0;
                transform: translateY(30px);
                transition: opacity 0.5s ease, transform 0.5s ease;
            }}

            #openseadragon1.show {{
                opacity: 1;
                transform: translateY(0);
            }}

            /* Footer Style */
            .footer {{
                margin-top: 20px;
                font-size: 12px;
                text-align: center;
                color: #d3e3f1;
            }}

            .footer a {{
                color: #00eaff;
                text-decoration: none;
            }}

            .footer a:hover {{
                text-decoration: underline;
            }}

            /* Light Theme */
            .light-theme {{
                background-color: #e8f4fc;
                color: #333;
            }}

            .light-theme .header {{
                background: linear-gradient(to right, #b8dff7, #e8f4fc);
                color: #003b66;
            }}

            .light-theme .theme-toggle {{
                background: #c8e8ff;
                color: #003b66;
                border-color: #003b66;
            }}

            .light-theme .theme-toggle:hover {{
                background: #003b66;
                color: #c8e8ff;
            }}

            .light-theme select {{
                background: #e8f4fc;
                color: #003b66;
                border-color: #003b66;
            }}

            .light-theme select:hover {{
                border-color: #005b8a;
            }}
        </style>
    </head>
    <body>
        <div class="header"> BMA and PBS Slide Viewer</div>
        <button class="theme-toggle" onclick="toggleTheme()"> 🔆 Switch to Light Theme</button>
        <label for="slide" style="text-align: center; display: block; font-size: 18px; margin-top: 20px;"> Select a Slide:</label>
        <select id="slide" onchange="initializeViewer()">
            {slide_options}
        </select>
        <div id="openseadragon1"></div>

        <script>
            let viewer;
            let isDarkTheme = true;

            // Toggle between dark and light themes
            function toggleTheme() {{
                isDarkTheme = !isDarkTheme;
                document.body.className = isDarkTheme ? "" : "light-theme";
                document.querySelector('.theme-toggle').textContent = isDarkTheme ? '🔆 Switch to Light Theme' : '🌙 Switch to Dark Theme';
            }}

            // Initialize OpenSeadragon viewer
            function initializeViewer() {{
                const slide = document.getElementById("slide").value;
                fetch(`/get_dimensions?slide=${{encodeURIComponent(slide)}}`)
                    .then(response => response.json())
                    .then(dimensions => {{
                        const width = dimensions.width;
                        const height = dimensions.height;

                        if (viewer) {{
                            viewer.destroy();
                        }}

                        viewer = OpenSeadragon({{
                            id: "openseadragon1",
                            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
                            tileSources: {{
                                width: width,
                                height: height,
                                tileSize: 512,
                                maxLevel: 18,
                                getTileUrl: function(level, x, y) {{
                                    return `/tile_api?slide=${{encodeURIComponent(slide)}}&level=${{level}}&x=${{x}}&y=${{y}}`;
                                }}
                            }},
                            showNavigator: true,
                            navigatorPosition: "BOTTOM_RIGHT",
                            minZoomLevel: 0.5,
                        }});

                        document.getElementById('openseadragon1').classList.add('show');
                    }})
                    .catch(error => console.error("Error fetching slide data:", error));
            }}
        </script>
    </body>
    </html>
    """

    return render_template_string(template)


@app.route("/get_dimensions", methods=["GET"])
def get_dimensions_api():
    slide = request.args.get("slide")
    width, height = get_dimensions(slide)
    return {"width": width, "height": height}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
