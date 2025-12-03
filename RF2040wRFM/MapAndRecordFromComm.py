# Andrew Wineinger
# -------------------------------------------------
# This code takes the received data from the Comm Terminal
# and saves the coordinates in a CSV. The code then takes the
# data in the CSV and maps it using open source mapping software.
# The data is inserted into a hex grid for territorial averages.
# The data format is: 
# "GPRMC,225446,A,3025.22,N,8419.05,W,000.5,054.7,191194,020.3,E*68"
# The RP2040 must have the Receiver firmware flashed.
# -------------------------------------------------

import serial
import csv
import os
import pandas as pd
import folium
import h3
from branca.colormap import linear
import time
import datetime
import traceback
import webbrowser
import re

# Serial Port Configuration
ser = serial.Serial('COM5', 115200)
csv_file = "led_state_log.csv"
map_folder = "map_history"  # Folder to store all map files
map_file_base = "hex_signal_map"  # Base name for map files
h3_resolution = 13  # Adjust resolution (lower = bigger hexes) 10 => 400(ft/hex) diameter, 13 => 30(ft/hex)diameter
update_interval = 2  # How often to update the map (seconds)
zoom_level = 15  # Higher number = more zoomed in
auto_refresh_interval = 5  # Auto-refresh interval in seconds

# Signal strength normalization parameters
min_raw_signal = 0.0  # Expected minimum value in raw data
max_raw_signal = 100.0  # Expected maximum value in raw data 
# These can be adjusted based on your actual signal range

# Debug mode
DEBUG = True

# Create map folder if it doesn't exist
if not os.path.exists(map_folder):
    os.makedirs(map_folder)
    print(f"Created map history folder: {map_folder}")

# Check if the CSV file exists
file_exists = os.path.exists(csv_file)

# Create CSV if it doesn't exist
if not file_exists:
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time_Stamp", "GPS_X", "GPS_Y", "Signal_Strength"])

def normalize_signal_strength(raw_signal):
    """
    Normalize signal strength to a value between 1 and 100
    """
    try:
        # Convert to float if it's not already
        raw_signal = float(raw_signal)
        
        # Basic normalization with clipping
        normalized = ((raw_signal - min_raw_signal) / (max_raw_signal - min_raw_signal)) * 99 + 1
        
        # Ensure the value is between 1 and 100
        normalized = max(1, min(100, normalized))
        
        return normalized
    except Exception as e:
        if DEBUG:
            print(f"Error normalizing signal strength: {e}")
        # Return a default value in case of error
        return 1.0

def convert_nmea_to_decimal_degrees(nmea_coord, direction):
    """
    Convert NMEA coordinate format (ddmm.mmmm) to decimal degrees
    
    Example:
    - "4916.45" with "N" becomes +49.2742 (49° 16.45' North)
    - "12311.12" with "W" becomes -123.1853 (123° 11.12' West)
    """
    try:
        if not nmea_coord or not direction:
            return None
            
        # Split the degrees and minutes parts
        nmea_coord = float(nmea_coord)
        degrees = int(nmea_coord / 100)
        minutes = nmea_coord - (degrees * 100)
        
        # Convert to decimal degrees
        decimal_degrees = degrees + (minutes / 60)
        
        # Apply negative value for South or West
        if direction in ['S', 'W']:
            decimal_degrees = -decimal_degrees
            
        return decimal_degrees
    except Exception as e:
        if DEBUG:
            print(f"Error converting NMEA coordinate: {e}")
            print(f"Input: {nmea_coord}, {direction}")
        return None

def parse_gprmc(gprmc_string):
    """
    Parse GPRMC NMEA sentence and extract position information
    
    Format: GPRMC,time,status,lat,lat_dir,lon,lon_dir,speed,course,date,mag_var,mag_var_dir,mode*checksum
    Example: GPRMC,225446,A,4916.45,N,12311.12,W,000.5,054.7,191194,020.3,E*68
    """
    try:
        if not gprmc_string.startswith('$GPRMC'):
            # Add $ if missing from start
            if gprmc_string.startswith('GPRMC'):
                gprmc_string = '$' + gprmc_string
            else:
                return None
        
        # Split the sentence into parts
        parts = gprmc_string.split(',')
        
        # Check if we have enough parts and the data is valid
        if len(parts) < 7:
            if DEBUG:
                print(f"Invalid GPRMC format (not enough parts): {gprmc_string}")
            return None
            
        # Check if the data is valid (A=Active, V=Void)
        if parts[2] != 'A':
            if DEBUG:
                print(f"GPS data not valid (status = {parts[2]})")
            return None
        
        # Extract time information (format: hhmmss)
        time_str = parts[1]
        if time_str and len(time_str) >= 6:
            hour = time_str[0:2]
            minute = time_str[2:4]
            second = time_str[4:6]
            time_formatted = f"{hour}:{minute}:{second}"
        else:
            time_formatted = datetime.datetime.now().strftime("%H:%M:%S")
            
        # Extract date information if available (format: ddmmyy)
        date_str = parts[9] if len(parts) > 9 and parts[9] else ''
        if date_str and len(date_str) >= 6: 
            day = date_str[0:2]
            month = date_str[2:4]
            year = '20' + date_str[4:6]  # Assuming 20xx for the year
            date_formatted = f"{year}-{month}-{day}"
            timestamp = f"{date_formatted} {time_formatted}"
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Extract and convert latitude and longitude
        latitude = parts[3]
        lat_dir = parts[4]
        longitude = parts[5]
        lon_dir = parts[6]
        
        lat_decimal = convert_nmea_to_decimal_degrees(latitude, lat_dir)
        lon_decimal = convert_nmea_to_decimal_degrees(longitude, lon_dir)
        
        if lat_decimal is None or lon_decimal is None:
            if DEBUG:
                print(f"Failed to convert coordinates: {latitude}{lat_dir}, {longitude}{lon_dir}")
            return None
        
        # Extract signal strength from magnetic variation field (index 10)
        # Format is typically something like "020.3,E*68"
        signal_strength = 1.0  # Default value
        if len(parts) > 10 and parts[10]:
            try:
                # Extract just the numeric part before any letter or symbol
                mag_var = parts[10]
                
                # If it contains letters or special characters, extract only the number
                signal_match = re.match(r'([0-9.]+)', mag_var)
                if signal_match:
                    raw_signal = float(signal_match.group(1))
                    # Normalize the signal strength to 1-100 range
                    signal_strength = normalize_signal_strength(raw_signal)
                else:
                    # Try direct conversion if it's just a number
                    raw_signal = float(mag_var)
                    signal_strength = normalize_signal_strength(raw_signal)
            except ValueError:
                if DEBUG:
                    print(f"Could not parse signal strength from: {parts[10]}")
        
        # Return the parsed data
        return [timestamp, lat_decimal, lon_decimal, signal_strength]
        
    except Exception as e:
        if DEBUG:
            print(f"Error parsing GPRMC data: {e}")
            print(f"Input: {gprmc_string}")
            print(traceback.format_exc())
        return None

def lat_lon_to_hex(lat, lon, resolution=h3_resolution):
    """ Converts latitude & longitude to an H3 hex cell index """
    try:
        # Convert string coordinates to float if needed
        lat = float(lat)
        lon = float(lon)
        # FIXED: Ensure correct order for H3 functions (lat, lon)
        return h3.latlng_to_cell(lat, lon, resolution)
    except Exception as e:
        print(f"Error in lat_lon_to_hex: {e}")
        print(f"Input values - lat: {lat} ({type(lat)}), lon: {lon} ({type(lon)})")
        return None

def get_hex_boundaries(hex_id):
    """ Gets the boundary coordinates of a hex cell for mapping """
    try:
        # FIXED: H3 boundaries are returned as lat,lon which is correct for folium
        boundary = h3.cell_to_boundary(hex_id)  # Returns list of (lat, lon) tuples
        return boundary  # Already in the correct format for folium
    except Exception as e:
        print(f"Error in get_hex_boundaries: {e}")
        print(f"Input hex_id: {hex_id}")
        return []

def insert_auto_refresh_code(html_content, refresh_interval=auto_refresh_interval):
    head_end = "</head>"
    auto_refresh_script = f"""
    <script>
        // Set auto-refresh interval (in milliseconds)
        var autoRefreshInterval = {refresh_interval} * 1000;
        var autoRefreshTimer;
        var autoRefreshEnabled = true;
        
        function startAutoRefresh() {{
            autoRefreshTimer = setTimeout(function() {{
                window.location.reload();
            }}, autoRefreshInterval);
        }}
        
        // Start the timer when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            startAutoRefresh();
        }});
        
        // Toggle function for pause/resume
        function toggleAutoRefresh() {{
            var btnElement = document.getElementById("pauseBtn");
            if (autoRefreshEnabled) {{
                clearTimeout(autoRefreshTimer);
                autoRefreshEnabled = false;
                btnElement.innerText = "Resume Auto-Refresh";
                btnElement.style.backgroundColor = "#f8d7da";
            }} else {{
                startAutoRefresh();
                autoRefreshEnabled = true;
                btnElement.innerText = "Pause Auto-Refresh";
                btnElement.style.backgroundColor = "#d4edda";
            }}
        }}
    </script>
    """
    if head_end in html_content:
        html_content = html_content.replace(head_end, auto_refresh_script + head_end)
    

    body_end = "</body>"
    pause_button = """
    <button id="pauseBtn" onclick="toggleAutoRefresh()" 
        style="position: fixed; bottom: 40px; left: 10px; z-index: 1000; 
        background-color: #d4edda; padding: 10px; border-radius: 5px; 
        border: 1px solid #c3e6cb; font-weight: bold; cursor: pointer;">
        Pause Auto-Refresh
    </button>
    """
    if body_end in html_content:
        html_content = html_content.replace(body_end, pause_button + body_end)
    
    return html_content

def enhance_hex_visualization(html_content):
    """Add enhanced styling for hex grid visualization"""
    body_end = "</body>"
    
    enhanced_style = """
    <style>
        /* Enhance hex grid visualization */
        .leaflet-interactive {
            transition: fill-opacity 0.3s ease, stroke-width 0.3s ease;
        }
        .leaflet-interactive:hover {
            fill-opacity: 0.9 !important;
            stroke-width: 3px !important;
        }
        
        /* Pulse animation for latest point */
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        .pulse-marker {
            animation: pulse 1.5s infinite;
        }
        
        /* Style for the marker icon */
        .leaflet-marker-icon {
            transition: transform 0.3s ease, opacity 0.3s ease;
        }
        
        /* Style for signal strength information */
        .signal-meter {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            margin-top: 5px;
            overflow: hidden;
            position: relative;
        }
        
        .signal-bar {
            height: 100%;
            background: linear-gradient(to right, #ff0000, #ffff00, #00ff00);
            width: 0%; /* Will be set dynamically */
            transition: width 0.5s ease;
        }
    </style>
    
    <script>
        // Add pulsing effect to the latest marker icon
        document.addEventListener('DOMContentLoaded', function() {
            // Give a moment for all elements to load
            setTimeout(function() {
                // Find the marker icon (should be the red info icon)
                const markerIcons = document.querySelectorAll('.leaflet-marker-icon');
                if (markerIcons.length > 0) {
                    // Get the last added marker (which should be our latest point)
                    const latestMarker = markerIcons[markerIcons.length - 1];
                    
                    // Apply pulse animation
                    if (latestMarker) {
                        latestMarker.classList.add('pulse-marker');
                        console.log("Applied pulse animation to latest marker");
                    }
                }
            }, 1000); // Increased delay to ensure all elements are loaded
        });
    </script>
    """
    
    if body_end in html_content:
        return html_content.replace(body_end, enhanced_style + body_end)
    else:
        return html_content

def update_hex_map():
    """ Reads CSV and updates the folium hex map with auto-refresh and ping animation """
    try:
        zoom_level = 17  # Increase this number for more zoom (closer view)
        
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            return None
            
        if DEBUG:
            print("Reading CSV file...")
            
        df = pd.read_csv(csv_file)
        
        if DEBUG:
            print(f"CSV loaded with {len(df)} rows")
            print(f"Column names: {df.columns.tolist()}")
            if len(df) > 0:
                print(f"First row: {df.iloc[0].to_dict()}")
                
        if len(df) == 0:
            print("No data found in CSV file")
            return None
            
        # Generate a timestamp for this map file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a unique map file name with sequence number and timestamp
        map_file = os.path.join(map_folder, f"{map_file_base}_{len(df)}_{timestamp}.html")
        
        # Create the latest map symlink/file for ease of access
        latest_map_file = os.path.join(map_folder, f"{map_file_base}_latest.html")
        
        # Convert coordinates to float if they are strings
        for col in ["GPS_X", "GPS_Y", "Signal_Strength"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=["GPS_X", "GPS_Y", "Signal_Strength"])
        
        # Ensure signal strength is normalized between 1-100
        df["Signal_Strength"] = df["Signal_Strength"].apply(normalize_signal_strength)
        
        if DEBUG:
            print(f"After conversion, data shape: {df.shape}")
            print(f"Data types: {df.dtypes}")
            print(f"Signal strength range: {df['Signal_Strength'].min()} - {df['Signal_Strength'].max()}")
        
        if len(df) == 0:
            print("No valid data points after cleaning")
            return None
        
        # Apply hex function with proper error handling
        df["Hex_Cell"] = None
        for idx, row in df.iterrows():
            hex_id = lat_lon_to_hex(row["GPS_X"], row["GPS_Y"])
            if hex_id is not None:
                df.at[idx, "Hex_Cell"] = hex_id
        
        # Drop rows with invalid hex cells
        df = df.dropna(subset=["Hex_Cell"])
        
        if len(df) == 0:
            print("No valid hex cells could be generated")
            return None
        
        # Aggregate signal strength by hex cell
        hex_agg = df.groupby("Hex_Cell")["Signal_Strength"].mean().reset_index()

        # Get the latest data point for centering and zooming the map
        latest_point = df.iloc[-1]
        center_lat = latest_point["GPS_X"]
        center_lon = latest_point["GPS_Y"]
        
        if DEBUG:
            print(f"Center coordinates: {center_lat}, {center_lon}")
        
        # Create a base map centered on the latest data point
        folium_map = folium.Map(
            location=[center_lat, center_lon],  # Ensure these are proper coordinates
            zoom_start=zoom_level,  # Using the higher zoom level
            control_scale=True,
            zoom_control=True,
            tiles='cartodbpositron'
        )

        # Use the normalized 1-100 scale for color mapping
        min_signal = 1
        max_signal = 100
        
        colormap = linear.RdYlGn_09.scale(min_signal, max_signal)  # Red to Green color scale
        
        # Create all hexagons for the covered area to ensure complete grid
        # First, get all unique hex cells
        unique_hex_cells = hex_agg["Hex_Cell"].unique()
        
        # For each hex in our dataset, generate the neighboring hexes too for a more complete grid
        neighbor_hexes = set()
        for hex_id in unique_hex_cells:
            # Add the hex itself
            neighbor_hexes.add(hex_id)
            # Get ring of neighbors (ring 1 = immediate neighbors)
            try:
                ring_neighbors = h3.grid_disk(hex_id, 1)
                for neighbor in ring_neighbors:
                    neighbor_hexes.add(neighbor)
            except Exception as e:
                print(f"Error getting neighbors: {e}")
        
        map_features = []
        
        # Add hexagons to the map - first add a complete grid
        for hex_id in neighbor_hexes:
            boundaries = get_hex_boundaries(hex_id)
            
            # Check if this hex is in our data
            hex_data = hex_agg[hex_agg["Hex_Cell"] == hex_id]
            
            if boundaries:  # Only add if boundaries exist
                if not hex_data.empty:  # If we have signal data for this hex
                    signal_val = hex_data.iloc[0]["Signal_Strength"]
                    tooltip_text = f"Signal Strength: {signal_val:.0f}/100"
                    fill_color = colormap(signal_val)
                    fill_opacity = 0.7
                else:  # For grid cells we don't have data for
                    tooltip_text = "No data"
                    fill_color = "#f0f0f0"  # Light gray
                    fill_opacity = 0.3
                
                # Create the polygon
                poly = folium.Polygon(
                    locations=boundaries,
                    color="#000000",
                    weight=1.5,
                    fill=True,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    tooltip=tooltip_text
                )
                
                # Add it to the map
                poly.add_to(folium_map)
                
                # Add boundaries to features list for bounds calculation
                map_features.append(boundaries)

        # Use normalized signal value for the popup
        signal_normalized = latest_point['Signal_Strength']
        
        latest_marker = folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Latest Reading: {signal_normalized:.0f}/100",
            tooltip=f"Signal: {signal_normalized:.0f}/100",
            icon=folium.Icon(color='red', icon='info-sign', prefix='fa')
        )
        latest_marker.add_to(folium_map)
        
        # Add circle marker for the latest point
        circle_marker = folium.CircleMarker(
            location=[center_lat, center_lon],
            radius=10,
            color='#FF0000',  # Red outline
            fill=True,
            fill_color='#FF0000',  # Red fill
            fill_opacity=0.6,
            weight=2,
            tooltip=f"Signal: {signal_normalized:.0f}/100"
        )
        circle_marker.add_to(folium_map)
        
        # Add a custom JavaScript to ensure the map centers and zooms on the latest point
        zoom_script = f"""
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                // Give the map a moment to initialize
                setTimeout(function() {{
                    // Get the Leaflet map object
                    var map = document.querySelector('.folium-map')._leaflet_map;
                    
                    // Center and zoom to the latest point with higher zoom
                    map.setView([{center_lat}, {center_lon}], {zoom_level});
                }}, 100);
            }});
        </script>
        """
        folium_map.get_root().html.add_child(folium.Element(zoom_script))

        # Add information about map sequence with visual signal meter
        signal_percentage = (signal_normalized - 1) / 99 * 100  # Convert 1-100 range to 0-100%
        
        map_info = f"""
        <div style="position: fixed; 
                    bottom: 10px; 
                    right: 10px; 
                    z-index: 1000; 
                    background-color: white; 
                    padding: 10px; 
                    border-radius: 5px; 
                    box-shadow: 0 0 5px rgba(0,0,0,0.3);">
            <b>Data Point #{len(df)}</b><br>
            Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Signal: <b>{signal_normalized:.0f}/100</b><br>
            <div class="signal-meter">
                <div class="signal-bar" style="width: {signal_percentage}%;"></div>
            </div>
            Hex Resolution: {h3_resolution}<br>
            Zoom Level: {zoom_level}
        </div>
        """
        folium_map.get_root().html.add_child(folium.Element(map_info))

        # Add color legend with appropriate label
        colormap.caption = "Signal Strength (1-100)"
        folium_map.add_child(colormap)
        
        # Save the map as an HTML file
        folium_map.save(map_file)
        
        # Read the HTML content
        with open(map_file, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Add auto-refresh functionality
        html_content = insert_auto_refresh_code(html_content)
        
        # Enhance hex grid visualization
        html_content = enhance_hex_visualization(html_content)
        
        # Write the modified HTML back
        with open(map_file, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        # Save/update the latest map file with auto-refresh
        with open(latest_map_file, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        print(f"New map created: {map_file}")
        
        # Return the path to the created map
        return map_file
                
    except Exception as e:
        print(f"Error in update_hex_map: {e}")
        print(traceback.format_exc())  # Print full traceback
        return None

def parse_data_line(line):
    """Parse a line of data from the serial port - now handling NMEA format"""
    try:
        # Check if this looks like a GPRMC NMEA sentence
        if "GPRMC" in line:
            return parse_gprmc(line)
        else:
            # Fallback to original CSV format parsing
            parts = line.strip().split(",")
            if len(parts) >= 4:
                # Try to convert GPS and signal data to float
                try:
                    timestamp = parts[0].strip()
                    # Add validation for timestamp format if needed
                    if not timestamp:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    gps_x = float(parts[1].strip())
                    gps_y = float(parts[2].strip())
                    raw_signal = float(parts[3].strip())
                    
                    # Normalize the signal strength to 1-100 range
                    signal = normalize_signal_strength(raw_signal)
                    
                    # Basic validation
                    if -90 <= gps_x <= 90 and -180 <= gps_y <= 180:
                        return [timestamp, gps_x, gps_y, signal]
                    else:
                        print(f"Invalid GPS coordinates: {gps_x}, {gps_y}")
                        return None
                except ValueError as ve:
                    print(f"Value conversion error: {ve}")
                    print(f"Raw data: {parts}")
                    return None
            else:
                print(f"Insufficient data elements ({len(parts)}): {line}")
                return None
    except Exception as e:
        print(f"Error parsing data line: {e}")
        print(f"Raw line: {line}")
        return None

def generate_info_page():
    """Generate an index.html file that automatically opens latest map"""
    index_file = os.path.join(map_folder, "index.html")
    latest_map = f"{map_file_base}_latest.html"
    
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Signal Strength Mapping Dashboard</title>
        <meta http-equiv="refresh" content="0;url={latest_map}">
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                margin-top: 100px;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .loading {{
                display: inline-block;
                width: 50px;
                height: 50px;
                border: 3px solid rgba(0,0,0,.3);
                border-radius: 50%;
                border-top-color: #000;
                animation: spin 1s ease-in-out infinite;
                margin: 20px 0;
            }}
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
            h1 {{
                color: #343a40;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Signal Strength Mapping Dashboard</h1>
            <div class="loading"></div>
            <p>Loading the latest signal map visualization...</p>
            <p>If the map doesn't load automatically, <a href="{latest_map}">click here</a>.</p>
            <p><small>Resolution: H3 Level {h3_resolution} | Auto-refresh: {auto_refresh_interval}s</small></p>
            <p><small>Signal strength is normalized to a 1-100 scale</small></p>
        </div>
    </body>
    </html>
    """
    
    with open(index_file, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    print(f"Created dashboard entry point: {index_file}")
    return index_file

def main():
    last_update_time = 0
    error_count = 0
    max_errors = 5  # Maximum number of consecutive errors before reset
    
    # Generate the dashboard entry point
    index_path = generate_info_page()
    
    # Automatically open tab of map
    try:
        webbrowser.open('file://' + os.path.abspath(index_path))
    except Exception as e:
        print(f"Error opening browser: {e}")
    
    print("Starting serial data collection and map generation...")
    print(f"Signal strength will be normalized to a range of 1-100")
    
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        while True:
            try:
                # Check if serial port is open
                if not ser.is_open:
                    print("Serial port closed, attempting to reopen...")
                    ser.open()
                    
                line = ser.readline().decode("utf-8").strip()  # Read serial data
                
                if DEBUG:
                    print(f"Raw data: {line}")  # Debugging output
                
                # Parse the line with better error handling - now includes NMEA support
                data = parse_data_line(line)
                if data:
                    writer.writerow(data)  # Save to CSV
                    file.flush()  # Save instantly
                    error_count = 0  # Reset error counter
                    
                    if DEBUG:
                        print(f"Parsed data: {data}")
                        print(f"Signal strength (1-100): {data[3]:.0f}")
                else:
                    error_count += 1
                    print(f"Error parsing data (count: {error_count})")
                
                # Reset if too many consecutive errors
                if error_count >= max_errors:
                    print("Too many errors, resetting serial connection...")
                    try:
                        ser.close()
                        time.sleep(1)
                        ser.open()
                        error_count = 0
                    except Exception as e:
                        print(f"Error resetting serial connection: {e}")
                
                # Update the map periodically, not on every data point
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    print("Updating hex map...")
                    new_map = update_hex_map()  # Update the map with new data
                    last_update_time = current_time
                    if new_map:
                        print(f"Created new map: {new_map}")
                    else:
                        print("No new map was created")
                    
            except KeyboardInterrupt:
                print("Program stopped by user (Ctrl+C)")
                break
            except serial.SerialException as se:
                print(f"Serial port error: {se}")
                time.sleep(1)  # Wait before retry
                try:
                    ser.close()
                    time.sleep(1)
                    ser.open()
                except Exception as e:
                    print(f"Error reopening serial port: {e}")
            except Exception as e:
                print(f"Error in main loop: {e}")
                print(traceback.format_exc())  # Print full traceback
                time.sleep(0.5)  # Prevent CPU overload on errors

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        print(traceback.format_exc())
        input("Press Enter to exit...")  # Keep console window open on fatal error