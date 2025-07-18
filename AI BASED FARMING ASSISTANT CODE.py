# -*- coding: utf-8 -*- # Added for explicit encoding declaration
import streamlit as st
import cv2
import torch
import google.generativeai as genai
import sqlite3
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import datetime
import tempfile # Needed for handling uploaded video files and saving processed video

# --- DeepSORT Imports (Adjust for deep-sort-realtime) ---
try:
    # Use the correct import for deep-sort-realtime
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
    print("Successfully imported DeepSort from deep_sort_realtime.")
except ImportError:
    print("WARNING: deep_sort_realtime library not found. Tracking features will be disabled.")
    DEEPSORT_AVAILABLE = False
    DeepSort = None # Define as None if not available
# -------------------------------------------------------

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Smart Farming Assistant", layout="wide")
# ------------------------------------------------------------------

# --- Configuration ---
# Configure Gemini API (Replace with your actual key)
GEMINI_API_KEY = "YOUR KEY " # <--- REPLACE WITH YOUR KEY
try:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        gemini_available = False
        chat_model = None
        st.sidebar.warning("⚠️ Gemini API Key missing. Chatbot disabled.")
        print("Gemini API Key missing or placeholder.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        chat_model = genai.GenerativeModel("gemini-1.5-pro-latest")
        gemini_available = True
        print("Gemini API Configured Successfully.")
except Exception as e:
    st.sidebar.error(f"⚠️ Gemini API Error: {e}. Chatbot disabled.")
    print(f"Gemini API Configuration Error: {e}")
    gemini_available = False
    chat_model = None

# Model Paths
MODEL_PATHS = {
    "Harmful Animal": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\harmful animal 25e\weights\best.pt",
    "Harmful Insect": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\final harmful insect\weights\best.pt",
    "Cattle Management": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\ANIMAL DETECTION LIVESTOCK\best.pt",
    "Wheat Disease": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\wheat disease\weights\best.pt",
    "Rice Disease": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\rice dusease\best.pt",
    "Tomato Disease": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\tomato disaese 50 train_results\weights\best.pt",
    "Corn Disease": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\corn disease\best.pt",
    "Weed Detection": r"C:\Users\Admin\Desktop\project\agriculture\final moedl folder\WeedDetectiontrain\weights\best.pt"
}
# Models to trigger alerts for
ALERT_MODELS = {"Harmful Animal", "Harmful Insect", "Weed Detection"}
ALERT_CONFIDENCE_THRESHOLD = 0.70

# --- NEW: Models requiring DeepSORT tracking ---
TRACKING_MODELS = {"Harmful Animal", "Harmful Insect", "Cattle Management"}
# ----------------------------------------------

# Load YOLO Models
@st.cache_resource
def load_models():
    """Loads YOLO models specified in MODEL_PATHS. Handles errors gracefully."""
    models = {}
    # missing_models = [] # Commented out as not used later
    loaded_model_names = []
    print("--- Loading Models ---")
    for name, path in MODEL_PATHS.items():
        absolute_path = os.path.abspath(path)
        if os.path.exists(absolute_path):
            try:
                models[name] = YOLO(absolute_path)
                print(f"  [ OK ] Successfully loaded model: {name} from {absolute_path}")
                loaded_model_names.append(name)
            except Exception as e:
                st.sidebar.error(f"Error loading '{name}': {e}", icon="⚠️")
                print(f"  [FAIL] Error loading model: {name} from {absolute_path} - {e}")
                # missing_models.append(name)
        else:
            st.sidebar.error(f"Path not found for '{name}': {absolute_path}", icon="❌")
            print(f"  [FAIL] Path not found: {name} at {absolute_path}")
            # missing_models.append(name)

    global ALERT_MODELS
    original_alert_models = ALERT_MODELS.copy()
    ALERT_MODELS.intersection_update(set(loaded_model_names))
    removed_alert_models = original_alert_models - ALERT_MODELS
    if removed_alert_models:
        print(f"  [WARN] Removed models from ALERT list due to loading issues: {', '.join(removed_alert_models)}")

    print(f"--- Model Loading Complete ({len(models)} loaded) ---")
    return models

MODELS = load_models()

# --- DeepSORT Initializer (for deep-sort-realtime - using default embedder) ---
@st.cache_resource # Cache tracker resource
def initialize_tracker():
    """Initializes the DeepSORT tracker using deep-sort-realtime library."""
    if not DEEPSORT_AVAILABLE:
        print("DeepSORT (deep_sort_realtime) not available, cannot initialize tracker.")
        return None

    try:
        # Initialize DeepSort from deep_sort_realtime
        # Using the default 'mobilenet' embedder confirmed to work
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet", # <--- Use the confirmed 'mobilenet'
            half=True,
            bgr=True,
            embedder_gpu=torch.cuda.is_available(), # Use GPU for embedder if available
            embedder_model_name=None, # Not needed when using standard built-in embedder
            embedder_wts=None, # Not needed when using standard built-in embedder
            polygon=False,
            today=None
        )
        print("DeepSORT Tracker (deep_sort_realtime) Initialized Successfully.")
        return tracker
    except Exception as e:
        print(f"ERROR: Failed to initialize DeepSORT tracker (deep_sort_realtime): {e}")
        # Show error in Streamlit sidebar if possible
        try:
            st.sidebar.error(f"DeepSORT Init Error: {e}", icon="🧭")
        except Exception:
            pass # Ignore if Streamlit context not available
        return None

# --- Initialize the tracker ---
deepsort_tracker = initialize_tracker()

# Display DeepSORT status warning in sidebar if needed
# Placed after initialization attempt
if not DEEPSORT_AVAILABLE:
    st.sidebar.warning("DeepSORT library missing. Tracking disabled.", icon="⚠️")
elif deepsort_tracker is None:
     st.sidebar.error("DeepSORT tracker failed to initialize. Tracking disabled.", icon="🧭")
# ------------------------------------------

# Database Setup
db_path = r"C:\Mr Rahul\KEEPer\appdata.db"
db_dir = os.path.dirname(db_path)
if not os.path.exists(db_dir):
    try:
        os.makedirs(db_dir)
        print(f"Created database directory: {db_dir}")
    except OSError as e:
        st.error(f"Fatal: Failed to create database directory {db_dir}: {e}. Cannot proceed.")
        print(f"Fatal: Failed to create database directory {db_dir}: {e}")
        st.stop()

# Initialize Session State Variables
if 'processing_video' not in st.session_state: st.session_state.processing_video = False
if 'processing_live' not in st.session_state: st.session_state.processing_live = False
if 'messages' not in st.session_state: st.session_state.messages = []
if 'processed_video_path' not in st.session_state: st.session_state.processed_video_path = None
if 'video_alerts' not in st.session_state: st.session_state.video_alerts = []
if 'video_detections_summary' not in st.session_state: st.session_state.video_detections_summary = {}
if 'video_analysis_text' not in st.session_state: st.session_state.video_analysis_text = None

def clear_video_results():
    st.session_state.processed_video_path = None
    st.session_state.video_alerts = []
    st.session_state.video_detections_summary = {}
    st.session_state.video_analysis_text = None
    print("Cleared previous video processing results from session state.")

# Database Connection and Table Creation
@st.cache_resource
def get_db_connection(path):
    print(f"Attempting to connect to database: {path}")
    try:
        conn_db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        print("Database connection successful.")
        return conn_db
    except sqlite3.Error as e:
        st.error(f"Database connection failed: {e}", icon="💾")
        print(f"Database connection failed: {e}")
        return None

conn = get_db_connection(db_path)

if conn:
    cursor = conn.cursor()
    try:
        # Detections Table (Added track_id column)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                detected TEXT NOT NULL,
                confidence REAL NOT NULL,
                track_id INTEGER DEFAULT NULL -- Optional: Add track ID
            )
        ''')
        # Chatbot Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chatbot_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, query TEXT, response TEXT
            )
        ''')
        # Alerts Table (Added track_id column)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, model TEXT NOT NULL,
                detected_object TEXT NOT NULL, confidence REAL NOT NULL,
                track_id INTEGER DEFAULT NULL -- Optional: Add track ID
            )
        ''')
        conn.commit()
        print("Database tables checked/created successfully.")
    except sqlite3.Error as e:
        st.error(f"Database table creation/check failed: {e}", icon="⚠️")
        print(f"Database table creation/check failed: {e}")
        conn = None
else:
    st.error("Halting execution: Database connection could not be established.", icon="🛑")
    st.stop()

# --- Helper Functions ---

def format_yolo_detections_for_deepsort(results, frame_shape):
    """
    Formats YOLOv8 detection results into the list of tuples format required
    by deep-sort-realtime: [( [left,top,w,h], confidence, class_id ), ...].
    """
    detections_for_ds = []
    # Add defensive checks
    if not results or not results[0] or not hasattr(results[0], 'boxes') or results[0].boxes is None:
        return []

    # Check if boxes tensor is empty
    if results[0].boxes.shape[0] == 0:
        return []

    try:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        confs = results[0].boxes.conf.cpu().numpy()  # Confidences
        clss = results[0].boxes.cls.cpu().numpy()    # Class IDs
    except Exception as e:
        print(f"Error accessing YOLO results tensors: {e}")
        return []


    for box, conf, cls_id_float in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
        class_id = int(cls_id_float) # Class ID needs to be int

        # Format required by deep-sort-realtime: [left, top, w, h]
        bbox_xywh = [x1, y1, w, h]

        # Filter out invalid boxes if necessary
        if w <= 0 or h <= 0:
            # print(f"Skipping invalid bbox: {bbox_xywh}") # Optional log
            continue

        # Append detection in deep-sort-realtime format
        detections_for_ds.append((bbox_xywh, float(conf), class_id)) # Use int class_id

    return detections_for_ds


def process_frame_for_alerts(frame_pil, selected_models):
    """
    Processes a single frame, performs detection/tracking, draws boxes,
    identifies/logs alerts, and logs all detections/tracks.
    """
    frame_alerts = []
    all_detections_log = [] # Log entries for this frame
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        np_frame = np.array(frame_pil.convert('RGB'))
        bgr_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        # frame_height, frame_width = bgr_frame.shape[:2] # Not used directly here

        for model_name in selected_models:
            if model_name not in MODELS:
                continue

            model = MODELS[model_name]
            results = model(frame_pil, verbose=False) # Get YOLO detections

            # --- Decide whether to apply DeepSORT tracking ---
            apply_tracking = model_name in TRACKING_MODELS and deepsort_tracker is not None

            if apply_tracking:
                # --- Format Detections for DeepSORT ---
                detections_ds_input = format_yolo_detections_for_deepsort(results, bgr_frame.shape)

                if detections_ds_input:
                    # --- Update DeepSORT Tracker ---
                    # Using frame=bgr_frame is crucial for the embedder if needed
                    tracks = deepsort_tracker.update_tracks(detections_ds_input, frame=bgr_frame)

                    # --- Process and Draw Tracked Objects ---
                    for track in tracks:
                        if not track.is_confirmed(): continue

                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = map(int, ltrb)
                        original_class_id = track.get_det_class()
                        class_name = original_class_id # Default
                        if isinstance(original_class_id, (int, float, str)) and str(original_class_id).isdigit():
                           class_name = model.names.get(int(original_class_id), f"Class_{int(original_class_id)}")
                        elif isinstance(original_class_id, str): class_name = original_class_id
                        confidence = track.get_det_conf()
                        if confidence is None: confidence = 0.99 # Use default if not provided

                        label = f"ID {track_id}: {class_name} ({confidence:.2f})"
                        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box
                        text_y = y1 - 10 if y1 > 20 else y1 + 15
                        cv2.putText(bgr_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        detection_info = f"Tracked ({model_name}): ID {track_id} - {class_name} ({confidence:.2f})"
                        all_detections_log.append(detection_info)

                        # --- Log Tracked Detection to DB ---
                        if conn:
                           try:
                               cursor.execute("INSERT INTO detections (timestamp, model, detected, confidence, track_id) VALUES (?, ?, ?, ?, ?)",
                                              (current_time, model_name, class_name, confidence, track_id))
                               conn.commit()
                           except sqlite3.Error as e: print(f"DB Error (logging track): {e}")

                        # --- Check Alert Conditions for Tracked Objects ---
                        if model_name in ALERT_MODELS and confidence > ALERT_CONFIDENCE_THRESHOLD:
                             alert_msg = f"🚨 ALERT ({current_time}): {model_name} tracked ID {track_id} ('{class_name}') with confidence {confidence:.2f}"
                             frame_alerts.append(alert_msg)
                             if conn:
                                 try:
                                     cursor.execute("INSERT INTO alerts (timestamp, model, detected_object, confidence, track_id) VALUES (?, ?, ?, ?, ?)",
                                                    (current_time, model_name, class_name, confidence, track_id))
                                     conn.commit()
                                 except sqlite3.Error as e: print(f"DB Error (logging tracked alert): {e}")

            else: # --- Process normally if tracking is NOT applied ---
                # Add defensive check for results structure
                if results and results[0] and hasattr(results[0], 'boxes') and results[0].boxes is not None and results[0].boxes.shape[0] > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    clss = results[0].boxes.cls.cpu().numpy()
                    names = results[0].names

                    for box, conf, cls_id_float in zip(boxes, confs, clss):
                        x1, y1, x2, y2 = map(int, box)
                        confidence = float(conf)
                        class_id = int(cls_id_float)

                        # Check class_id validity against the 'names' dict from the model
                        if names and (class_id < 0 or class_id >= len(names)):
                             # print(f"Warning: Invalid class_id {class_id} from {model_name}. Skipping.")
                             continue

                        detected_obj = names.get(class_id, f"Unknown_{class_id}") if names else f"Class_{class_id}"
                        label = f"{detected_obj} ({confidence:.2f})"

                        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                        text_y = y1 - 10 if y1 > 20 else y1 + 15
                        cv2.putText(bgr_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        detection_info = f"{model_name}: {detected_obj} ({confidence:.2f})"
                        all_detections_log.append(detection_info)

                        if conn:
                           try:
                               cursor.execute("INSERT INTO detections (timestamp, model, detected, confidence) VALUES (?, ?, ?, ?)",
                                              (current_time, model_name, detected_obj, confidence))
                               conn.commit()
                           except sqlite3.Error as e: print(f"DB Error (logging detection): {e}")

                        # Check standard alert conditions
                        if model_name in ALERT_MODELS and confidence > ALERT_CONFIDENCE_THRESHOLD:
                             alert_msg = f"🚨 ALERT ({current_time}): {model_name} detected '{detected_obj}' with confidence {confidence:.2f}"
                             frame_alerts.append(alert_msg)
                             if conn:
                                 try:
                                     cursor.execute("INSERT INTO alerts (timestamp, model, detected_object, confidence) VALUES (?, ?, ?, ?)",
                                                    (current_time, model_name, detected_obj, confidence))
                                     conn.commit()
                                 except sqlite3.Error as e: print(f"DB Error (logging alert): {e}")

        processed_frame_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return processed_frame_rgb, frame_alerts, all_detections_log

    except Exception as e:
        print(f"ERROR in process_frame_for_alerts: {e}")
        try:
             return np.array(frame_pil.convert('RGB')), [], []
        except Exception:
             print("Failed to return original frame on error, returning blank.")
             return np.zeros((100, 100, 3), dtype=np.uint8), [], []


# --- Streamlit App UI ---
st.title("🌾 AI-powered Smart Farming Assistant")

# Sidebar Navigation
st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Choose an option:",
    ["Live Feed", "Manual Input", "Chatbot", "View Analysis", "View Alerts"],
    key="navigation_option",
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.header("Detection Settings")

available_model_keys = list(MODELS.keys())
default_selection = available_model_keys[:1] if available_model_keys else [] # Default to first model only
selected_models_sidebar = st.sidebar.multiselect(
    "Select Models for Detection",
    available_model_keys,
    default=default_selection,
    key="model_selector",
    help="Choose AI models. Tracking applied to Harmful Animal/Insect, Cattle."
)

if option not in ["View Analysis", "View Alerts"]:
    with st.expander("🌟 Project Overview & Status", expanded=False):
        tracking_status = "Ready" if DEEPSORT_AVAILABLE and deepsort_tracker else "Disabled"
        if not DEEPSORT_AVAILABLE: tracking_status += " (Lib Missing)"
        elif not deepsort_tracker: tracking_status += " (Init Failed)"

        st.markdown(f"""
        This assistant uses AI models to help with farming tasks:
        - Detects **harmful animals & insects** (Tracked, Alerts if confidence > {ALERT_CONFIDENCE_THRESHOLD*100:.0f}%)
        - Monitors **livestock** (Tracked).
        - Identifies potential crop **diseases** (Detection only).
        - Spots **weed infestations** (Detection only, Alerts if confidence > {ALERT_CONFIDENCE_THRESHOLD*100:.0f}%).

        ---
        **System Status:**
        - Models Loaded: **{len(MODELS)}** / {len(MODEL_PATHS)}
        - Active Alert Models: **{len(ALERT_MODELS)}** ({', '.join(ALERT_MODELS) if ALERT_MODELS else 'None'})
        - Object Tracking: **{tracking_status}**
        - Database: **{'Connected' if conn else 'Disconnected'}**
        - Gemini Chatbot: **{'Ready' if gemini_available else 'Not Configured'}**
        """)
    st.markdown("---")

# --- App Sections ---

# Live Feed Detection
if option == "Live Feed":
    st.header("📹 Live Feed Detection")

    if not MODELS: st.error("No detection models loaded.", icon="🚫")
    elif not selected_models_sidebar: st.warning("Select models from sidebar.", icon="👈")
    else:
        col1, col2 = st.columns(2)
        with col1: start_button = st.button("🚀 Start Live Feed", key="start_live", disabled=st.session_state.processing_live, use_container_width=True)
        with col2: stop_button = st.button("🛑 Stop Live Feed", key="stop_live", disabled=not st.session_state.processing_live, use_container_width=True)

        if start_button: st.session_state.processing_live = True; st.rerun()
        if stop_button: st.session_state.processing_live = False; st.rerun()

        if st.session_state.processing_live:
            st.info("Connecting to webcam...", icon="📷")
            live_feed_container = st.container()
            with live_feed_container:
                stframe = st.empty()
                alert_placeholder = st.container()

            cap = None # Initialize cap to None
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Error: Could not open webcam.", icon="❌")
                    st.session_state.processing_live = False; st.rerun()
                else:
                    st.toast("Webcam connected!", icon="✅")
                    while st.session_state.processing_live:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Webcam frame grab failed. Stopping.", icon="⚠️")
                            st.session_state.processing_live = False # Set flag first
                            # No need to release cap here, finally block handles it
                            st.rerun() # Rerun to update UI
                            break # Exit loop

                        # Process frame only if processing is still active
                        if st.session_state.processing_live:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_frame)
                            processed_frame_rgb, frame_alerts, frame_detections = process_frame_for_alerts(pil_image, selected_models_sidebar)
                            stframe.image(processed_frame_rgb, channels="RGB", caption="Live Feed (Blue boxes = Tracked)")
                            with alert_placeholder:
                                alert_placeholder.empty() # Clear previous frame's alerts
                                if frame_alerts:
                                    for alert in frame_alerts: st.warning(alert)

                        # Re-check the flag after processing/displaying
                        if not st.session_state.processing_live:
                            print("Stop button pressed during loop, breaking.")
                            break

            except Exception as e:
                st.error(f"Live feed error: {e}", icon="🔥")
                print(f"Live feed error: {e}")
                st.session_state.processing_live = False # Ensure stop on error

            finally: # Use finally to guarantee release
                 if cap is not None and cap.isOpened():
                     cap.release()
                     print("Webcam released in finally block.")
                 # Only show "stopped" message if stop was pressed
                 if stop_button:
                     st.info("Live feed stopped.")
                     # Rerun required after manual stop to clear processing state visually
                     # This was missing in the if block, adding here ensures it runs
                     # We should check if we are already stopping to avoid double rerun
                     if st.session_state.processing_live: # Should be False here if stop pressed
                         pass # Already handled by rerun in stop_button logic
                     else:
                        # If stopped by error or other means, just show message, rerun handled elsewhere or not needed.
                        pass


        # Display message if not processing and wasn't just stopped by button
        elif not st.session_state.processing_live and not stop_button:
             st.info("Press 'Start Live Feed'.", icon="▶️")


# Manual Input (Image/Video)
elif option == "Manual Input":
    st.header("📤 Manual Input")
    uploaded_file = st.file_uploader( "Upload an Image or Video File", type=["jpg", "png", "jpeg", "mp4", "avi", "mov", "mkv"], key="file_uploader", help="Select an image or video file for analysis.", on_change=clear_video_results)

    if uploaded_file:
        file_type = uploaded_file.type
        if not MODELS: st.error("No models loaded.", icon="🚫")
        elif not selected_models_sidebar: st.warning("Select models from sidebar.", icon="👈")
        else:
            # --- Image Processing ---
            if file_type and file_type.startswith("image/"):
                st.subheader("🖼️ Image Analysis Results")
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    col1, col2 = st.columns(2)
                    with col1: st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                    with st.spinner("Analyzing image..."):
                        processed_image, alerts, detections = process_frame_for_alerts(image, selected_models_sidebar)
                    with col2: st.image(processed_image, caption="Processed Image (Blue boxes = Tracked)", use_column_width=True)
                    if alerts:
                        st.warning("Alerts Found:")
                        # Use list comprehension for cleaner display code if desired
                        for a in alerts:
                            st.write(a)
                    else: st.success("No high-confidence alerts found.", icon="✅")

                    if (detections or alerts) and gemini_available and chat_model:
                        with st.expander("💡 AI Farming Advisor Feedback (Image)"):
                            try:
                                # Corrected f-string date formatting
                                prompt = (f"CONTEXT: Analysis of image '{uploaded_file.name}'. Location: Jaipur, Rajasthan. Date: {datetime.date.today():%B %d, %Y}.\n\n"
                                          f"DETECTIONS/TRACKS: {', '.join(detections) if detections else 'None'}\n"
                                          f"ALERTS: {', '.join(alerts) if alerts else 'None'}\n\n"
                                          f"TASK: As Rajasthan farm advisor, provide concise, actionable advice based *only* on these findings. Prioritize alerts. Suggest local treatments/prevention. Mention persistence if track IDs present.")
                                with st.spinner("Generating AI feedback..."):
                                    response = chat_model.generate_content(prompt)
                                    st.markdown(response.text)
                            except Exception as e: st.error(f"AI feedback error: {e}", icon="🤖"); print(f"Gemini feedback error (image): {e}")
                except Exception as e: st.error(f"Image processing error: {e}", icon="🔥"); print(f"Image processing error: {e}")

            # --- Video Processing ---
            elif file_type and file_type.startswith("video/"):
                st.subheader("🎬 Video Analysis")
                if not st.session_state.processing_video and not st.session_state.processed_video_path: st.video(uploaded_file)
                process_vid_button = st.button("🎞️ Process Uploaded Video", key="process_vid", disabled=st.session_state.processing_video, help="Analyze video, save results. Tracking for Animal/Insect/Cattle.")

                if process_vid_button: clear_video_results(); st.session_state.processing_video = True; st.rerun()

                if st.session_state.processing_video:
                    input_video_path, output_video_path, output_video, cap, processing_successful = None, None, None, None, False
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile_in:
                            tfile_in.write(uploaded_file.read()); input_video_path = tfile_in.name
                        st.info(f"Processing video: {uploaded_file.name}...", icon="⏳")
                        cap = cv2.VideoCapture(input_video_path)
                        if not cap.isOpened(): raise IOError(f"Cannot open video: {input_video_path}")
                        width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
                        fps = 30 if fps <= 0 else fps
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        temp_dir = tempfile.gettempdir()
                        safe_filename_base = "".join(c for c in os.path.splitext(uploaded_file.name)[0] if c.isalnum() or c in ('_','-')).rstrip()
                        output_video_path = os.path.join(temp_dir, f"processed_{safe_filename_base}.mp4")
                        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        if not output_video.isOpened(): raise IOError(f"Cannot open video writer: {output_video_path}")
                        print(f"Output video: {output_video_path}")

                        status_container = st.container(); prog_bar = status_container.progress(0, text="Initializing...")
                        all_video_alerts_accum, all_video_detections_accum, frame_count = [], {}, 0

                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            frame_count += 1
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); pil_image = Image.fromarray(rgb_frame)
                            processed_frame_rgb, frame_alerts, frame_detections_log = process_frame_for_alerts(pil_image, selected_models_sidebar)
                            bgr_frame_to_write = cv2.cvtColor(processed_frame_rgb, cv2.COLOR_RGB2BGR)
                            output_video.write(bgr_frame_to_write)
                            progress = int(100 * frame_count / total_frames) if total_frames > 0 else 100
                            prog_bar.progress(progress, text=f"Processing... {progress}% (Frame {frame_count}{f'/{total_frames}' if total_frames > 0 else ''})")
                            if frame_alerts: all_video_alerts_accum.extend(frame_alerts)
                            # Corrected list comprehension for updating dict
                            if frame_detections_log:
                                for di in frame_detections_log:
                                    all_video_detections_accum[di] = all_video_detections_accum.get(di, 0) + 1
                        print(f"Video processing done ({frame_count} frames)."); processing_successful = True
                    except Exception as e: st.error(f"Video processing error: {e}", icon="🔥"); print(f"Video processing error: {e}"); processing_successful, output_video_path = False, None
                    finally:
                        # Use finally to ensure resources are released
                        if output_video and output_video.isOpened():
                            output_video.release()
                            print("Output video released.")
                        if cap and cap.isOpened():
                            cap.release()
                            print("Input video released.")
                        # --- Clean up temporary input file (Corrected Syntax) ---
                        if input_video_path and os.path.exists(input_video_path):
                            try: # Indented try
                                os.unlink(input_video_path)
                                print(f"Deleted temporary input file: {input_video_path}")
                            except Exception as e_unlink: # Indented except
                                print(f"Error deleting temp input file {input_video_path}: {e_unlink}")
                        # --- End Corrected Syntax ---

                    if processing_successful and output_video_path and os.path.exists(output_video_path):
                        st.session_state.processed_video_path, st.session_state.video_alerts, st.session_state.video_detections_summary = output_video_path, sorted(list(set(all_video_alerts_accum))), all_video_detections_accum
                        if gemini_available and chat_model:
                            with st.spinner("Generating AI analysis..."):
                                try:
                                    summary_items = [f"{k} ({v} times)" for k, v in sorted(st.session_state.video_detections_summary.items(), key=lambda item: item[1], reverse=True)]
                                    det_summary = ", ".join(summary_items) or "None"
                                    alert_summary = ", ".join(st.session_state.video_alerts) or "None"
                                    # Corrected f-string date formatting
                                    prompt = (f"CONTEXT: Video Analysis '{uploaded_file.name}'. Location: Jaipur. Date: {datetime.date.today():%B %d, %Y}.\n\n"
                                              f"ALERTS: {alert_summary}\nDETECTION/TRACKING SUMMARY: {det_summary}\n\n"
                                              f"TASK: As Rajasthan farm advisor, provide concise analysis & actionable solutions. Prioritize alerts. Focus on frequent/critical findings. Mention persistence if track IDs appear.")
                                    response = chat_model.generate_content(prompt)
                                    if response.parts: st.session_state.video_analysis_text = response.text
                                    elif response.prompt_feedback.block_reason: st.session_state.video_analysis_text = f"⚠️ AI analysis blocked: {response.prompt_feedback.block_reason}"
                                    else: st.session_state.video_analysis_text = "⚠️ AI analysis generation failed."
                                    print("AI video analysis done.")
                                except Exception as e: st.session_state.video_analysis_text = f"⚠️ AI analysis error: {e}"; print(f"Gemini video analysis error: {e}")
                        else: st.session_state.video_analysis_text = "AI analysis unavailable."; print("AI video analysis skipped.")
                    else:
                        if not processing_successful: st.error("Video processing failed.")
                        clear_video_results()
                    st.session_state.processing_video = False; print("Processing flag False. Rerunning."); st.rerun()

                if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                    st.markdown("---"); st.subheader("Processed Video & Analysis")
                    try:
                        with open(st.session_state.processed_video_path, 'rb') as vf: video_bytes = vf.read()
                        st.video(video_bytes, format='video/mp4')
                        st.download_button(label="⬇️ Download Processed Video", data=video_bytes, file_name=f"processed_{os.path.basename(uploaded_file.name)}.mp4", mime='video/mp4')
                    except Exception as e: st.error(f"Error reading processed video: {e}", icon="📺"); print(f"Error reading processed video: {e}")
                    st.markdown("---"); st.markdown("**Alerts Summary (from video)**")
                    if st.session_state.video_alerts: st.warning(f"Alerts found:\n```\n" + "\n".join([f"- {a}" for a in st.session_state.video_alerts]) + "\n```")
                    else: st.success("No high-confidence alerts found.", icon="✅")
                    if st.session_state.video_detections_summary:
                        with st.expander("Show Full Detection/Tracking Summary"):
                             # Corrected list comprehension usage
                             summary_lines = [f"- {k}: {v} times" for k, v in sorted(st.session_state.video_detections_summary.items(), key=lambda item: item[1], reverse=True)]
                             st.code("\n".join(summary_lines), language=None)
                    st.markdown("**💡 AI Analysis & Solution (from video)**")
                    if st.session_state.video_analysis_text: st.markdown(st.session_state.video_analysis_text)
                    else: st.info("AI Analysis unavailable or pending.")

# Chatbot
elif option == "Chatbot":
    st.header("💬 AI Farming Advisor Chat")
    if not gemini_available or not chat_model: st.error("Chatbot unavailable. Configure Gemini API Key.", icon="🤖")
    else:
        st.info("Ask farming questions. For image/video analysis, use 'Manual Input'.", icon="💡")
        # Corrected Chat History Display Loop
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): # Indented with
                st.markdown(msg["content"]) # Indented markdown

        if prompt := st.chat_input("Ask your farming query here..."):
            # Corrected User Message Display Block
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): # Indented with
                st.markdown(prompt) # Indented markdown

            with st.spinner("Thinking..."):
                try:
                    loc, dt = "Jaipur, Rajasthan", f"{datetime.date.today():%B %d, %Y}"
                    # Corrected f-string date formatting
                    full_prompt = (f"CONTEXT: Expert AI farming advisor for {loc}. Date: {dt}.\n\nUSER QUERY: {prompt}\n\n"
                                   f"TASK: Answer comprehensively with practical, local advice (climate, soil, crops like bajra/wheat/mustard, pests). Ask for clarity if needed.")
                    response = chat_model.generate_content(full_prompt)
                    resp_text = response.text if response.parts else (f"⚠️ Blocked: {response.prompt_feedback.block_reason}" if response.prompt_feedback.block_reason else "⚠️ No response.")
                    if response.prompt_feedback.block_reason: st.warning(resp_text)
                except Exception as e: resp_text = f"AI Error: {e}"; st.error(resp_text, icon="🔥"); print(f"Gemini Error: {e}")

                with st.chat_message("assistant"): st.markdown(resp_text)
                st.session_state.messages.append({"role": "assistant", "content": resp_text})

                # Corrected Database Logging Block
                if conn:
                    try: # Indented try
                        cursor.execute("INSERT INTO chatbot_queries (timestamp, query, response) VALUES (?, ?, ?)",
                                      (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prompt, resp_text)) # Args on next line
                        conn.commit()
                    except sqlite3.Error as e: # Indented except
                        print(f"DB Error(chat): {e}") # Indented print

# View Analysis
elif option == "View Analysis":
    st.header("📊 Detection History"); st.subheader("Recent Detections & Tracks Log"); st.caption("Shows latest objects (incl. tracked IDs).")
    if conn:
        try:
            models_list = ["All"] + available_model_keys; filter_model = st.selectbox("Filter by Model:", models_list, index=0)
            query = "SELECT timestamp, model, detected, confidence, track_id FROM detections"; params = []
            if filter_model != "All": query += " WHERE model = ?"; params.append(filter_model)
            query += " ORDER BY id DESC LIMIT 100"
            cursor.execute(query, params); detections = cursor.fetchall()
            if detections:
                col1, col2, col3, col4, col5 = st.columns([2, 2, 3, 1, 1])
                col1.markdown("**Timestamp**"); col2.markdown("**Model**"); col3.markdown("**Object**"); col4.markdown("**Conf.**"); col5.markdown("**TrackID**"); st.markdown("---")
                for det in detections:
                    # Use inner columns for alignment within the loop
                    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns([2, 2, 3, 1, 1])
                    rcol1.text(det[0]); rcol2.text(det[1]); rcol3.text(det[2]); rcol4.text(f"{det[3]:.2f}"); rcol5.text(f"{det[4] if det[4] is not None else '-'}")
            else: st.info(f"No detections recorded yet{f' for {filter_model}' if filter_model != 'All' else ''}.", icon="⏱️")
        except sqlite3.Error as e: st.error(f"DB Error (reading detections): {e}", icon="💾"); print(f"DB Error (detections): {e}")
    else: st.error("Database connection unavailable.", icon="🔌")
    st.markdown("---"); st.subheader("💬 Recent Chatbot Interactions"); st.caption("Latest questions and answers.")
    if conn:
        try:
            cursor.execute("SELECT timestamp, query, response FROM chatbot_queries ORDER BY id DESC LIMIT 10"); queries = cursor.fetchall()
            # Corrected list comprehension for expanders
            if queries:
                for q in queries:
                    with st.expander(f"Chat from {q[0]}"):
                         st.markdown(f"**👤 You:**\n```\n{q[1]}\n```\n**🤖 Advisor:**\n{q[2]}")
            else: st.info("No chatbot interactions recorded.", icon="💬")
        except sqlite3.Error as e: st.error(f"DB Error (reading chat): {e}", icon="💾"); print(f"DB Error (chat): {e}")
    else: st.error("Database connection unavailable.", icon="🔌")

# View Alerts
elif option == "View Alerts":
    st.header("🚨 Alert History"); st.subheader(f"Recorded Alerts (> {ALERT_CONFIDENCE_THRESHOLD*100:.0f}%)"); st.caption(f"High-confidence detections from: {', '.join(ALERT_MODELS) or 'None'}")
    if conn:
        try:
            alert_list = ["All"] + list(ALERT_MODELS); filter_alert = st.selectbox("Filter by Alert Model:", alert_list, index=0)
            query = "SELECT timestamp, model, detected_object, confidence, track_id FROM alerts"; params = []
            if filter_alert != "All": query += " WHERE model = ?"; params.append(filter_alert)
            query += " ORDER BY id DESC LIMIT 100"
            cursor.execute(query, params); alerts = cursor.fetchall()
            if alerts:
                col1, col2, col3, col4, col5 = st.columns([2, 2, 3, 1, 1])
                col1.markdown("**Timestamp**"); col2.markdown("**Model**"); col3.markdown("**Object**"); col4.markdown("**Conf.**"); col5.markdown("**TrackID**"); st.markdown("---")
                for alert in alerts:
                    # Use inner columns for alignment
                    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns([2, 2, 3, 1, 1])
                    rcol1.text(alert[0]); rcol2.text(alert[1]); rcol3.text(alert[2]); rcol4.warning(f"**{alert[3]:.2f}**"); rcol5.text(f"{alert[4] if alert[4] is not None else '-'}")
            else: st.info(f"No alerts recorded yet{f' for {filter_alert}' if filter_alert != 'All' else ''}.", icon="🔕")
        except sqlite3.Error as e: st.error(f"DB Error (reading alerts): {e}", icon="💾"); print(f"DB Error (alerts): {e}")
    else: st.error("Database connection unavailable.", icon="🔌")

# --- Footer / Sidebar System Status ---
st.sidebar.markdown("---")
st.sidebar.header("System Status")
if MODELS:
    st.sidebar.success(f"Models Loaded: {len(MODELS)}/{len(MODEL_PATHS)}", icon="🧠")
    st.sidebar.info(f"Alert Models: {len(ALERT_MODELS)} ({', '.join(ALERT_MODELS) or 'None'})", icon="🔔")
    st.sidebar.caption(f"Alert Threshold: > {ALERT_CONFIDENCE_THRESHOLD*100:.0f}%")
    tracking_icon = "🧭" if DEEPSORT_AVAILABLE and deepsort_tracker else "⚠️"
    tracking_text = "Tracking Ready" if DEEPSORT_AVAILABLE and deepsort_tracker else "Tracking Disabled"
    st.sidebar.info(f"{tracking_text}", icon=tracking_icon)


else: st.sidebar.error("No models loaded.", icon="❌")
if conn: st.sidebar.success("Database Connected", icon="💾")
else: st.sidebar.error("Database Disconnected", icon="🔌")
if gemini_available: st.sidebar.success("Gemini API Ready", icon="🤖")

st.sidebar.caption(f"Location Context: Jaipur, Rajasthan")
# Corrected f-string time formatting
st.sidebar.caption(f"Current Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

print("--- Streamlit script execution finished ---")
