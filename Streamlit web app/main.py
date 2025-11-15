# main.py
import tempfile
import os
import numpy as np
import pandas as pd
import streamlit as st
# from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detection import create_colors_info, detect
import base64
from pathlib import Path

st.set_page_config(page_title="Football Player Detection", layout="wide", initial_sidebar_state="expanded")


def get_img_as_base64(file_path: str) -> str:
    """Return base64 string for a local image file path."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def safe_load_yolo(model_path: str):
    """Load a YOLO model safely and return (model, error_message)."""
    try:
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def build_detection_grid(detections_imgs_list, pad_size=(80, 60, 3)):
    """
    Build a 2-row grid of detected player thumbnails.
    Returns a numpy uint8 image ready to pass to streamlit_image_coordinates.
    Handles zero, odd or even number of thumbnails.
    """
    if len(detections_imgs_list) == 0:
        # return a white placeholder image
        placeholder = np.ones((pad_size[0] * 2, pad_size[1] * 4, 3), dtype=np.uint8) * 255
        return placeholder.astype(np.uint8)

    # resize each to consistent size and ensure dtype
    thumbs = [cv2.resize(img, (pad_size[1], pad_size[0])) if (img.shape[0] != pad_size[0] or img.shape[1] != pad_size[1]) else img
              for img in detections_imgs_list]
    thumbs = [t.astype(np.uint8) for t in thumbs]

    # choose number per row (try to balance)
    num = len(thumbs)
    row_count = 2
    per_row = (num + 1) // row_count

    rows = []
    idx = 0
    for r in range(row_count):
        row_items = thumbs[idx: idx + per_row]
        idx += per_row
        # if this row has fewer items than per_row, pad with white images
        while len(row_items) < per_row:
            row_items.append(np.ones((pad_size[0], pad_size[1], 3), dtype=np.uint8) * 255)
        row_img = cv2.hconcat(row_items)
        rows.append(row_img)

    # If rows length < 2, pad another white row
    while len(rows) < 2:
        rows.append(np.ones_like(rows[0]) * 255)

    grid = cv2.vconcat(rows)
    return grid.astype(np.uint8)


def main():
    # ---- Background / sidebar image (optional) ----
    # If you want to set a sidebar background image, provide its local path here.
    # sidebar_image_path = "image.jpg"
    # if Path(sidebar_image_path).exists():
    #     img_b64 = get_img_as_base64(sidebar_image_path)
    #     page_bg_img = f"""
    #     <style>
    #     [data-testid="stSidebar"] > div:first-child {{
    #         background-image: url("data:image/png;base64,{img_b64}");
    #         background-position: center;
    #         background-repeat: no-repeat;
    #     }}
    #     </style>
    #     """
    #     st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Football Player Detection using Deep Learning")

    # ---- Model paths: adjust these to your actual files/folders ----
    # Avoid spaces in folder names (YOLO or OS might cause issues).
    default_players_model_path = "../models/Yolo8L_Players/weights/best.pt"
    default_keypoints_model_path = "../models/Yolo8M_Field_Keypoints/weights/best.pt"

    st.sidebar.header("Model configuration")

    players_model_path = st.sidebar.text_input("Players model path", value=default_players_model_path)
    keypoints_model_path = st.sidebar.text_input("Field keypoints model path", value=default_keypoints_model_path)

    # Attempt to load models (show helpful messages)
    with st.spinner("Loading models..."):
        model_players, err_players = safe_load_yolo(players_model_path)
        model_keypoints, err_keypoints = safe_load_yolo(keypoints_model_path)

    if err_players:
        st.sidebar.error(f"Could not load player detection model: {err_players}")
    else:
        st.sidebar.success("Player detection model loaded")

    if err_keypoints:
        st.sidebar.error(f"Could not load keypoints model: {err_keypoints}")
    else:
        st.sidebar.success("Field keypoints model loaded")

    # ---- Tabs ----
    tab1, tab2, tab3 = st.tabs(["Upload Video", "Parameters & Detection", "Project Info"])

    # ---- Demo videos mapping ----
    demo_vid_paths = {
        "Demo 1 (France vs Switzerland)": "./demo_vid_1.mp4",
        "Demo 2 (Chelsea vs ManCity)": "./demo_vid_2.mp4"
    }

    demo_team_info = {
        "Demo 1 (France vs Switzerland)": {
            "team1_name": "France",
            "team2_name": "Switzerland",
            "team1_p_color": '#1E2530',
            "team1_gk_color": '#F5FD15',
            "team2_p_color": '#FBFCFA',
            "team2_gk_color": '#B1FCC4'
        },
        "Demo 2 (Chelsea vs ManCity)": {
            "team1_name": "Chelsea",
            "team2_name": "Manchester City",
            "team1_p_color": '#29478A',
            "team1_gk_color": '#DC6258',
            "team2_p_color": '#90C8FF',
            "team2_gk_color": '#BCC703'
        }
    }

    # ---- Tab 1: Upload and pick frame for team color selection ----
    with tab1:
        st.subheader("How to use")
        st.markdown(
            """1. Upload a video or choose a demo video.
2. Enter the team names for the video.
3. Select a frame where players and goalkeepers from both teams are visible.
4. Click on player thumbnails to pick team colors, then adjust using the color pickers.
5. Go to the 'Parameters & Detection' tab and run detection.
6. If 'Save output' is selected, processed video will be saved to `outputs/`."""
        )
        st.warning("This app is designed for tactical/camera-overhead style videos.")

        demo_selected = st.radio("Select a demo video", options=list(demo_vid_paths.keys()), index=0, horizontal=True)

        input_video_file = st.file_uploader("Upload a video (mp4, mov, avi, m4v, asf)", type=['mp4', 'mov', 'avi', 'm4v', 'asf'])
        # prepare temporary file
        tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tempf_path = tempf.name

        # If user uploaded a file, write it to temp file. Otherwise use demo
        if input_video_file is not None:
            tempf.write(input_video_file.read())
            tempf.flush()
            st.text("Uploaded video preview")
            st.video(tempf_path)
        else:
            demo_vid_path = demo_vid_paths[demo_selected]
            if Path(demo_vid_path).exists():
                # copy demo file path into tempf by setting tempf_path to demo
                tempf.close()
                tempf_path = demo_vid_path
                st.text("Demo video preview")
                with open(tempf_path, "rb") as f:
                    st.video(f.read())
            else:
                st.error(f"Demo video not found at {demo_vid_path}. Please upload a video.")
                # stop here if no video
                st.stop()

        # Team names
        selected_team_info = demo_team_info.get(demo_selected, {})
        team1_name = st.text_input("Enter Team 1 name", value=selected_team_info.get("team1_name", "Team 1"))
        team2_name = st.text_input("Enter Team 2 name", value=selected_team_info.get("team2_name", "Team 2"))

        # read frame count and select frame
        cap_temp = cv2.VideoCapture(tempf_path)
        if not cap_temp.isOpened():
            st.error("Could not open video. Check the file and the path.")
            st.stop()

        frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            st.error("Video has zero frames or cannot read frame count.")
            cap_temp.release()
            st.stop()

        frame_nbr = st.slider("Select a frame index to pick team colors from", min_value=1, max_value=frame_count, value=1, step=1)
        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr - 1)  # zero-based
        success, frame = cap_temp.read()
        cap_temp.release()

        if not success or frame is None:
            st.error("Could not read the selected frame.")
            st.stop()

        # Run player detection on the selected frame (if model available)
        detections_imgs_list = []
        if model_players is not None:
            with st.spinner("Detecting players in selected frame..."):
                # ultralytics returns results; we guard against exceptions
                try:
                    results = model_players(frame, conf=0.7)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        bboxes = boxes.xyxy.cpu().numpy()
                        labels = boxes.cls.cpu().numpy()
                    else:
                        bboxes = np.array([])
                        labels = np.array([])

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for i, cls in enumerate(labels):
                        # class 0 = person in many YOLO datasets - adapt if your model differs
                        if int(cls) == 0 and bboxes.size != 0:
                            x1, y1, x2, y2 = map(int, bboxes[i, :4])
                            # safety clamp
                            h, w = frame_rgb.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            if x2 > x1 and y2 > y1:
                                obj_img = frame_rgb[y1:y2, x1:x2]
                                # if too small, skip
                                if obj_img.size == 0:
                                    continue
                                thumb = cv2.resize(obj_img, (60, 80))
                                detections_imgs_list.append(thumb)
                except Exception as e:
                    st.warning(f"Player detection on frame failed: {e}")
        else:
            st.info("Player detection model not loaded; skipping player thumbnails.")

        concat_det_imgs = build_detection_grid(detections_imgs_list)

        st.write("Detected player thumbnails (click to pick a color)")
        # streamlit_image_coordinates expects an image (RGB) and returns coordinates on click
        # value = streamlit_image_coordinates(concat_det_imgs, key="player_thumbs")

        st.image(concat_det_imgs, caption="Click on the image to pick a color", use_column_width=False)

        # Native Streamlit click handler
        if "click_x" not in st.session_state:
            st.session_state["click_x"] = None
        if "click_y" not in st.session_state:
            st.session_state["click_y"] = None

        # Use image-click event
        click = st.experimental_data_editor(
            pd.DataFrame({"x": [st.session_state["click_x"]], "y": [st.session_state["click_y"]]})
        )

        if click is not None:
            try:
                x = int(click.iloc[0]["x"])
                y = int(click.iloc[0]["y"])
                if not np.isnan(x) and not np.isnan(y):
                    picked_color = concat_det_imgs[y, x]
                    hex_color = '#%02x%02x%02x' % tuple(picked_color)
                    st.session_state[active_color] = hex_color
                    st.success(f"Set {active_color} to {hex_color}")
            except:
                pass



        # Color pickers - store chosen hex in session_state to persist
        if f"{team1_name} Player color" not in st.session_state:
            st.session_state[f"{team1_name} Player color"] = selected_team_info.get("team1_p_color", "#000000")
        if f"{team1_name} GK color" not in st.session_state:
            st.session_state[f"{team1_name} GK color"] = selected_team_info.get("team1_gk_color", "#ffffff")
        if f"{team2_name} Player color" not in st.session_state:
            st.session_state[f"{team2_name} Player color"] = selected_team_info.get("team2_p_color", "#000000")
        if f"{team2_name} GK color" not in st.session_state:
            st.session_state[f"{team2_name} GK color"] = selected_team_info.get("team2_gk_color", "#ffffff")

        radio_options = [f"{team1_name} Player color", f"{team1_name} GK color",
                         f"{team2_name} Player color", f"{team2_name} GK color"]
        active_color = st.radio("Choose which team color slot to set by clicking on thumbnails", options=radio_options, horizontal=True)

        if value is not None:
            # value has keys 'x', 'y' referencing the pixel clicked on concat_det_imgs
            picked_color = concat_det_imgs[value['y'], value['x'], :]
            hex_color = '#%02x%02x%02x' % tuple(picked_color)
            st.session_state[active_color] = hex_color
            st.success(f"Set {active_color} to {hex_color}")

        st.write("Use the color pickers below to fine-tune the selected colors.")
        cp1, cp2, cp3, cp4 = st.columns([1, 1, 1, 1])
        with cp1:
            team1_p_color = st.color_picker(label=f"{team1_name} player color", value=st.session_state[f"{team1_name} Player color"], key='t1p')
            st.session_state[f"{team1_name} Player color"] = team1_p_color
        with cp2:
            team1_gk_color = st.color_picker(label=f"{team1_name} goalkeeper color", value=st.session_state[f"{team1_name} GK color"], key='t1gk')
            st.session_state[f"{team1_name} GK color"] = team1_gk_color
        with cp3:
            team2_p_color = st.color_picker(label=f"{team2_name} player color", value=st.session_state[f"{team2_name} Player color"], key='t2p')
            st.session_state[f"{team2_name} Player color"] = team2_p_color
        with cp4:
            team2_gk_color = st.color_picker(label=f"{team2_name} goalkeeper color", value=st.session_state[f"{team2_name} GK color"], key='t2gk')
            st.session_state[f"{team2_name} GK color"] = team2_gk_color

        # create colors data structures for later use
        colors_dic, color_list_lab = create_colors_info(
            team1_name, st.session_state[f"{team1_name} Player color"], st.session_state[f"{team1_name} GK color"],
            team2_name, st.session_state[f"{team2_name} Player color"], st.session_state[f"{team2_name} GK color"]
        )

    # ---- Tab 2: Parameters and run detection ----
    with tab2:
        st.header("Detection Parameters")

        c1, c2 = st.columns(2)
        with c1:
            player_model_conf_thresh = st.slider('Player detection confidence threshold', min_value=0.0, max_value=1.0, value=0.6)
        with c2:
            keypoints_model_conf_thresh = st.slider('Field keypoints confidence threshold', min_value=0.0, max_value=1.0, value=0.7)

        # fixed or adjustable hyperparams
        keypoints_displacement_mean_tol = 7
        detection_hyper_params = {
            0: player_model_conf_thresh,
            1: keypoints_model_conf_thresh,
            2: keypoints_displacement_mean_tol
        }

        st.subheader("Overlay options")
        show_players = st.checkbox("Show player detections", value=True)
        show_ball_track = st.checkbox("Show ball track", value=True)
        show_pal = st.checkbox("Show team palette", value=True)
        show_keypoints = st.checkbox("Show field keypoints", value=False)

        plot_hyperparams = {
            0: show_keypoints,
            1: show_pal,
            2: show_ball_track,
            3: show_players
        }

        st.markdown("---")
        save_output = st.checkbox("Save output video", value=False)
        output_file_name = None
        if save_output:
            output_file_name = st.text_input("Output file name (without extension)", placeholder="result_video")

        st.markdown("---")
        # Ball tracking hyperparams (kept fixed for now)
        nbr_frames_no_ball_thresh = 30
        ball_track_dist_thresh = 100
        max_track_length = 35
        ball_track_hyperparams = {
            0: nbr_frames_no_ball_thresh,
            1: ball_track_dist_thresh,
            2: max_track_length
        }

        # Start/Stop buttons
        col_start, col_stop, _ = st.columns([1, 1, 6])
        start_detection = col_start.button("Start Detection", disabled=(model_players is None or model_keypoints is None))
        stop_detection = col_stop.button("Stop Detection")

    # ---- Detection runner (outside tabs so we can display frames) ----
    stframe = st.empty()

    # Open video capture
    cap = cv2.VideoCapture(tempf_path)
    status = False

    if start_detection and not stop_detection:
        st.info("Starting detection...")
        try:
            # Ensure models loaded
            if model_players is None or model_keypoints is None:
                st.error("Required models not loaded. Aborting detection.")
            else:
                status = detect(
                    cap,
                    stframe,
                    output_file_name,
                    save_output,
                    model_players,
                    model_keypoints,
                    detection_hyper_params,
                    ball_track_hyperparams,
                    plot_hyperparams,
                    num_pal_colors=3,
                    colors_dic=colors_dic,
                    color_list_lab=color_list_lab
                )
        except Exception as ex:
            st.error(f"Detection failed: {ex}")
        finally:
            try:
                cap.release()
            except Exception:
                pass

    else:
        # If user didn't start detection, ensure capture closed
        try:
            cap.release()
        except Exception:
            pass

    if status:
        st.success("Detection completed successfully!")

    # ---- Tab 3: Project info ----
    with tab3:
        st.subheader("Main features:")
        st.markdown(
            """1. Player, referee and ball detection.
2. Team prediction for each player.
3. Estimation of player and ball positions on a tactical map.
4. Ball tracking across frames."""
        )

        st.markdown(
            """made with ❤️ by Ajay Bind"""
        )
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
