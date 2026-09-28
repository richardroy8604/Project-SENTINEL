"""
ULTRON UI — Tactical Cyberpunk Dashboard (JARVIS / Iron Man Aesthetics)
========================================================================
Main tactical command center featuring:
  - Live optical surveillance feed via GPU raw texture (CH-01)
  - Holographic Quantum Arc Reactor Core (multi-ring animated AI visualizer)
  - Real-time Audio Frequency Spectrum & Dynamic Equalizer Ring
  - Dedicated Tactical Comms & Dialogue Stream (No scrolling required)
  - Optical Telemetry & Target Recognition Status Strip
  - Compact Tactical Event Audit Ticker & Hardware Subsystems Matrix
"""

import math
import time
import numpy as np
import dearpygui.dearpygui as dpg

import config


class Dashboard:
    """
    Tactical Dear PyGui Cyberpunk Dashboard for ULTRON.
    Renders high-FPS camera feed, holographic Arc Reactor AI core,
    prominent Comms Stream, and system audit telemetry.
    """

    def __init__(self):
        # Video texture dimensions
        self._video_width = config.VIDEO_DISPLAY_WIDTH
        self._video_height = config.VIDEO_DISPLAY_HEIGHT
        self._video_data = np.zeros(
            self._video_width * self._video_height * 3, dtype=np.float32
        )

        # Arc Reactor animation state
        self._blob_time = 0.0
        self._blob_speaking = False
        self._blob_intensity = 0.0
        self._security_state = config.SecurityState.MONITORING

        # Audio level state
        self._audio_level = 0.0

        # Performance & Telemetry state
        self._cam_fps = 0.0
        self._ui_fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()

        # Telemetry cache
        self._person_count = 0
        self._inference_ms = 0.0
        self._comms_count = 0

    def setup(self):
        """Initialize Dear PyGui context, register textures, theme, and full 3-column layout."""
        dpg.create_context()

        # ── Register Video Texture ──────────────────────────────────────
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=self._video_width,
                height=self._video_height,
                default_value=self._video_data,
                tag="video_texture",
                format=dpg.mvFormat_Float_rgb,
            )

        # ── Cyberpunk JARVIS Theme ──────────────────────────────────────
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Deep obsidian & midnight slate palette
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (10, 12, 18, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (15, 19, 28, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (18, 22, 34, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (26, 34, 52, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (225, 232, 245, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (36, 48, 72, 255))
                dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 26, 38, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (28, 36, 52, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (35, 46, 68, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (12, 14, 22, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (32, 44, 66, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (44, 60, 90, 255))
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (0, 229, 255, 255))
                # Cybernetic geometry rounding
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 2)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 5)
        dpg.bind_theme(global_theme)

        # ── Primary Window ─────────────────────────────────────────────
        with dpg.window(tag="primary_window"):

            # ── Top Tactical Status Bar ────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("[ U L T R O N ]", color=(240, 50, 50, 255))
                dpg.add_spacer(width=8)
                dpg.add_text("// TACTICAL AI SENTINEL", color=(95, 135, 175, 255))
                dpg.add_spacer(width=28)
                dpg.add_text("* SYSTEM ARMED", tag="status_text", color=(0, 230, 118, 255))
                dpg.add_spacer(width=24)
                dpg.add_text(
                    "UI: 60 FPS | CAM: 30 FPS",
                    tag="fps_text",
                    color=(115, 135, 165, 255),
                )
                dpg.add_spacer(width=24)
                dpg.add_text("CLUSTER: HIVEMQ TLS [ACTIVE]", color=(0, 200, 255, 200))

            dpg.add_spacer(height=3)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # ── Main 3-Column Workspace ────────────────────────────────
            with dpg.group(horizontal=True):

                # ──────────────────────────────────────────────────────────
                # COLUMN 1: Live Optical Feed & Target Telemetry (Left)
                # ──────────────────────────────────────────────────────────
                with dpg.child_window(width=656, height=542, border=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("OPTICAL SENSOR", color=(0, 229, 255, 255))
                        dpg.add_spacer(width=8)
                        dpg.add_text("// CH-01 (640x480 @ 30 FPS)", color=(85, 115, 150, 255))

                    dpg.add_spacer(height=2)
                    dpg.add_image("video_texture")
                    dpg.add_spacer(height=4)

                    # Optical Telemetry Mini-Strip
                    with dpg.group(horizontal=True):
                        dpg.add_text("TARGETS:", color=(100, 130, 160, 255))
                        dpg.add_text(
                            "0 DETECTED",
                            tag="optical_targets_text",
                            color=(115, 135, 165, 255),
                        )
                        dpg.add_spacer(width=16)
                        dpg.add_text("THREAT:", color=(100, 130, 160, 255))
                        dpg.add_text(
                            "NOMINAL",
                            tag="optical_threat_text",
                            color=(0, 229, 255, 255),
                        )
                        dpg.add_spacer(width=16)
                        dpg.add_text("OPTICS:", color=(100, 130, 160, 255))
                        dpg.add_text(
                            "CLEAR",
                            tag="optical_tamper_text",
                            color=(0, 230, 118, 255),
                        )

                dpg.add_spacer(width=8)

                # ──────────────────────────────────────────────────────────
                # COLUMN 2: Holographic Quantum Arc Reactor Core (Center)
                # ──────────────────────────────────────────────────────────
                with dpg.child_window(width=380, height=542, border=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("NEURAL CORE", color=(255, 61, 0, 255))
                        dpg.add_spacer(width=8)
                        dpg.add_text("// QUANTUM REACTOR", color=(255, 171, 0, 255))

                    # High-Detail Arc Reactor Drawing Canvas
                    with dpg.drawlist(width=360, height=295, tag="blob_canvas"):
                        pass

                    dpg.add_spacer(height=3)
                    dpg.add_separator()
                    dpg.add_spacer(height=3)

                    # Security State Readout
                    with dpg.group(horizontal=True):
                        dpg.add_text("SECURITY STATE:", color=(110, 135, 170, 255))
                        dpg.add_text(
                            "MONITORING",
                            tag="security_state_text",
                            color=(0, 229, 255, 255),
                        )

                    dpg.add_spacer(height=2)

                    # Telemetry Metrics Grid
                    with dpg.group(horizontal=True):
                        dpg.add_text("PERSONS:", color=(110, 135, 170, 255))
                        dpg.add_text("0", tag="person_count_text", color=(115, 135, 165, 255))
                        dpg.add_spacer(width=12)
                        dpg.add_text("DETECT:", color=(110, 135, 170, 255))
                        dpg.add_text("-- ms", tag="detect_ms_text", color=(100, 130, 160, 255))
                        dpg.add_spacer(width=12)
                        dpg.add_text("CAM:", color=(110, 135, 170, 255))
                        dpg.add_text("0", tag="cam_fps_text", color=(0, 230, 118, 255))

                    dpg.add_spacer(height=4)

                    # Microphone Sensor Level (VU Meter)
                    with dpg.group(horizontal=True):
                        dpg.add_text("MIC SENSOR:", color=(110, 135, 170, 255))
                        dpg.add_progress_bar(
                            tag="audio_level_bar",
                            default_value=0.0,
                            width=220,
                            overlay="0%",
                        )

                dpg.add_spacer(width=8)

                # ──────────────────────────────────────────────────────────
                # COLUMN 3: Tactical Comms Stream (Fills All Remaining Space!)
                # ──────────────────────────────────────────────────────────
                with dpg.child_window(width=-1, height=542, border=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("TACTICAL COMMS STREAM", color=(0, 229, 255, 255))
                        dpg.add_spacer(width=8)
                        dpg.add_text("// REAL-TIME TRANSCRIPT", color=(85, 115, 150, 255))

                    dpg.add_spacer(height=4)

                    # Active Visitor Speech Card (No scrolling required!)
                    with dpg.child_window(width=-1, height=112, border=True):
                        with dpg.group(horizontal=True):
                            dpg.add_text("[* VISITOR // ACOUSTIC INPUT]", color=(0, 215, 255, 255))
                        dpg.add_spacer(height=2)
                        dpg.add_text(
                            "Awaiting acoustic input...",
                            tag="last_speech_text",
                            color=(235, 242, 255, 255),
                            wrap=0,
                        )

                    dpg.add_spacer(height=4)

                    # Active Ultron Response Card (No scrolling required!)
                    with dpg.child_window(width=-1, height=122, border=True):
                        with dpg.group(horizontal=True):
                            dpg.add_text("[* ULTRON SENTINEL // SYNTHESIS]", color=(255, 175, 0, 255))
                        dpg.add_spacer(height=2)
                        dpg.add_text(
                            "Sector perimeter armed and nominal.",
                            tag="ultron_reply_text",
                            color=(255, 225, 135, 255),
                            wrap=0,
                        )

                    dpg.add_spacer(height=4)

                    # Chronological Dialogue Stream
                    with dpg.group(horizontal=True):
                        dpg.add_text("COMMS HISTORY", color=(110, 135, 170, 255))
                        dpg.add_spacer(width=8)
                        dpg.add_text("// LOGGED TURNS", color=(75, 95, 125, 255))

                    with dpg.child_window(
                        width=-1, height=-1, tag="comms_history_window", border=False
                    ):
                        dpg.add_text(
                            "[SYS] Acoustic & Synthesis bus synchronized.",
                            color=(75, 100, 130, 255),
                        )

            dpg.add_spacer(height=4)

            # ── Bottom: Tactical Event Ticker & Hardware Subsystems Strip ──
            with dpg.child_window(width=-1, height=105, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("TACTICAL AUDIT TICKER", color=(140, 155, 195, 255))
                    dpg.add_spacer(width=8)
                    dpg.add_text("// SYSTEM EVENTS", color=(80, 100, 130, 255))

                # Compact Event Ticker (No more massive raw text box!)
                dpg.add_input_text(
                    tag="event_log",
                    multiline=True,
                    readonly=True,
                    height=45,
                    width=-1,
                    default_value="[ULTRON] Core security sentinel online. Awaiting visual stream...\n",
                )

                dpg.add_spacer(height=2)

                # Hardware Subsystems Matrix
                with dpg.group(horizontal=True):
                    dpg.add_text("* CAM: 30 FPS", color=(0, 230, 118, 255))
                    dpg.add_spacer(width=10)
                    dpg.add_text("* YOLO11s: RTX GPU", color=(0, 230, 118, 255))
                    dpg.add_spacer(width=10)
                    dpg.add_text("* SILERO VAD: ARMED", color=(0, 230, 118, 255))
                    dpg.add_spacer(width=10)
                    dpg.add_text("* WHISPER STT: READY", color=(0, 230, 118, 255))
                    dpg.add_spacer(width=10)
                    dpg.add_text("* GROQ LLM: ONLINE", color=(0, 230, 118, 255))
                    dpg.add_spacer(width=10)
                    dpg.add_text("* VOICE SYNTH: ACTIVE", color=(0, 229, 255, 255))
                    dpg.add_spacer(width=10)
                    dpg.add_text("* HIVEMQ CLOUD: CONNECTED", color=(0, 229, 255, 255))

        # ── Viewport Configuration ─────────────────────────────────────
        dpg.create_viewport(
            title=config.WINDOW_TITLE,
            width=config.WINDOW_WIDTH,
            height=config.WINDOW_HEIGHT,
            resizable=True,
            vsync=True,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)

    def update_video(self, rgb_frame: np.ndarray):
        """Update the video texture with a new RGB camera frame."""
        if (
            rgb_frame.shape[1] != self._video_width
            or rgb_frame.shape[0] != self._video_height
        ):
            import cv2
            rgb_frame = cv2.resize(
                rgb_frame, (self._video_width, self._video_height)
            )

        self._video_data = (
            rgb_frame.astype(np.float32).ravel() / 255.0
        )
        dpg.set_value("video_texture", self._video_data)

    def update_camera_fps(self, fps: float):
        """Update displayed camera FPS."""
        self._cam_fps = fps
        dpg.set_value("cam_fps_text", f"{fps:.1f}")

    def update_detection_info(self, person_count: int, inference_ms: float):
        """Update person count, detection latency, and optical telemetry badges."""
        self._person_count = person_count
        self._inference_ms = inference_ms

        if person_count == 0:
            count_color = (115, 135, 165, 255)
            opt_text = "0 DETECTED"
        else:
            count_color = (0, 230, 118, 255)
            opt_text = f"{person_count} DETECTED"

        dpg.set_value("person_count_text", str(person_count))
        dpg.configure_item("person_count_text", color=count_color)
        dpg.set_value("optical_targets_text", opt_text)
        dpg.configure_item("optical_targets_text", color=count_color)

        dpg.set_value("detect_ms_text", f"{inference_ms:.1f} ms")

    def update_audio_level(self, level: float):
        """Update microphone sensor level (0.0 to 1.0)."""
        self._audio_level = max(0.0, min(1.0, level))

    def update_last_speech(self, text: str):
        """Update last recognized speech card and append to comms history."""
        cleaned = text.strip()
        if not cleaned:
            return
        dpg.set_value("last_speech_text", f"\"{cleaned}\"")

        # Append to comms history
        t_str = time.strftime("%H:%M:%S")
        if dpg.does_item_exist("comms_history_window"):
            dpg.add_text(
                f"[{t_str}] VISITOR: \"{cleaned}\"",
                color=(0, 215, 255, 255),
                parent="comms_history_window",
            )
            self._comms_count += 1
            if self._comms_count > 35:
                children = dpg.get_item_children("comms_history_window", 1)
                if children and len(children) > 1:
                    dpg.delete_item(children[0])
            dpg.set_y_scroll("comms_history_window", -1.0)

    def update_ultron_reply(self, text: str):
        """Update ULTRON's latest reply card and append to comms history."""
        cleaned = text.strip()
        if not cleaned:
            return
        dpg.set_value("ultron_reply_text", f"\"{cleaned}\"")

        # Append to comms history
        t_str = time.strftime("%H:%M:%S")
        if dpg.does_item_exist("comms_history_window"):
            dpg.add_text(
                f"[{t_str}] ULTRON: \"{cleaned}\"",
                color=(255, 185, 45, 255),
                parent="comms_history_window",
            )
            self._comms_count += 1
            if self._comms_count > 35:
                children = dpg.get_item_children("comms_history_window", 1)
                if children and len(children) > 1:
                    dpg.delete_item(children[0])
            dpg.set_y_scroll("comms_history_window", -1.0)

    def set_security_state(self, state: str):
        """Update security state indicators, colors, and threat badge."""
        self._security_state = state
        color_map = {
            config.SecurityState.IDLE: (115, 135, 165, 255),
            config.SecurityState.MONITORING: (0, 229, 255, 255),
            config.SecurityState.ATTENTION: (255, 171, 0, 255),
            config.SecurityState.SUSPICIOUS: (255, 23, 68, 255),
        }
        color = color_map.get(state, (200, 200, 200, 255))
        dpg.set_value("security_state_text", state)
        dpg.configure_item("security_state_text", color=color)

        threat_map = {
            config.SecurityState.IDLE: ("STANDBY", (115, 135, 165, 255)),
            config.SecurityState.MONITORING: ("NOMINAL", (0, 229, 255, 255)),
            config.SecurityState.ATTENTION: ("ELEVATED", (255, 171, 0, 255)),
            config.SecurityState.SUSPICIOUS: ("CRITICAL", (255, 23, 68, 255)),
        }
        lbl, th_col = threat_map.get(state, ("NOMINAL", (0, 229, 255, 255)))
        dpg.set_value("optical_threat_text", lbl)
        dpg.configure_item("optical_threat_text", color=th_col)

    def log_event(self, message: str):
        """Append an event to the compact tactical audit ticker."""
        if not dpg.does_item_exist("event_log"):
            return
        current = dpg.get_value("event_log")
        timestamp = time.strftime("%H:%M:%S")
        lines = (current + f"[{timestamp}] {message}\n").splitlines()
        # Keep recent 25 lines to prevent memory bloating
        trimmed = "\n".join(lines[-25:]) + "\n"
        dpg.set_value("event_log", trimmed)

        # Update tamper badge if message indicates camera tamper
        if "tamper detected" in message.lower():
            dpg.set_value("optical_tamper_text", "OBSTRUCTED")
            dpg.configure_item("optical_tamper_text", color=(255, 23, 68, 255))
        elif "tamper cleared" in message.lower():
            dpg.set_value("optical_tamper_text", "CLEAR")
            dpg.configure_item("optical_tamper_text", color=(0, 230, 118, 255))

    def set_blob_speaking(self, speaking: bool, intensity: float = 0.5):
        """Set speaking state for dynamic Arc Reactor modulation."""
        self._blob_speaking = speaking
        self._blob_intensity = max(0.0, min(1.0, intensity))

    def _draw_blob(self):
        """
        Draw the high-detail Holographic Arc Reactor AI Core (JARVIS style).
        Features:
          - Rotating outer compass tick ring with cardinal degree marks
          - Counter-rotating segmented HUD aperture arcs
          - Dynamic radial frequency equalizer wave bars reactive to audio
          - Internal titanium turbine aperture blades
          - Multi-layered plasma glow sphere with breathing sine modulation
          - Anamorphic white-hot core lens flare
          - Orbiting tachyon data points with crosshairs
          - Cybernetic technical status overlays
        """
        dpg.delete_item("blob_canvas", children_only=True)

        t = self._blob_time
        cx, cy = 180, 142

        # State color selection
        color_map = {
            config.SecurityState.IDLE: (65, 125, 185),
            config.SecurityState.MONITORING: (0, 229, 255),
            config.SecurityState.ATTENTION: (255, 171, 0),
            config.SecurityState.SUSPICIOUS: (255, 23, 68),
        }
        base_r, base_g, base_b = color_map.get(
            self._security_state, (0, 229, 255)
        )

        # Base breathing radius
        breath_speed = 1.3 if not self._blob_speaking else 3.2
        breath_amplitude = 6 if not self._blob_speaking else 15
        base_radius = 48 + breath_amplitude * math.sin(t * breath_speed)

        if self._blob_speaking:
            base_radius += 10 * math.sin(t * 8.5) * self._blob_intensity

        # ── 1. Outer Compass & Coordinate Degree Ticks (R ≈ 124) ──────
        num_ticks = 36
        for i in range(num_ticks):
            angle = t * 0.15 + i * (2 * math.pi / num_ticks)
            is_major = (i % 3 == 0)
            r_inner = 114 if is_major else 118
            r_outer = 124
            x1 = cx + r_inner * math.cos(angle)
            y1 = cy + r_inner * math.sin(angle)
            x2 = cx + r_outer * math.cos(angle)
            y2 = cy + r_outer * math.sin(angle)
            alpha = 190 if is_major else 70
            dpg.draw_line(
                (x1, y1),
                (x2, y2),
                color=(base_r, base_g, base_b, alpha),
                thickness=1.5 if is_major else 1.0,
                parent="blob_canvas",
            )

        # ── 2. Counter-Rotating Segmented Aperture Ring (R ≈ 104) ────
        num_segs = 6
        for i in range(num_segs):
            a_start = -t * 0.35 + i * (2 * math.pi / num_segs)
            a_end = a_start + (math.pi / num_segs) * 0.72
            pts = []
            for step in range(5):
                ang = a_start + (a_end - a_start) * (step / 4.0)
                pts.append((cx + 104 * math.cos(ang), cy + 104 * math.sin(ang)))
            for k in range(len(pts) - 1):
                dpg.draw_line(
                    pts[k],
                    pts[k + 1],
                    color=(base_r, base_g, base_b, 170),
                    thickness=2.0,
                    parent="blob_canvas",
                )

        # ── 3. Radial Frequency Equalizer Wave Bars (R ≈ 86) ──────────
        num_bars = 28
        eff_audio = max(self._audio_level, 0.75 if self._blob_speaking else 0.0)
        for i in range(num_bars):
            ang = i * (2 * math.pi / num_bars)
            bar_pulse = abs(math.sin(t * 5.0 + i * 0.65))
            bar_len = 3 + (eff_audio * 28 + (12 if self._blob_speaking else 0)) * bar_pulse
            x1 = cx + 84 * math.cos(ang)
            y1 = cy + 84 * math.sin(ang)
            x2 = cx + (84 + bar_len) * math.cos(ang)
            y2 = cy + (84 + bar_len) * math.sin(ang)
            bar_alpha = int(100 + 155 * (bar_len / 40.0))
            dpg.draw_line(
                (x1, y1),
                (x2, y2),
                color=(min(255, base_r + 40), min(255, base_g + 40), min(255, base_b + 40), bar_alpha),
                thickness=2.0,
                parent="blob_canvas",
            )

        # ── 4. Internal Turbine Aperture Vanes (R ≈ 64) ───────────────
        num_vanes = 10
        for i in range(num_vanes):
            ang = t * 0.45 + i * (2 * math.pi / num_vanes)
            x1 = cx + 52 * math.cos(ang)
            y1 = cy + 52 * math.sin(ang)
            x2 = cx + 70 * math.cos(ang + 0.22)
            y2 = cy + 70 * math.sin(ang + 0.22)
            dpg.draw_line(
                (x1, y1),
                (x2, y2),
                color=(base_r, base_g, base_b, 130),
                thickness=1.8,
                parent="blob_canvas",
            )

        # ── 5. Multi-Layered Plasma Glow Spheres ──────────────────────
        # Outer atmospheric halos
        for i in range(4, 0, -1):
            halo_r = base_radius + i * 11
            halo_alpha = int(14 - i * 2)
            wobble = 2 * math.sin(t * 1.1 + i)
            dpg.draw_circle(
                center=(cx + wobble, cy),
                radius=halo_r,
                color=(base_r, base_g, base_b, halo_alpha),
                fill=(base_r, base_g, base_b, halo_alpha),
                parent="blob_canvas",
            )

        # Core outer shell
        dpg.draw_circle(
            center=(cx, cy),
            radius=base_radius * 0.72,
            color=(base_r, base_g, base_b, 180),
            fill=(base_r, base_g, base_b, 70),
            parent="blob_canvas",
        )

        # Bright inner reactor core
        inner_pulse = 4 * math.sin(t * 2.8)
        dpg.draw_circle(
            center=(cx, cy),
            radius=base_radius * 0.38 + inner_pulse,
            color=(min(255, base_r + 90), min(255, base_g + 90), min(255, base_b + 90), 220),
            fill=(min(255, base_r + 110), min(255, base_g + 110), min(255, base_b + 110), 140),
            parent="blob_canvas",
        )

        # White-hot singularity center
        dpg.draw_circle(
            center=(cx, cy),
            radius=7 + 2 * math.sin(t * 4.0),
            color=(255, 255, 255, 255),
            fill=(255, 255, 255, 240),
            parent="blob_canvas",
        )

        # Anamorphic horizontal lens flare line
        flare_w = 42 + 10 * math.sin(t * 3.5)
        dpg.draw_line(
            (cx - flare_w, cy),
            (cx + flare_w, cy),
            color=(255, 255, 255, 140),
            thickness=1.5,
            parent="blob_canvas",
        )

        # ── 6. Orbiting Tachyon Data Drones ───────────────────────────
        for i in range(3):
            ang = t * (0.85 + i * 0.35) + i * 2.15
            orbit_r = 92 + 16 * math.sin(t * 1.4 + i)
            px = cx + orbit_r * math.cos(ang)
            py = cy + orbit_r * math.sin(ang)
            dpg.draw_circle(
                center=(px, py),
                radius=3,
                color=(255, 255, 255, 230),
                fill=(base_r, base_g, base_b, 220),
                parent="blob_canvas",
            )
            # Small crosshair tick on the drone
            dpg.draw_line((px - 5, py), (px + 5, py), color=(base_r, base_g, base_b, 140), parent="blob_canvas")
            dpg.draw_line((px, py - 5), (px, py + 5), color=(base_r, base_g, base_b, 140), parent="blob_canvas")

        # ── 7. Technical HUD Canvas Typography ────────────────────────
        dpg.draw_text(
            pos=(12, 10),
            text="MATRIX: 44.1kHz",
            color=(base_r, base_g, base_b, 170),
            size=12,
            parent="blob_canvas",
        )
        dpg.draw_text(
            pos=(260, 10),
            text="CORE: SYNC",
            color=(base_r, base_g, base_b, 170),
            size=12,
            parent="blob_canvas",
        )

        # Dynamic Status Sub-badge under the reactor
        if self._blob_speaking:
            status_badge = "[* VOCAL TRANSMISSION ACTIVE]"
            badge_col = (255, 185, 45, 230)
        elif self._audio_level > 0.08:
            status_badge = "[* ACOUSTIC SENSOR ACTIVE]"
            badge_col = (0, 230, 255, 230)
        else:
            status_badge = f"[* PERIMETER {self._security_state}]"
            badge_col = (base_r, base_g, base_b, 200)

        dpg.draw_text(
            pos=(cx - len(status_badge) * 3.5, 274),
            text=status_badge,
            color=badge_col,
            size=12,
            parent="blob_canvas",
        )

    def render_frame(self) -> bool:
        """
        Render one frame of the tactical UI.

        Returns:
            False if the window was closed, True otherwise.
        """
        if not dpg.is_dearpygui_running():
            return False

        # Update animation clock
        self._blob_time = time.perf_counter()

        # Re-draw the Holographic Arc Reactor
        self._draw_blob()

        # UI FPS tracking
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._ui_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now
            dpg.set_value(
                "fps_text",
                f"UI: {self._ui_fps:.0f} FPS | CAM: {self._cam_fps:.0f} FPS",
            )
            dpg.set_value("status_text", "* SYSTEM ARMED")

        # Smooth microphone VU meter
        pct = int(self._audio_level * 100)
        dpg.set_value("audio_level_bar", self._audio_level)
        dpg.configure_item("audio_level_bar", overlay=f"{pct}%")

        # Reactive intensity pulse when listening to speech
        if not self._blob_speaking:
            if self._audio_level > 0.05:
                self._blob_intensity = min(1.0, self._audio_level * 1.6)
            else:
                self._blob_intensity = max(0.0, self._blob_intensity - 0.04)

        dpg.render_dearpygui_frame()
        return True

    def shutdown(self):
        """Clean up Dear PyGui resources."""
        dpg.destroy_context()
        print("[ULTRON UI] Tactical dashboard closed.")
