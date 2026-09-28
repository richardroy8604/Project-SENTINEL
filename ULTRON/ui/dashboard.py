"""
ULTRON UI — Dear PyGui Dashboard
==================================
Main dashboard featuring:
  - Live webcam feed via GPU raw texture
  - Animated AI blob visualization (JARVIS-style pulsating orb)
  - System state indicators
  - FPS counter
  - Event log (placeholder for future stages)

The blob is a pulsating, glowing sphere that:
  - Breathes with sine-wave radius modulation
  - Changes color based on security state
  - Will react to ULTRON's voice in later stages
"""

import math
import time
import numpy as np
import dearpygui.dearpygui as dpg

import config


class Dashboard:
    """
    Dear PyGui dashboard for ULTRON.
    Renders the webcam feed, AI blob, status indicators, and event log.
    """

    def __init__(self):
        # Video texture data — will be updated each frame
        self._video_width = config.VIDEO_DISPLAY_WIDTH
        self._video_height = config.VIDEO_DISPLAY_HEIGHT
        self._video_data = np.zeros(
            self._video_width * self._video_height * 3, dtype=np.float32
        )

        # Blob animation state
        self._blob_time = 0.0
        self._blob_speaking = False       # Will pulse more when ULTRON speaks
        self._blob_intensity = 0.0        # Audio-reactive intensity (0.0 - 1.0)
        self._security_state = config.SecurityState.MONITORING

        # Audio level state
        self._audio_level = 0.0

        # FPS display
        self._cam_fps = 0.0
        self._ui_fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()

    def setup(self):
        """Initialize Dear PyGui context, create all UI elements."""
        dpg.create_context()

        # ── Register video texture ──────────────────────────────────────
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=self._video_width,
                height=self._video_height,
                default_value=self._video_data,
                tag="video_texture",
                format=dpg.mvFormat_Float_rgb,
            )

        # ── Theme: Dark cyberpunk ───────────────────────────────────────
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (12, 12, 18, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (20, 20, 30, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (30, 30, 50, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (18, 18, 26, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 235, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (40, 40, 60, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (25, 25, 40, 255))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.bind_theme(global_theme)

        # ── Main window ────────────────────────────────────────────────
        with dpg.window(tag="primary_window"):

            # Top bar — Title + Status
            with dpg.group(horizontal=True):
                dpg.add_text("U L T R O N", color=(200, 50, 50, 255))
                dpg.add_spacer(width=20)
                dpg.add_text("", tag="status_text", color=(0, 200, 100, 255))
                dpg.add_spacer(width=20)
                dpg.add_text("", tag="fps_text", color=(100, 100, 140, 255))

            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            # Main content: Video (left) + Blob & Info (right)
            with dpg.group(horizontal=True):

                # ── LEFT: Video Feed ────────────────────────────────────
                with dpg.child_window(
                    width=self._video_width + 20,
                    height=self._video_height + 50,
                    border=True,
                ):
                    dpg.add_text(
                        "LIVE FEED", color=(0, 180, 255, 255)
                    )
                    dpg.add_image("video_texture")

                dpg.add_spacer(width=12)

                # ── RIGHT: Blob + Info Panel ────────────────────────────
                with dpg.child_window(width=-1, height=self._video_height + 50):

                    # AI Blob canvas
                    dpg.add_text(
                        "ULTRON CORE", color=(200, 50, 50, 255)
                    )
                    with dpg.drawlist(
                        width=350, height=280, tag="blob_canvas"
                    ):
                        pass  # Drawn dynamically in _draw_blob()

                    dpg.add_spacer(height=8)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)

                    # Security State
                    with dpg.group(horizontal=True):
                        dpg.add_text("STATE:", color=(100, 100, 140, 255))
                        dpg.add_text(
                            "MONITORING",
                            tag="security_state_text",
                            color=(0, 180, 255, 255),
                        )

                    # Person count
                    with dpg.group(horizontal=True):
                        dpg.add_text("PERSONS:", color=(100, 100, 140, 255))
                        dpg.add_text(
                            "0",
                            tag="person_count_text",
                            color=(0, 200, 100, 255),
                        )

                    # Detection latency
                    with dpg.group(horizontal=True):
                        dpg.add_text("DETECT:", color=(100, 100, 140, 255))
                        dpg.add_text(
                            "-- ms",
                            tag="detect_ms_text",
                            color=(100, 100, 140, 255),
                        )

                    # Camera FPS
                    with dpg.group(horizontal=True):
                        dpg.add_text("CAM FPS:", color=(100, 100, 140, 255))
                        dpg.add_text("0", tag="cam_fps_text", color=(0, 200, 100, 255))

                    dpg.add_spacer(height=4)
                    dpg.add_separator()
                    dpg.add_spacer(height=4)

                    # Audio VU Meter
                    with dpg.group(horizontal=True):
                        dpg.add_text("MIC LEVEL:", color=(100, 100, 140, 255))
                        dpg.add_progress_bar(
                            tag="audio_level_bar",
                            default_value=0.0,
                            width=180,
                            overlay="0%",
                        )

                    # Last Recognized Speech
                    dpg.add_spacer(height=2)
                    dpg.add_text("HEARD:", color=(100, 100, 140, 255))
                    dpg.add_text(
                        "...",
                        tag="last_speech_text",
                        color=(0, 220, 255, 255),
                        wrap=330,
                    )

                    # ULTRON's Reply
                    dpg.add_spacer(height=2)
                    dpg.add_text("ULTRON:", color=(255, 160, 40, 255))
                    dpg.add_text(
                        "...",
                        tag="ultron_reply_text",
                        color=(255, 220, 120, 255),
                        wrap=330,
                    )

            dpg.add_spacer(height=8)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # ── Bottom: Event Log ───────────────────────────────────────
            dpg.add_text("EVENT LOG", color=(100, 100, 140, 255))
            dpg.add_input_text(
                tag="event_log",
                multiline=True,
                readonly=True,
                height=120,
                width=-1,
                default_value="[ULTRON] System initialized. Awaiting visual input...\n",
            )

        # ── Viewport ───────────────────────────────────────────────────
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
        """
        Update the video texture with a new RGB frame.

        Args:
            rgb_frame: numpy array of shape (H, W, 3), dtype uint8, RGB color order.
        """
        # Resize if needed
        if (
            rgb_frame.shape[1] != self._video_width
            or rgb_frame.shape[0] != self._video_height
        ):
            import cv2
            rgb_frame = cv2.resize(
                rgb_frame, (self._video_width, self._video_height)
            )

        # Convert uint8 [0-255] to float32 [0.0-1.0] and flatten
        self._video_data = (
            rgb_frame.astype(np.float32).ravel() / 255.0
        )
        dpg.set_value("video_texture", self._video_data)

    def update_camera_fps(self, fps: float):
        """Update the displayed camera FPS."""
        self._cam_fps = fps
        dpg.set_value("cam_fps_text", f"{fps:.1f}")

    def update_detection_info(self, person_count: int, inference_ms: float):
        """Update person count and detection latency display."""
        # Person count with color coding
        if person_count == 0:
            color = (100, 100, 140, 255)  # Dim gray — no one
        else:
            color = (0, 255, 100, 255)    # Green — persons detected

        dpg.set_value("person_count_text", str(person_count))
        dpg.configure_item("person_count_text", color=color)

        # Inference latency
        dpg.set_value("detect_ms_text", f"{inference_ms:.1f} ms")

    def update_audio_level(self, level: float):
        """Update microphone VU meter level (0.0 to 1.0)."""
        self._audio_level = max(0.0, min(1.0, level))

    def update_last_speech(self, text: str):
        """Update last recognized speech text."""
        dpg.set_value("last_speech_text", f"\"{text}\"")

    def update_ultron_reply(self, text: str):
        """Update ULTRON's latest reply text."""
        dpg.set_value("ultron_reply_text", f"\"{text}\"")



    def set_security_state(self, state: str):
        """Update the displayed security state and blob color."""
        self._security_state = state
        color_map = {
            config.SecurityState.IDLE: (100, 100, 140, 255),
            config.SecurityState.MONITORING: (0, 180, 255, 255),
            config.SecurityState.ATTENTION: (255, 180, 0, 255),
            config.SecurityState.SUSPICIOUS: (255, 30, 30, 255),
        }
        color = color_map.get(state, (200, 200, 200, 255))
        dpg.set_value("security_state_text", state)
        dpg.configure_item("security_state_text", color=color)

    def log_event(self, message: str):
        """Append a message to the event log."""
        current = dpg.get_value("event_log")
        timestamp = time.strftime("%H:%M:%S")
        dpg.set_value("event_log", f"{current}[{timestamp}] {message}\n")

    def set_blob_speaking(self, speaking: bool, intensity: float = 0.5):
        """Set whether ULTRON is currently speaking (affects blob animation)."""
        self._blob_speaking = speaking
        self._blob_intensity = max(0.0, min(1.0, intensity))

    def _draw_blob(self):
        """
        Draw the animated AI blob — a pulsating, glowing orb.

        The orb is rendered using multiple concentric circles with
        varying opacity and sine-wave radius modulation to create
        a breathing, living effect.
        """
        dpg.delete_item("blob_canvas", children_only=True)

        t = self._blob_time
        cx, cy = 175, 140  # Center of the drawlist

        # Color based on security state
        color_map = {
            config.SecurityState.IDLE: (100, 100, 160),
            config.SecurityState.MONITORING: (0, 140, 255),
            config.SecurityState.ATTENTION: (255, 160, 0),
            config.SecurityState.SUSPICIOUS: (255, 30, 30),
        }
        base_r, base_g, base_b = color_map.get(
            self._security_state, (0, 140, 255)
        )

        # Base radius with breathing animation
        breath_speed = 1.2 if not self._blob_speaking else 3.0
        breath_amplitude = 8 if not self._blob_speaking else 18
        base_radius = config.BLOB_RADIUS + breath_amplitude * math.sin(
            t * breath_speed
        )

        # Speaking pulse overlay
        if self._blob_speaking:
            base_radius += 10 * math.sin(t * 8.0) * self._blob_intensity

        # ── Outer glow layers (large, faint circles) ──────────────────
        for i in range(5, 0, -1):
            glow_r = base_radius + i * 14
            alpha = int(12 - i * 2)
            # Slight sine wobble per layer for organic feel
            wobble = 3 * math.sin(t * 0.7 + i * 1.3)
            dpg.draw_circle(
                center=(cx + wobble, cy),
                radius=glow_r,
                color=(base_r, base_g, base_b, alpha),
                fill=(base_r, base_g, base_b, alpha),
                parent="blob_canvas",
            )

        # ── Core orb (solid, bright) ──────────────────────────────────
        dpg.draw_circle(
            center=(cx, cy),
            radius=base_radius * 0.65,
            color=(base_r, base_g, base_b, 160),
            fill=(base_r, base_g, base_b, 80),
            parent="blob_canvas",
        )

        # ── Inner bright core ─────────────────────────────────────────
        inner_pulse = 5 * math.sin(t * 2.5)
        dpg.draw_circle(
            center=(cx, cy),
            radius=base_radius * 0.3 + inner_pulse,
            color=(
                min(255, base_r + 80),
                min(255, base_g + 80),
                min(255, base_b + 80),
                200,
            ),
            fill=(
                min(255, base_r + 100),
                min(255, base_g + 100),
                min(255, base_b + 100),
                120,
            ),
            parent="blob_canvas",
        )

        # ── Hotspot (white center) ────────────────────────────────────
        hot_pulse = 3 * math.sin(t * 3.5)
        dpg.draw_circle(
            center=(cx, cy),
            radius=base_radius * 0.12 + hot_pulse,
            color=(255, 255, 255, 180),
            fill=(255, 255, 255, 100),
            parent="blob_canvas",
        )

        # ── Orbiting particles ────────────────────────────────────────
        num_particles = 6 if not self._blob_speaking else 10
        for i in range(num_particles):
            angle = t * (0.8 + i * 0.15) + i * (2 * math.pi / num_particles)
            orbit_r = base_radius * 0.75 + 10 * math.sin(t * 1.3 + i)
            px = cx + orbit_r * math.cos(angle)
            py = cy + orbit_r * math.sin(angle)
            p_size = 3 + 2 * math.sin(t * 2.0 + i * 0.8)
            dpg.draw_circle(
                center=(px, py),
                radius=p_size,
                color=(
                    min(255, base_r + 60),
                    min(255, base_g + 60),
                    min(255, base_b + 60),
                    180,
                ),
                fill=(
                    min(255, base_r + 80),
                    min(255, base_g + 80),
                    min(255, base_b + 80),
                    120,
                ),
                parent="blob_canvas",
            )

        # ── State label under the orb ─────────────────────────────────
        state_text = self._security_state
        dpg.draw_text(
            pos=(cx - len(state_text) * 4, cy + base_radius * 0.85 + 20),
            text=state_text,
            color=(base_r, base_g, base_b, 200),
            size=16,
            parent="blob_canvas",
        )

    def render_frame(self) -> bool:
        """
        Render one frame of the UI.

        Returns:
            False if the window was closed, True otherwise.
        """
        if not dpg.is_dearpygui_running():
            return False

        # Update animation time
        self._blob_time = time.perf_counter()

        # Draw the animated blob
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
            dpg.set_value("status_text", "● ONLINE")

        # Smooth audio meter update
        pct = int(self._audio_level * 100)
        dpg.set_value("audio_level_bar", self._audio_level)
        dpg.configure_item("audio_level_bar", overlay=f"{pct}%")

        # Audio-reactive pulse on the orb when hearing speech
        if not self._blob_speaking:
            if self._audio_level > 0.05:
                self._blob_intensity = min(1.0, self._audio_level * 1.5)
            else:
                self._blob_intensity = max(0.0, self._blob_intensity - 0.03)

        dpg.render_dearpygui_frame()
        return True

    def shutdown(self):
        """Clean up Dear PyGui resources."""
        dpg.destroy_context()
        print("[ULTRON UI] Dashboard closed.")
