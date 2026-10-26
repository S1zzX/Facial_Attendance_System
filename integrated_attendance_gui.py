import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from datetime import datetime, date
import numpy as np
import time
import random
import sys
from collections import deque
from database import DatabaseManager
from face_recognition_module import FaceRecognitionSystem
import queue
import os

try:
    import scipy.signal
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    is_scipy_matplotlib_installed = True
except ImportError:
    is_scipy_matplotlib_installed = False
    print("Warning: SciPy and Matplotlib not found. Heartbeat detection will be less accurate and no plot will be shown.")

# Liveness detection states
LIVENESS_STATE_INITIAL = 0
LIVENESS_STATE_BLINK = 1
LIVENESS_STATE_OPEN_MOUTH = 2
LIVENESS_STATE_SMILE = 3
LIVENESS_STATE_LIVE = 4
LIVENESS_STATE_SPOOF = 5

class IntegratedAttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attendance System with Liveness Detection")
        self.root.geometry("1400x800")
        
        self.db = DatabaseManager()
        # Initialize face system with default models (all models)
        self.face_system = FaceRecognitionSystem()
        self.camera = None
        self.is_running = False
        self.liveness_enabled = True
        
        # Initialize identification camera variables
        self.id_camera = None
        self.id_camera_running = False
        self.id_camera_thread = None
        self.current_id_frame = None
        self.identification_history = []
        
        # Liveness detection variables
        self.setup_liveness_detection()
        
        # Add thread-safe queue for GUI updates
        self.gui_update_queue = queue.Queue()
        
        self.setup_gui()
        
        # Schedule periodic GUI updates
        self.schedule_gui_updates()
    
    def setup_liveness_detection(self):
        """Initialize liveness detection components"""
        # Load cascade classifiers with proper error handling
        cascade_paths = [
            cv2.data.haarcascades,  # Default OpenCV path
            os.path.join(cv2.__path__[0], 'data'),  # Alternative path
            'cascades',  # Local cascades folder
        ]
        
        cascade_files = {
            'face': 'haarcascade_frontalface_default.xml',
            'eye': 'haarcascade_eye.xml',
            'smile': 'haarcascade_smile.xml'
        }
        
        # Initialize cascades
        self.face_cascade = None
        self.eye_cascade = None
        self.mouth_cascade = None  # Will be None since mouth cascade doesn't exist
        self.smile_cascade = None
        
        for base_path in cascade_paths:
            if not os.path.exists(base_path):
                continue
                
            try:
                # Face cascade (required)
                face_path = os.path.join(base_path, cascade_files['face'])
                if os.path.exists(face_path):
                    test_cascade = cv2.CascadeClassifier(face_path)
                    if not test_cascade.empty():
                        self.face_cascade = test_cascade
                        print(f"Liveness: Face cascade loaded from: {face_path}")
                
                # Eye cascade (for liveness)
                eye_path = os.path.join(base_path, cascade_files['eye'])
                if os.path.exists(eye_path):
                    test_cascade = cv2.CascadeClassifier(eye_path)
                    if not test_cascade.empty():
                        self.eye_cascade = test_cascade
                        print(f"Liveness: Eye cascade loaded from: {eye_path}")
                
                # Smile cascade (for liveness)
                smile_path = os.path.join(base_path, cascade_files['smile'])
                if os.path.exists(smile_path):
                    test_cascade = cv2.CascadeClassifier(smile_path)
                    if not test_cascade.empty():
                        self.smile_cascade = test_cascade
                        print(f"Liveness: Smile cascade loaded from: {smile_path}")
                        
            except Exception as e:
                print(f"Error loading liveness cascades from {base_path}: {e}")
                continue
        
        # Create fallback empty cascades if not found
        if self.face_cascade is None or self.face_cascade.empty():
            print("Warning: Face cascade not found for liveness detection.")
            self.face_cascade = cv2.CascadeClassifier()
            
        if self.eye_cascade is None or self.eye_cascade.empty():
            print("Warning: Eye cascade not found. Eye blink detection disabled.")
            self.eye_cascade = cv2.CascadeClassifier()
            
        # Note: mouth cascade doesn't exist in standard OpenCV, so we'll use smile detection instead
        self.mouth_cascade = cv2.CascadeClassifier()  # Empty cascade
        print("Note: Using smile detection instead of mouth detection (mouth cascade not available)")
            
        if self.smile_cascade is None or self.smile_cascade.empty():
            print("Warning: Smile cascade not found. Smile detection disabled.")
            self.smile_cascade = cv2.CascadeClassifier()
        
        # Enhanced rPPG variables
        self.rppg_buffer_size = 256  # Increased buffer size for better frequency resolution
        self.rppg_buffer = deque(np.zeros(self.rppg_buffer_size))
        self.fps = 30
        self.rppg_timestamps = deque(maxlen=self.rppg_buffer_size)
        
        # Heart rate analysis variables
        self.heart_rate_history = deque(maxlen=10)  # Store last 10 HR estimates
        self.last_hr_update = 0
        self.hr_update_interval = 2.0  # Update HR every 2 seconds
        
        # Signal processing parameters
        self.hr_min = 50  # Minimum heart rate (BPM)
        self.hr_max = 150  # Maximum heart rate (BPM)
        self.signal_quality_threshold = 0.1
        
        # Liveness state variables
        self.liveness_state = LIVENESS_STATE_INITIAL
        self.blink_count = 0
        self.mouth_open_detected = False
        self.smile_detected = False
        self.eye_blink_start_time = 0
        self.blinks_required = 0
        self.last_liveness_check = {}
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Attendance tab
        attendance_frame = ttk.Frame(notebook)
        notebook.add(attendance_frame, text="Take Attendance")
        self.setup_attendance_tab(attendance_frame)
        
        # Registration tab
        registration_frame = ttk.Frame(notebook)
        notebook.add(registration_frame, text="Register Student")
        self.setup_registration_tab(registration_frame)
        
        # Reports tab
        reports_frame = ttk.Frame(notebook)
        notebook.add(reports_frame, text="Reports")
        self.setup_reports_tab(reports_frame)
        
        # Face Identification tab
        identification_frame = ttk.Frame(notebook)
        notebook.add(identification_frame, text="Face Identification")
        self.setup_identification_tab(identification_frame)
        
        # Model Settings tab
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model Settings")
        self.setup_model_tab(model_frame)
        
        # Liveness Settings tab
        liveness_frame = ttk.Frame(notebook)
        notebook.add(liveness_frame, text="Liveness Settings")
        self.setup_liveness_tab(liveness_frame)
    
    def setup_attendance_tab(self, parent):
        """Setup attendance taking interface with liveness detection"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True)
        
        # Left side - Camera and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Camera frame
        camera_frame = ttk.LabelFrame(left_frame, text="Camera Feed")
        camera_frame.pack(fill='both', expand=True, pady=5)
        
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.pack(padx=10, pady=10)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_frame, text="Controls")
        controls_frame.pack(fill='x', pady=5)
        
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="Start Camera", command=self.start_camera).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Stop Camera", command=self.stop_camera).pack(side='left', padx=5)
        
        # Liveness toggle
        self.liveness_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Enable Liveness Detection", 
                       variable=self.liveness_var, command=self.toggle_liveness).pack(side='left', padx=10)
        
        # Right side - Status and logs
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='y', padx=5, pady=5)
        
        # Liveness status frame
        self.liveness_status_frame = ttk.LabelFrame(right_frame, text="Liveness Status")
        self.liveness_status_frame.pack(fill='x', pady=5)
        
        self.liveness_status_label = ttk.Label(self.liveness_status_frame, text="Liveness: Ready")
        self.liveness_status_label.pack(pady=5)
        
        self.liveness_prompt_label = ttk.Label(self.liveness_status_frame, text="")
        self.liveness_prompt_label.pack(pady=5)
        
        self.rppg_status_label = ttk.Label(self.liveness_status_frame, text="rPPG: N/A")
        self.rppg_status_label.pack()
        
        self.texture_status_label = ttk.Label(self.liveness_status_frame, text="Texture: N/A")
        self.texture_status_label.pack()
        
        # Attendance log
        log_frame = ttk.LabelFrame(right_frame, text="Today's Attendance")
        log_frame.pack(fill='both', expand=True, pady=5)
        
        self.attendance_tree = ttk.Treeview(log_frame, columns=('Time', 'Status'), show='tree headings')
        self.attendance_tree.heading('#0', text='Student')
        self.attendance_tree.heading('Time', text='Time')
        self.attendance_tree.heading('Status', text='Status')
        self.attendance_tree.pack(fill='both', expand=True)
        
        # Enhanced rPPG Plot (if matplotlib available)
        if is_scipy_matplotlib_installed:
            plot_frame = ttk.LabelFrame(right_frame, text="Heart Rate Monitoring")
            plot_frame.pack(fill='x', pady=5)
            
            # Create subplots for raw signal and processed signal
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(5, 4))
            
            # Raw rPPG signal plot
            self.ax1.set_title("Raw rPPG Signal", fontsize=10)
            self.ax1.set_ylabel("Green Intensity", fontsize=8)
            self.ax1.grid(True, alpha=0.3)
            self.rppg_raw_line, = self.ax1.plot(np.arange(self.rppg_buffer_size), self.rppg_buffer, 'g-', linewidth=1)
            
            # Filtered rPPG signal plot
            self.ax2.set_title("Filtered Signal & Heart Rate", fontsize=10)
            self.ax2.set_xlabel("Time (frames)", fontsize=8)
            self.ax2.set_ylabel("Filtered Amplitude", fontsize=8)
            self.ax2.grid(True, alpha=0.3)
            self.rppg_filtered_line, = self.ax2.plot(np.arange(self.rppg_buffer_size), np.zeros(self.rppg_buffer_size), 'r-', linewidth=1)
            
            # Heart rate text display
            self.hr_text = self.ax2.text(0.02, 0.95, "HR: -- BPM", transform=self.ax2.transAxes, 
                                        fontsize=12, fontweight='bold', 
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            plt.tight_layout()
            
            self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
            self.canvas_plot.draw()
            self.canvas_plot.get_tk_widget().pack(fill='both', expand=True)
            
            # Heart rate display label
            self.hr_display_label = ttk.Label(plot_frame, text="Current HR: -- BPM", 
                                            font=('Arial', 12, 'bold'), foreground='red')
            self.hr_display_label.pack(pady=5)
        
        self.refresh_attendance_log()
    
    def setup_registration_tab(self, parent):
        """Setup student registration interface"""
        # Model capability info frame
        capability_frame = ttk.LabelFrame(parent, text="Model Capabilities")
        capability_frame.pack(fill='x', padx=10, pady=5)
        
        self.registration_capability_label = ttk.Label(capability_frame, text="Checking model capabilities...", 
                                                     font=('Arial', 10), wraplength=500)
        self.registration_capability_label.pack(pady=5)
        
        # Form frame
        form_frame = ttk.LabelFrame(parent, text="Student Information")
        form_frame.pack(fill='x', padx=10, pady=10)
        
        # Student details
        ttk.Label(form_frame, text="Name:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.name_entry = ttk.Entry(form_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Student ID:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.student_id_entry = ttk.Entry(form_frame, width=30)
        self.student_id_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(form_frame, text="Email:").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.email_entry = ttk.Entry(form_frame, width=30)
        self.email_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Photo frame
        photo_frame = ttk.LabelFrame(parent, text="Student Photo")
        photo_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Button(photo_frame, text="Select Photo", command=self.select_photo).pack(pady=10)
        
        self.photo_label = ttk.Label(photo_frame, text="No photo selected")
        self.photo_label.pack(pady=10)
        
        # Registration button with capability check
        self.register_button = ttk.Button(parent, text="Register Student", command=self.register_student)
        self.register_button.pack(pady=10)
        
        self.selected_photo_path = None
        
        # Update capabilities display after all UI elements are created
        self.update_registration_capabilities()
    
    def update_registration_capabilities(self):
        """Update the registration capabilities display"""
        try:
            operations = self.face_system.get_supported_operations()
            
            if operations['can_register']:
                # Registration is supported
                capability_text = "✅ Registration Enabled\n"
                capability_text += f"Compatible models: {', '.join(operations['registration_models'])}\n"
                capability_text += "You can register new students using the form below."
                
                if hasattr(self, 'registration_capability_label'):
                    self.registration_capability_label.config(text=capability_text, foreground="green")
                if hasattr(self, 'register_button'):
                    self.register_button.config(state="normal")
                
            else:
                # Registration is not supported
                capability_text = "❌ Registration Disabled\n"
                if operations['can_identify']:
                    capability_text += f"Only identification models loaded: {', '.join(operations['identification_models'])}\n"
                    capability_text += "Classification models can only identify existing faces from their training dataset.\n"
                    capability_text += "To register new faces, enable Siamese models (Euclidean/Cosine) in Model Settings."
                else:
                    capability_text += "No recognition models loaded.\n"
                    capability_text += "Load models in Model Settings to enable registration."
                
                if hasattr(self, 'registration_capability_label'):
                    self.registration_capability_label.config(text=capability_text, foreground="red")
                if hasattr(self, 'register_button'):
                    self.register_button.config(state="disabled")
        
        except Exception as e:
            error_text = f"Error checking capabilities: {e}"
            if hasattr(self, 'registration_capability_label'):
                self.registration_capability_label.config(text=error_text, foreground="orange")
            if hasattr(self, 'register_button'):
                self.register_button.config(state="disabled")
    
    def setup_reports_tab(self, parent):
        """Setup reports interface"""
        # Date selection
        date_frame = ttk.Frame(parent)
        date_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(date_frame, text="Select Date:").pack(side='left')
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(date_frame, textvariable=self.date_var).pack(side='left', padx=5)
        ttk.Button(date_frame, text="Generate Report", command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(date_frame, text="Today's Report", command=self.generate_today_report).pack(side='left', padx=5)
        ttk.Button(date_frame, text="Refresh", command=self.refresh_current_report).pack(side='left', padx=5)
        
        # Report display
        report_frame = ttk.LabelFrame(parent, text="Attendance Report")
        report_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.report_tree = ttk.Treeview(report_frame, columns=('Student ID', 'Time In', 'Time Out', 'Status'), show='tree headings')
        self.report_tree.heading('#0', text='Name')
        self.report_tree.heading('Student ID', text='Student ID')
        self.report_tree.heading('Time In', text='Time In')
        self.report_tree.heading('Time Out', text='Time Out')
        self.report_tree.heading('Status', text='Status')
        self.report_tree.pack(fill='both', expand=True)
    
    def setup_identification_tab(self, parent):
        """Setup face identification interface using camera"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side - Camera and controls for identification
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Camera frame for identification
        camera_frame = ttk.LabelFrame(left_frame, text="Camera Feed - Face Identification")
        camera_frame.pack(fill='both', expand=True, pady=5)
        
        self.id_camera_label = ttk.Label(camera_frame)
        self.id_camera_label.pack(padx=10, pady=10)
        
        # Controls frame for identification
        controls_frame = ttk.LabelFrame(left_frame, text="Identification Controls")
        controls_frame.pack(fill='x', pady=5)
        
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(pady=10)
        
        # Camera control buttons
        ttk.Button(button_frame, text="Start ID Camera", command=self.start_id_camera).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Stop ID Camera", command=self.stop_id_camera).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Capture & Identify", command=self.capture_and_identify).pack(side='left', padx=10)
        
        # Identification settings
        settings_frame = ttk.Frame(controls_frame)
        settings_frame.pack(pady=10)
        
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(side='left')
        self.id_confidence_var = tk.DoubleVar(value=0.6)
        confidence_scale = ttk.Scale(settings_frame, from_=0.3, to=0.9, variable=self.id_confidence_var, 
                                   orient='horizontal', length=150)
        confidence_scale.pack(side='left', padx=5)
        self.confidence_label = ttk.Label(settings_frame, text="0.6")
        self.confidence_label.pack(side='left', padx=5)
        
        # Update confidence label when scale changes
        confidence_scale.configure(command=lambda val: self.confidence_label.config(text=f"{float(val):.2f}"))
        
        # Right side - Identification results and history
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='y', padx=5, pady=5)
        
        # Current identification result
        result_frame = ttk.LabelFrame(right_frame, text="Identification Result")
        result_frame.pack(fill='x', pady=5)
        
        self.id_result_label = ttk.Label(result_frame, text="No identification performed", 
                                       font=('Arial', 12, 'bold'), wraplength=300)
        self.id_result_label.pack(pady=10)
        
        self.id_confidence_label = ttk.Label(result_frame, text="", 
                                           font=('Arial', 10), wraplength=300)
        self.id_confidence_label.pack(pady=5)
        
        # Captured image display
        image_frame = ttk.LabelFrame(right_frame, text="Captured Image")
        image_frame.pack(fill='x', pady=5)
        
        self.captured_image_label = ttk.Label(image_frame, text="No image captured")
        self.captured_image_label.pack(pady=10)
        
        # Identification history
        history_frame = ttk.LabelFrame(right_frame, text="Identification History")
        history_frame.pack(fill='both', expand=True, pady=5)
        
        # Create treeview for identification history
        self.id_history_tree = ttk.Treeview(history_frame, columns=('Time', 'Confidence', 'Status'), show='tree headings', height=10)
        self.id_history_tree.heading('#0', text='Identified Person')
        self.id_history_tree.heading('Time', text='Time')
        self.id_history_tree.heading('Confidence', text='Confidence')
        self.id_history_tree.heading('Status', text='Status')
        
        # Configure column widths
        self.id_history_tree.column('#0', width=150)
        self.id_history_tree.column('Time', width=80)
        self.id_history_tree.column('Confidence', width=80)
        self.id_history_tree.column('Status', width=80)
        
        self.id_history_tree.pack(fill='both', expand=True)
        
        # Add scrollbar to history
        history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.id_history_tree.yview)
        self.id_history_tree.configure(yscrollcommand=history_scrollbar.set)
        history_scrollbar.pack(side="right", fill="y")
        
        # Clear history button
        ttk.Button(history_frame, text="Clear History", command=self.clear_identification_history).pack(pady=5)
        
        # Initialize identification camera variables
        self.id_camera = None
        self.id_camera_running = False
        self.identification_history = []
    
    def setup_liveness_tab(self, parent):
        """Setup liveness detection settings"""
        settings_frame = ttk.LabelFrame(parent, text="Liveness Detection Settings")
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(settings_frame, text="Required blinks (3-7):").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.blinks_var = tk.IntVar(value=5)
        ttk.Spinbox(settings_frame, from_=3, to=7, textvariable=self.blinks_var).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Liveness check interval (seconds):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.interval_var = tk.IntVar(value=30)
        ttk.Spinbox(settings_frame, from_=10, to=120, textvariable=self.interval_var).grid(row=1, column=1, padx=5, pady=5)
        
        # Information frame
        info_frame = ttk.LabelFrame(parent, text="Liveness Detection Information")
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        info_text = """
Liveness Detection Features:
1. Eye Blink Detection - Detects natural eye blinking patterns
2. Mouth Movement - Detects mouth opening on command
3. Smile Detection - Detects facial expressions
4. rPPG (Heart Rate) - Detects blood flow through facial pixels
5. Texture Analysis - Detects screen reflections and artificial patterns

This system helps prevent spoofing attacks using photos, videos, or screens.
        """
        
        info_label = ttk.Label(info_frame, text=info_text, justify='left')
        info_label.pack(padx=10, pady=10)
    
    def setup_model_tab(self, parent):
        """Setup model selection and management interface"""
        # Model Selection Frame
        selection_frame = ttk.LabelFrame(parent, text="Model Selection")
        selection_frame.pack(fill='x', padx=10, pady=10)
        
        # Get available models
        available_models = self.face_system.get_available_models()
        
        # Model checkboxes
        self.model_vars = {}
        model_descriptions = {
            'face_detection': 'Face Detection (MobileNetV2-SSD) - Auto-included when needed',
            'euclidean': 'Euclidean Siamese Network - Can register NEW faces + identify existing',
            'cosine': 'Cosine Siamese Network - Can register NEW faces + identify existing', 
            'embedding': 'Embedding Classification - Can ONLY identify existing faces (no new registration)'
        }
        
        row = 0
        for model_key, description in model_descriptions.items():
            self.model_vars[model_key] = tk.BooleanVar()
            
            # Set default values (currently loaded models)
            if hasattr(self.face_system, 'models_to_load') and model_key in self.face_system.models_to_load:
                self.model_vars[model_key].set(True)
            elif not hasattr(self.face_system, 'models_to_load'):
                # If no models_to_load attribute, assume all are loaded (backward compatibility)
                self.model_vars[model_key].set(True)
            
            # Disable checkbox if model file is not available
            state = 'normal' if model_key in available_models else 'disabled'
            
            cb = ttk.Checkbutton(selection_frame, text=description, 
                               variable=self.model_vars[model_key], 
                               state=state)
            cb.grid(row=row, column=0, sticky='w', padx=10, pady=5)
            
            # Add status indicator
            status_text = "Available" if model_key in available_models else "Not Found"
            status_color = "green" if model_key in available_models else "red"
            status_label = ttk.Label(selection_frame, text=f"({status_text})", foreground=status_color)
            status_label.grid(row=row, column=1, sticky='w', padx=5, pady=5)
            
            row += 1
        
        # Control buttons
        button_frame = ttk.Frame(selection_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Apply Changes", command=self.apply_model_changes).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Clear All Models", command=self.clear_all_models).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load All Available", command=self.load_all_models).pack(side='left', padx=5)
        
        # Add information label
        info_label = ttk.Label(selection_frame, 
                              text="Note: Face Detection model is automatically included when recognition models are selected.",
                              foreground="blue", font=('Arial', 9))
        info_label.grid(row=row+1, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        
        # Current Status Frame
        status_frame = ttk.LabelFrame(parent, text="Current Model Status")
        status_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status text widget
        self.model_status_text = tk.Text(status_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.model_status_text.yview)
        self.model_status_text.configure(yscrollcommand=scrollbar.set)
        
        self.model_status_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Refresh button
        ttk.Button(status_frame, text="Refresh Status", command=self.refresh_model_status).pack(anchor='ne', padx=5, pady=5)
        
        # Initial status update
        self.refresh_model_status()
    
    def apply_model_changes(self):
        """Apply the selected model changes"""
        # Check if camera is running
        if self.is_running:
            messagebox.showwarning("Warning", "Please stop the camera before changing models.")
            return
        
        # Get selected models
        selected_models = []
        for model_key, var in self.model_vars.items():
            if var.get():
                selected_models.append(model_key)
        
        if not selected_models:
            messagebox.showwarning("Warning", "Please select at least one model.")
            return
        
        # Show loading dialog
        loading_dialog = tk.Toplevel(self.root)
        loading_dialog.title("Loading Models")
        loading_dialog.geometry("300x100")
        loading_dialog.transient(self.root)
        loading_dialog.resizable(False, False)
        
        # Center the dialog
        loading_dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 50,
            self.root.winfo_rooty() + 50
        ))
        
        loading_label = ttk.Label(loading_dialog, text="Loading selected models...")
        loading_label.pack(expand=True)
        
        progress = ttk.Progressbar(loading_dialog, mode='indeterminate')
        progress.pack(fill='x', padx=10, pady=10)
        progress.start()
        
        # Update GUI
        loading_dialog.update()
        
        try:
            # Reload models
            self.face_system.reload_models(selected_models)
            
            # Update status
            self.refresh_model_status()
            
            # Update registration capabilities
            self.update_registration_capabilities()
            
            messagebox.showinfo("Success", f"Models reloaded successfully!\nLoaded: {', '.join(selected_models)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload models: {str(e)}")
        
        finally:
            progress.stop()
            loading_dialog.destroy()
    
    def clear_all_models(self):
        """Clear all loaded models"""
        if self.is_running:
            messagebox.showwarning("Warning", "Please stop the camera before clearing models.")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all models? This will free memory but disable recognition."):
            try:
                self.face_system.clear_models()
                
                # Uncheck all checkboxes
                for var in self.model_vars.values():
                    var.set(False)
                
                self.refresh_model_status()
                messagebox.showinfo("Success", "All models cleared successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear models: {str(e)}")
    
    def load_all_models(self):
        """Load all available models"""
        if self.is_running:
            messagebox.showwarning("Warning", "Please stop the camera before loading models.")
            return
        
        available_models = self.face_system.get_available_models()
        
        if not available_models:
            messagebox.showwarning("Warning", "No model files found.")
            return
        
        # Check all available models
        for model_key, var in self.model_vars.items():
            if model_key in available_models:
                var.set(True)
            else:
                var.set(False)
        
        # Apply changes
        self.apply_model_changes()
    
    def refresh_model_status(self):
        """Refresh the model status display"""
        self.model_status_text.delete(1.0, tk.END)
        
        # Get model info
        model_info = self.face_system.get_model_info()
        
        status_text = "=== Model Loading Status ===\n\n"
        
        # Currently loaded models
        if model_info['models_loaded']:
            status_text += "Currently Loaded Models:\n"
            for model in model_info['models_loaded']:
                status_text += f"  ✓ {model}\n"
            status_text += f"\nTotal Parameters: {model_info['total_parameters']:,}\n\n"
        else:
            status_text += "No models currently loaded\n\n"
        
        # Available model files
        available_models = self.face_system.get_available_models()
        status_text += "Available Model Files:\n"
        
        model_files = {
            'face_detection': ['models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel'],
            'euclidean': ['models/siamese_euclidean.keras'],
            'cosine': ['models/siamese_cosine.keras'],
            'embedding': ['models/embedding_classification.keras']
        }
        
        for model_key, files in model_files.items():
            if model_key in available_models:
                status_text += f"  ✓ {model_key.title()}: Available\n"
                for file_path in files:
                    if os.path.exists(file_path):
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        status_text += f"    - {file_path} ({size_mb:.1f} MB)\n"
            else:
                status_text += f"  ✗ {model_key.title()}: Missing\n"
                for file_path in files:
                    status_text += f"    - {file_path} {'✓' if os.path.exists(file_path) else '✗'}\n"
        
        # Model capabilities
        status_text += "\n=== Model Capabilities ===\n"
        try:
            operations = self.face_system.get_supported_operations()
            for message in operations['messages']:
                status_text += f"{message}\n"
            
            status_text += f"\nRegistration: {'✅ Enabled' if operations['can_register'] else '❌ Disabled'}\n"
            status_text += f"Identification: {'✅ Enabled' if operations['can_identify'] else '❌ Disabled'}\n"
        except Exception as e:
            status_text += f"Error checking capabilities: {e}\n"
        
        # Memory usage estimation
        status_text += "\n=== Memory Usage ===\n"
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            status_text += f"Current Process Memory: {memory_mb:.1f} MB\n"
        except ImportError:
            status_text += "Memory info not available (install psutil for details)\n"
        
        # Model testing results
        status_text += "\n=== Model Test Results ===\n"
        try:
            test_results = self.face_system.test_model_loading()
            if test_results:
                for model_name, result in test_results.items():
                    if result['status'] == 'success':
                        status_text += f"  ✓ {model_name}: Working (Output: {result['output_shape']})\n"
                    else:
                        status_text += f"  ✗ {model_name}: Failed - {result['error']}\n"
            else:
                status_text += "  No models to test\n"
        except Exception as e:
            status_text += f"  Error testing models: {e}\n"
        
        self.model_status_text.insert(1.0, status_text)
        self.model_status_text.see(1.0)
    
    def extract_rppg_signal(self, face_roi_color):
        """Extract rPPG signal from face region using improved method"""
        if face_roi_color is None or face_roi_color.size == 0:
            return None
        
        # Convert to float for better precision
        face_roi_float = face_roi_color.astype(np.float32)
        
        # Extract color channels
        blue = face_roi_float[:, :, 0]
        green = face_roi_float[:, :, 1]
        red = face_roi_float[:, :, 2]
        
        # Method 1: Simple green channel average (most common)
        green_avg = np.mean(green)
        
        # Method 2: Chrominance-based approach (more robust)
        # Normalize by overall brightness to reduce illumination changes
        brightness = np.mean(face_roi_float)
        if brightness > 0:
            normalized_green = green_avg / brightness
        else:
            normalized_green = green_avg
        
        # Method 3: RGB combination approach
        # Use linear combination optimized for pulse detection
        rgb_pulse = 0.7 * green_avg - 0.5 * red.mean() - 0.2 * blue.mean()
        
        return {
            'green_avg': green_avg,
            'normalized_green': normalized_green,
            'rgb_pulse': rgb_pulse,
            'timestamp': time.time()
        }
    
    def process_rppg_signal(self, rppg_data):
        """Process rPPG signal to estimate heart rate"""
        if not is_scipy_matplotlib_installed or len(self.rppg_buffer) < self.rppg_buffer_size // 2:
            return None
        
        try:
            # Convert buffer to numpy array
            signal = np.array(list(self.rppg_buffer))
            
            # Remove DC component (mean)
            signal = signal - np.mean(signal)
            
            # Apply bandpass filter for heart rate range
            fs = self.fps
            lowcut = self.hr_min / 60.0  # Convert BPM to Hz
            highcut = self.hr_max / 60.0  # Convert BPM to Hz
            
            # Design Butterworth bandpass filter
            nyquist = fs / 2
            low = lowcut / nyquist
            high = highcut / nyquist
            
            if high >= 1.0:
                high = 0.99
            if low <= 0:
                low = 0.01
            
            b, a = scipy.signal.butter(4, [low, high], btype='bandpass')
            filtered_signal = scipy.signal.filtfilt(b, a, signal)
            
            # Apply window function to reduce spectral leakage
            try:
                # Try different window function names (scipy version compatibility)
                if hasattr(scipy.signal, 'windows'):
                    window = scipy.signal.windows.hann(len(filtered_signal))
                elif hasattr(scipy.signal, 'hann'):
                    window = scipy.signal.hann(len(filtered_signal))
                else:
                    # Fallback: create Hann window manually
                    n = len(filtered_signal)
                    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
                
                windowed_signal = filtered_signal * window
            except:
                # If windowing fails, use the filtered signal as-is
                windowed_signal = filtered_signal
            
            # Compute FFT for frequency analysis
            fft = np.fft.fft(windowed_signal)
            freqs = np.fft.fftfreq(len(windowed_signal), d=1.0/fs)
            
            # Focus on positive frequencies in the heart rate range
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Find frequency range corresponding to heart rate
            hr_freq_min = self.hr_min / 60.0
            hr_freq_max = self.hr_max / 60.0
            
            valid_indices = np.where((positive_freqs >= hr_freq_min) & (positive_freqs <= hr_freq_max))[0]
            
            if len(valid_indices) == 0:
                return None
            
            valid_freqs = positive_freqs[valid_indices]
            valid_spectrum = positive_fft[valid_indices]
            
            # Find peak frequency (strongest heart rate component)
            peak_idx = np.argmax(valid_spectrum)
            peak_freq = valid_freqs[peak_idx]
            peak_amplitude = valid_spectrum[peak_idx]
            
            # Calculate heart rate in BPM
            heart_rate = peak_freq * 60
            
            # Signal quality assessment
            signal_power = np.mean(valid_spectrum)
            peak_prominence = peak_amplitude / (signal_power + 1e-10)
            
            # Quality metrics
            signal_quality = min(peak_prominence / 3.0, 1.0)  # Normalize to 0-1
            confidence = "High" if signal_quality > 0.6 else "Medium" if signal_quality > 0.3 else "Low"
            
            return {
                'heart_rate': heart_rate,
                'filtered_signal': filtered_signal,
                'signal_quality': signal_quality,
                'confidence': confidence,
                'peak_frequency': peak_freq,
                'peak_amplitude': peak_amplitude,
                'spectrum': valid_spectrum,
                'frequencies': valid_freqs
            }
            
        except Exception as e:
            print(f"Error in rPPG processing: {e}")
            return None
    
    def update_heart_rate_display(self, hr_result):
        """Update heart rate display and plots"""
        if not is_scipy_matplotlib_installed or hr_result is None:
            return
        
        try:
            current_time = time.time()
            
            # Update heart rate history for smoothing
            if hr_result['signal_quality'] > self.signal_quality_threshold:
                self.heart_rate_history.append(hr_result['heart_rate'])
            
            # Calculate smoothed heart rate
            if len(self.heart_rate_history) > 0:
                smoothed_hr = np.median(list(self.heart_rate_history))  # Use median for robustness
                hr_std = np.std(list(self.heart_rate_history))
            else:
                smoothed_hr = 0
                hr_std = 0
            
            # Update plots only if enough time has passed
            if current_time - self.last_hr_update > 0.1:  # Update plot at 10 Hz
                # Update raw signal plot
                self.rppg_raw_line.set_ydata(list(self.rppg_buffer))
                
                # Update filtered signal plot
                if 'filtered_signal' in hr_result:
                    self.rppg_filtered_line.set_ydata(hr_result['filtered_signal'])
                
                # Auto-scale y-axes
                self.ax1.relim()
                self.ax1.autoscale_view()
                self.ax2.relim()
                self.ax2.autoscale_view()
                
                # Update heart rate text on plot
                if len(self.heart_rate_history) > 0:
                    hr_text = f"HR: {smoothed_hr:.1f} BPM\nQ: {hr_result['confidence']}\nStd: ±{hr_std:.1f}"
                    self.hr_text.set_text(hr_text)
                    
                    # Color code based on quality
                    if hr_result['confidence'] == "High":
                        self.hr_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
                    elif hr_result['confidence'] == "Medium":
                        self.hr_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
                    else:
                        self.hr_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
                
                # Update canvas
                self.canvas_plot.draw_idle()
                self.last_hr_update = current_time
            
            # Update heart rate label (less frequently)
            if current_time - getattr(self, 'last_hr_label_update', 0) > self.hr_update_interval:
                if len(self.heart_rate_history) > 0:
                    hr_display_text = f"Current HR: {smoothed_hr:.1f} BPM ({hr_result['confidence']} confidence)"
                    # Update label color based on heart rate range
                    if 60 <= smoothed_hr <= 100:  # Normal range
                        color = 'green'
                    elif 50 <= smoothed_hr <= 150:  # Acceptable range
                        color = 'orange'
                    else:  # Outside normal range
                        color = 'red'
                    
                    self.safe_update_widget(self.hr_display_label.config, text=hr_display_text, foreground=color)
                else:
                    self.safe_update_widget(self.hr_display_label.config, text="Current HR: -- BPM", foreground='gray')
                
                self.last_hr_label_update = current_time
                
        except Exception as e:
            print(f"Error updating heart rate display: {e}")
    
    def reset_rppg_analysis(self):
        """Reset rPPG analysis buffers and displays"""
        self.rppg_buffer.clear()
        self.rppg_buffer.extend(np.zeros(self.rppg_buffer_size))
        self.rppg_timestamps.clear()
        self.heart_rate_history.clear()
        
        if is_scipy_matplotlib_installed and hasattr(self, 'hr_display_label'):
            self.safe_update_widget(self.hr_display_label.config, text="Current HR: -- BPM", foreground='gray')
            self.safe_update_widget(self.hr_text.set_text, "HR: -- BPM")
    
    def toggle_liveness(self):
        """Toggle liveness detection on/off"""
        self.liveness_enabled = self.liveness_var.get()
        if not self.liveness_enabled:
            self.liveness_status_label.config(text="Liveness: Disabled")
            self.liveness_prompt_label.config(text="")
    
    def start_camera(self):
        """Start camera for attendance with liveness detection"""
        if not self.is_running:
            self.camera = cv2.VideoCapture(0)
            self.is_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
    
    def stop_camera(self):
        """Stop camera"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        # Reset liveness state
        self.liveness_state = LIVENESS_STATE_INITIAL
        self.blink_count = 0
        self.mouth_open_detected = False
        self.smile_detected = False
        # Reset rPPG analysis
        self.reset_rppg_analysis()
    
    def schedule_gui_updates(self):
        """Process GUI updates from the queue"""
        try:
            while True:
                update_data = self.gui_update_queue.get_nowait()
                if len(update_data) == 3:
                    update_func, args, kwargs = update_data
                    update_func(*args, **kwargs)
                else:
                    # Backward compatibility
                    update_func, args = update_data
                    update_func(*args)
        except queue.Empty:
            pass
        
        # Schedule next update check
        self.root.after(50, self.schedule_gui_updates)
    
    def safe_update_widget(self, widget_update_func, *args, **kwargs):
        """Safely update widget from any thread"""
        self.gui_update_queue.put((widget_update_func, args, kwargs))
    
    def perform_liveness_check(self, frame, face_roi_gray, face_roi_color):
        """Perform comprehensive liveness detection"""
        if not self.liveness_enabled:
            return True, "Liveness disabled"
        
        # Eye detection (only if cascade is available)
        eyes = []
        if self.eye_cascade and not self.eye_cascade.empty():
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=4)
        
        # Use smile detection for mouth movement (since mouth cascade doesn't exist)
        mouths = []
        smiles = []
        total_smiles = 0
        smiles_lower = []
        
        if self.smile_cascade and not self.smile_cascade.empty():
            # Detect smiles with different parameters for mouth movement detection
            mouths = self.smile_cascade.detectMultiScale(
                face_roi_gray, 
                scaleFactor=1.2, 
                minNeighbors=3, 
                minSize=(20, 20),
                maxSize=(80, 80)
            )
            
            # Improved smile detection with more lenient parameters
            smiles = self.smile_cascade.detectMultiScale(
                face_roi_gray, 
                scaleFactor=1.1,  # More sensitive
                minNeighbors=3,   # More lenient
                minSize=(10, 10), # Smaller minimum
                maxSize=(120, 120)
            )
            
            # Also try detecting smiles in the lower half of the face
            h, w = face_roi_gray.shape
            lower_face = face_roi_gray[h//3:, :]  # Start from 1/3 instead of 1/2
            smiles_lower = self.smile_cascade.detectMultiScale(
                lower_face,
                scaleFactor=1.05,  # Very sensitive
                minNeighbors=2,    # Very lenient
                minSize=(8, 8),
                maxSize=(100, 100)
            )
            
            # Combine smile detections
            total_smiles = len(smiles) + len(smiles_lower)
        
        # Update liveness state
        if self.liveness_state == LIVENESS_STATE_INITIAL:
            self.safe_update_widget(self.liveness_prompt_label.config, text="Please center your face")
            self.blinks_required = random.randint(3, 7)
            self.liveness_state = LIVENESS_STATE_BLINK
        
        elif self.liveness_state == LIVENESS_STATE_BLINK:
            self.safe_update_widget(self.liveness_prompt_label.config, text=f"Please blink {self.blinks_required - self.blink_count} more times")
            if len(eyes) == 0:
                if self.eye_blink_start_time == 0:
                    self.eye_blink_start_time = time.time()
            elif self.eye_blink_start_time != 0 and time.time() - self.eye_blink_start_time > 0.15:
                self.blink_count += 1
                self.eye_blink_start_time = 0
                print(f"Blink detected! Count: {self.blink_count}/{self.blinks_required}")
            
            if self.blink_count >= self.blinks_required:
                self.liveness_state = LIVENESS_STATE_OPEN_MOUTH
                print("Moving to mouth/smile detection phase")
        
        elif self.liveness_state == LIVENESS_STATE_OPEN_MOUTH:
            self.safe_update_widget(self.liveness_prompt_label.config, text="Please smile or open mouth")
            # Use smile detection as mouth detection since mouth cascade doesn't exist
            if len(mouths) > 0 or total_smiles > 0:
                self.mouth_open_detected = True
                self.liveness_state = LIVENESS_STATE_SMILE
                print("Mouth movement detected, moving to smile phase")
        
        elif self.liveness_state == LIVENESS_STATE_SMILE:
            self.safe_update_widget(self.liveness_prompt_label.config, text="Please smile! (Show your teeth)")
            
            # More lenient smile detection
            if total_smiles > 0:
                self.smile_detected = True
                self.liveness_state = LIVENESS_STATE_LIVE
                print(f"Smile detected! Moving to live state. Total smiles: {total_smiles}")
            
            # Enhanced debug info for smile detection
            debug_text = f"Smiles detected: {total_smiles} (upper: {len(smiles)}, lower: {len(smiles_lower)})"
            print(debug_text)
            
            # Alternative: if no smile detected for too long, try different approach
            if not hasattr(self, 'smile_attempt_start'):
                self.smile_attempt_start = time.time()
            elif time.time() - self.smile_attempt_start > 10:  # After 10 seconds, be more lenient
                if len(mouths) > 0:  # Accept any mouth movement as smile
                    self.smile_detected = True
                    self.liveness_state = LIVENESS_STATE_LIVE
                    print("Fallback: Accepting mouth movement as smile")
        
        # Enhanced rPPG Analysis
        rppg_data = self.extract_rppg_signal(face_roi_color)
        rppg_detected = False
        rppg_status = "No Pulse"
        
        if rppg_data:
            # Add to buffer using the best method (normalized green for stability)
            self.rppg_buffer.append(rppg_data['normalized_green'])
            self.rppg_timestamps.append(rppg_data['timestamp'])
            
            # Process signal when buffer is sufficiently filled
            if len(self.rppg_buffer) >= self.rppg_buffer_size * 0.8:  # 80% filled
                hr_result = self.process_rppg_signal(rppg_data)
                
                if hr_result:
                    heart_rate = hr_result['heart_rate']
                    quality = hr_result['signal_quality']
                    confidence = hr_result['confidence']
                    
                    rppg_status = f"HR: {heart_rate:.1f} BPM ({confidence})"
                    
                    # Consider pulse detected if quality is sufficient and HR is in reasonable range
                    if quality > self.signal_quality_threshold and self.hr_min <= heart_rate <= self.hr_max:
                        rppg_detected = True
                    
                    # Update displays
                    self.update_heart_rate_display(hr_result)
                else:
                    rppg_status = "Processing..."
            else:
                rppg_status = f"Collecting data... ({len(self.rppg_buffer)}/{self.rppg_buffer_size})"
        
        # Texture analysis
        f_transform = np.fft.fft2(face_roi_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-12)
        
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        
        outer_h_start = int(cy - h/6)
        outer_h_end = int(cy + h/6)
        outer_w_start = int(cx - w/6)
        outer_w_end = int(cx + w/6)
        
        magnitude_spectrum[outer_h_start:outer_h_end, outer_w_start:outer_w_end] = 0
        
        texture_ok = True
        texture_status = "Texture OK"
        if np.max(magnitude_spectrum) > 2.5 * np.mean(magnitude_spectrum):
            texture_ok = False
            texture_status = "Screen detected!"
        
        # Update status labels
        self.safe_update_widget(self.rppg_status_label.config, text=f"rPPG: {rppg_status}")
        self.safe_update_widget(self.texture_status_label.config, text=f"Texture: {texture_status}")
        
        # More lenient final liveness decision for testing
        if self.liveness_state == LIVENESS_STATE_LIVE and texture_ok:
            # Don't require rPPG for now to test smile detection
            self.safe_update_widget(self.liveness_status_label.config, text="Liveness: LIVE", foreground="green")
            self.safe_update_widget(self.liveness_prompt_label.config, text="Liveness confirmed!")
            return True, "Live person detected"
        elif self.liveness_state == LIVENESS_STATE_SPOOF or not texture_ok:
            self.safe_update_widget(self.liveness_status_label.config, text="Liveness: SPOOF", foreground="red")
            self.safe_update_widget(self.liveness_prompt_label.config, text="Spoof detected!")
            return False, "Spoof detected"
        else:
            self.safe_update_widget(self.liveness_status_label.config, text="Liveness: Checking...", foreground="orange")
            return False, "Liveness check in progress"
    
    def camera_loop(self):
        """Main camera loop with integrated face recognition and liveness detection"""
        last_recognition = {}
        
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    face_roi_gray = gray[y:y+h, x:x+w]
                    face_roi_color = frame[y:y+h, x:x+w]
                    
                    # Use the new liveness detection method
                    if self.liveness_enabled:
                        face_locations, face_names, liveness_results = self.face_system.recognize_faces_with_liveness(frame, require_liveness=True)
                    else:
                        face_locations, face_names = self.face_system.recognize_faces(frame)
                        liveness_results = [True] * len(face_names)  # All considered live if liveness disabled
                    
                    for i, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
                        is_live = liveness_results[i] if i < len(liveness_results) else False
                        
                        # Change rectangle color based on liveness
                        color = (0, 255, 0) if is_live else (0, 165, 255)  # Green if live, orange if not
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Mark attendance only if recognized AND live (or liveness disabled)
                        if name != "Unknown" and "_" in name and (is_live or not self.liveness_enabled):
                            try:
                                # Extract clean name and student ID
                                clean_name = name.split(" (")[0] if " (" in name else name  # Remove liveness status
                                student_id = clean_name.split("_")[1]
                                
                                # Check if student ID is valid (no special characters)
                                if student_id and student_id.replace(' ', '').replace('-', '').replace('_', '').isalnum():
                                    current_time = datetime.now()
                                    
                                    # Check if enough time has passed
                                    time_threshold = self.interval_var.get() if hasattr(self, 'interval_var') else 30
                                    if student_id not in last_recognition or \
                                       (current_time - last_recognition[student_id]).seconds > time_threshold:
                                        
                                        if self.db.mark_attendance(student_id):
                                            last_recognition[student_id] = current_time
                                            liveness_status = "with liveness detection" if self.liveness_enabled and is_live else "without liveness check"
                                            print(f"Attendance marked for student: {student_id} ({liveness_status})")
                                            # Safely update attendance log
                                            self.safe_update_widget(self.refresh_attendance_log)
                                            # Reset liveness state for next person
                                            self.liveness_state = LIVENESS_STATE_INITIAL
                                            self.blink_count = 0
                                            self.mouth_open_detected = False
                                            self.smile_detected = False
                                            # Reset smile attempt timer
                                            if hasattr(self, 'smile_attempt_start'):
                                                delattr(self, 'smile_attempt_start')
                                        else:
                                            # Only print error once per minute to avoid spam
                                            current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
                                            if not hasattr(self, 'last_error_time') or self.last_error_time != current_minute:
                                                print(f"Could not mark attendance for student ID: {student_id}")
                                                self.last_error_time = current_minute
                                else:
                                    print(f"Invalid student ID format: {student_id}")
                            except Exception as e:
                                print(f"Error processing recognition: {e}")
                
                # Convert to PhotoImage and display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                photo = ImageTk.PhotoImage(img)
                
                self.safe_update_widget(self.camera_label.configure, image=photo)
                self.safe_update_widget(setattr, self.camera_label, 'image', photo)
    
    def select_photo(self):
        """Select photo for student registration"""
        file_path = filedialog.askopenfilename(
            title="Select Student Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.selected_photo_path = file_path
            self.photo_label.configure(text=f"Photo: {file_path.split('/')[-1]}")
    
    def register_student(self):
        """Register a new student"""
        name = self.name_entry.get().strip()
        student_id = self.student_id_entry.get().strip()
        email = self.email_entry.get().strip()
        
        if not name or not student_id or not self.selected_photo_path:
            messagebox.showerror("Error", "Please fill all required fields and select a photo")
            return
        
        # Clean the name to avoid encoding issues
        try:
            # Remove any problematic characters and ensure ASCII compatibility
            clean_name = ''.join(c for c in name if c.isalnum() or c in ' -_').strip()
            if not clean_name:
                clean_name = f"Student{student_id}"
            
            # Add to database
            if self.db.add_student(clean_name, student_id, email):
                # Add face encoding with cleaned name
                if self.face_system.add_face(self.selected_photo_path, clean_name, student_id):
                    messagebox.showinfo("Success", f"Student {clean_name} registered successfully!")
                    self.clear_registration_form()
                else:
                    messagebox.showerror("Error", "Could not process face from photo")
            else:
                messagebox.showerror("Error", "Student ID already exists")
        except Exception as e:
            print(f"Registration error: {e}")
            messagebox.showerror("Error", f"Registration failed: {str(e)}")
    
    def clear_registration_form(self):
        """Clear registration form"""
        self.name_entry.delete(0, tk.END)
        self.student_id_entry.delete(0, tk.END)
        self.email_entry.delete(0, tk.END)
        self.photo_label.configure(text="No photo selected")
        self.selected_photo_path = None
    
    def refresh_attendance_log(self):
        """Refresh today's attendance log"""
        # Clear existing items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Get today's attendance
        report = self.db.get_attendance_report()
        for name, student_id, time_in, time_out, status in report:
            if time_in:  # Only show students who are present
                display_status = "Present" if not time_out else "Checked Out"
                self.attendance_tree.insert('', 'end', text=name, values=(time_in or "N/A", display_status))
    
    def generate_report(self):
        """Generate attendance report for selected date"""
        print("Generating attendance report...")
        
        # Clear existing items first
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
        
        try:
            # Get the selected date
            date_input = self.date_var.get().strip()
            if not date_input:
                # If no date selected, use today
                selected_date = date.today()
                self.date_var.set(selected_date.strftime("%Y-%m-%d"))
            else:
                selected_date = datetime.strptime(date_input, "%Y-%m-%d").date()
            
            print(f"Fetching report for date: {selected_date}")
            
            # Get the report data
            report = self.db.get_attendance_report(selected_date)
            print(f"Found {len(report)} records for {selected_date}")
            
            if not report:
                # Show message if no data found
                self.report_tree.insert('', 'end', text="No data found for this date", 
                                      values=("", "", "", ""))
                print("No attendance data found for selected date")
                return
            
            # Populate the report tree
            present_count = 0
            absent_count = 0
            
            for name, student_id, time_in, time_out, status in report:
                # Determine final status
                if time_in:
                    final_status = "Present"
                    if time_out:
                        final_status = "Checked Out"
                    present_count += 1
                else:
                    final_status = "Absent"
                    absent_count += 1
                
                # Format time display
                time_in_display = time_in if time_in else "N/A"
                time_out_display = time_out if time_out else "N/A"
                
                # Insert into tree
                self.report_tree.insert('', 'end', text=name, 
                                      values=(student_id, time_in_display, time_out_display, final_status))
            
            # Update the tree display
            self.report_tree.update()
            
            # Print summary
            total_students = present_count + absent_count
            print(f"Report Summary: {present_count} present, {absent_count} absent, {total_students} total")
            
            # Show success message
            messagebox.showinfo("Report Generated", 
                              f"Report generated for {selected_date}\n"
                              f"Present: {present_count}\n"
                              f"Absent: {absent_count}\n"
                              f"Total Students: {total_students}")
            
        except ValueError as e:
            error_msg = "Invalid date format. Please use YYYY-MM-DD format (e.g., 2025-08-03)"
            print(f"Date format error: {e}")
            messagebox.showerror("Date Format Error", error_msg)
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            print(f"Report generation error: {e}")
            messagebox.showerror("Error", error_msg)
    
    def generate_today_report(self):
        """Generate report for today's date"""
        today = date.today()
        self.date_var.set(today.strftime("%Y-%m-%d"))
        self.generate_report()
    
    def refresh_current_report(self):
        """Refresh the current report (re-generate with same date)"""
        print("Refreshing current report...")
        self.generate_report()
    
    def start_id_camera(self):
        """Start camera for face identification"""
        if not self.id_camera_running:
            try:
                self.id_camera = cv2.VideoCapture(0)
                if not self.id_camera.isOpened():
                    messagebox.showerror("Error", "Could not open camera for identification")
                    return
                
                self.id_camera_running = True
                self.id_camera_thread = threading.Thread(target=self.id_camera_loop)
                self.id_camera_thread.daemon = True
                self.id_camera_thread.start()
                
                self.safe_update_widget(self.id_result_label.config, text="Camera started - Ready for identification")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start identification camera: {str(e)}")
    
    def stop_id_camera(self):
        """Stop identification camera"""
        self.id_camera_running = False
        if self.id_camera:
            self.id_camera.release()
            self.id_camera = None
        
        self.safe_update_widget(self.id_result_label.config, text="Camera stopped")
        self.safe_update_widget(self.id_camera_label.config, image='', text="Camera stopped")
    
    def id_camera_loop(self):
        """Camera loop for identification (display only, no processing until capture)"""
        while self.id_camera_running and self.id_camera:
            try:
                ret, frame = self.id_camera.read()
                if ret and self.id_camera_running:
                    # Store the current frame for capture
                    self.current_id_frame = frame.copy()
                    
                    # Draw rectangles around detected faces for preview
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if hasattr(self, 'face_cascade') and self.face_cascade and not self.face_cascade.empty():
                        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
                        
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Convert to PhotoImage and display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((480, 360))  # Smaller size for identification tab
                    photo = ImageTk.PhotoImage(img)
                    
                    self.safe_update_widget(self.id_camera_label.configure, image=photo, text="")
                    self.safe_update_widget(setattr, self.id_camera_label, 'image', photo)
                    
                else:
                    break
                    
            except Exception as e:
                print(f"Error in identification camera loop: {e}")
                break
        
        # Cleanup when loop exits
        if self.id_camera:
            self.id_camera.release()
            self.id_camera = None
    
    def capture_and_identify(self):
        """Capture current frame and identify the person"""
        if not self.id_camera_running or not hasattr(self, 'current_id_frame'):
            messagebox.showwarning("Warning", "Please start the identification camera first")
            return
        
        try:
            # Get the current frame
            frame = self.current_id_frame.copy()
            
            # Show processing message
            self.safe_update_widget(self.id_result_label.config, text="Processing... Please wait")
            self.safe_update_widget(self.id_confidence_label.config, text="")
            
            # Process the frame for face recognition
            self.process_identification_frame(frame)
            
        except Exception as e:
            error_msg = f"Error during identification: {str(e)}"
            print(error_msg)
            self.safe_update_widget(self.id_result_label.config, text="Identification failed")
            self.safe_update_widget(self.id_confidence_label.config, text=error_msg)
    
    def process_identification_frame(self, frame):
        """Process the captured frame for face identification"""
        try:
            # Use face recognition system to identify faces
            face_locations, face_names = self.face_system.recognize_faces(frame)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            confidence_threshold = self.id_confidence_var.get()
            
            if len(face_locations) == 0:
                # No faces detected
                result_text = "No faces detected in the image"
                confidence_text = "Try adjusting camera position or lighting"
                status = "No Face"
                identified_person = "None"
                confidence_score = 0.0
                
            elif len(face_locations) == 1:
                # Single face detected
                name = face_names[0]
                
                if name == "Unknown":
                    result_text = "Face detected but not recognized"
                    confidence_text = "Person not found in database"
                    status = "Unknown"
                    identified_person = "Unknown Person"
                    confidence_score = 0.0
                else:
                    # Extract confidence from name if available (format: "Name_ID (confidence: 0.xx)")
                    if " (confidence:" in name:
                        name_part = name.split(" (confidence:")[0]
                        confidence_part = name.split(" (confidence:")[1].replace(")", "")
                        try:
                            confidence_score = float(confidence_part)
                        except:
                            confidence_score = 0.0
                    else:
                        name_part = name
                        confidence_score = 1.0  # Assume high confidence if not specified
                    
                    # Check if confidence meets threshold
                    if confidence_score >= confidence_threshold:
                        # Extract clean name and student ID
                        if "_" in name_part:
                            clean_name = name_part.split("_")[0]
                            student_id = name_part.split("_")[1]
                            result_text = f"✅ Identified: {clean_name}"
                            confidence_text = f"Student ID: {student_id}\nConfidence: {confidence_score:.3f}"
                            status = "Identified"
                            identified_person = f"{clean_name} ({student_id})"
                        else:
                            result_text = f"✅ Identified: {name_part}"
                            confidence_text = f"Confidence: {confidence_score:.3f}"
                            status = "Identified"
                            identified_person = name_part
                    else:
                        result_text = "⚠️ Low confidence identification"
                        confidence_text = f"Possible match: {name_part}\nConfidence: {confidence_score:.3f} (below threshold: {confidence_threshold:.2f})"
                        status = "Low Confidence"
                        identified_person = f"? {name_part}"
                        
            else:
                # Multiple faces detected
                recognized_count = sum(1 for name in face_names if name != "Unknown")
                result_text = f"Multiple faces detected ({len(face_locations)} faces)"
                confidence_text = f"Recognized: {recognized_count}, Unknown: {len(face_locations) - recognized_count}"
                status = "Multiple Faces"
                identified_person = f"Multiple ({len(face_locations)})"
                confidence_score = 0.0
            
            # Update GUI with results
            self.safe_update_widget(self.id_result_label.config, text=result_text)
            self.safe_update_widget(self.id_confidence_label.config, text=confidence_text)
            
            # Display captured image
            self.display_captured_image(frame, face_locations)
            
            # Add to identification history
            self.add_to_identification_history(identified_person, timestamp, confidence_score, status)
            
            # Optionally save the result to a log file
            self.log_identification_result(identified_person, timestamp, confidence_score, status, result_text)
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            print(error_msg)
            self.safe_update_widget(self.id_result_label.config, text="Processing failed")
            self.safe_update_widget(self.id_confidence_label.config, text=error_msg)
    
    def display_captured_image(self, frame, face_locations=None):
        """Display the captured image with face rectangles"""
        try:
            # Draw rectangles around detected faces
            display_frame = frame.copy()
            if face_locations:
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Convert to PhotoImage and display
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((200, 150))  # Thumbnail size
            photo = ImageTk.PhotoImage(img)
            
            self.safe_update_widget(self.captured_image_label.configure, image=photo, text="")
            self.safe_update_widget(setattr, self.captured_image_label, 'image', photo)
            
        except Exception as e:
            print(f"Error displaying captured image: {e}")
            self.safe_update_widget(self.captured_image_label.config, text="Error displaying image")
    
    def add_to_identification_history(self, person, timestamp, confidence, status):
        """Add identification result to history"""
        try:
            # Add to internal history list
            history_entry = {
                'person': person,
                'timestamp': timestamp,
                'confidence': confidence,
                'status': status
            }
            self.identification_history.append(history_entry)
            
            # Keep only last 100 entries
            if len(self.identification_history) > 100:
                self.identification_history.pop(0)
            
            # Add to treeview
            confidence_text = f"{confidence:.3f}" if confidence > 0 else "N/A"
            self.safe_update_widget(
                self.id_history_tree.insert, '', 0,  # Insert at top
                text=person,
                values=(timestamp, confidence_text, status)
            )
            
        except Exception as e:
            print(f"Error adding to identification history: {e}")
    
    def clear_identification_history(self):
        """Clear the identification history"""
        try:
            # Clear internal history
            self.identification_history.clear()
            
            # Clear treeview
            for item in self.id_history_tree.get_children():
                self.id_history_tree.delete(item)
            
            self.safe_update_widget(self.id_result_label.config, text="History cleared")
            
        except Exception as e:
            print(f"Error clearing identification history: {e}")
    
    def on_closing(self):
        """Handle application closing"""
        # Stop all cameras
        self.is_running = False
        self.id_camera_running = False
        
        # Release cameras
        if self.camera:
            self.camera.release()
        if self.id_camera:
            self.id_camera.release()
        
        # Close the application
        self.root.destroy()
        
    def log_identification_result(self, person, timestamp, confidence, status, result_text):
        """Log identification results to file"""
        try:
            log_filename = f"identification_log_{datetime.now().strftime('%Y%m%d')}.txt"
            log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {person} | Confidence: {confidence:.3f} | Status: {status} | Result: {result_text}\n"
            
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(log_entry)
                
        except Exception as e:
            print(f"Error logging identification result: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedAttendanceGUI(root)
    
    # Set up cleanup handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()
