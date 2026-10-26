import os
# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import tkinter as tk
from integrated_attendance_gui import IntegratedAttendanceGUI

def main():
    root = tk.Tk()
    app = IntegratedAttendanceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()