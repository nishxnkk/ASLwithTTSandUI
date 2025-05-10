import cv2
import numpy as np
import math
import time
import threading
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pyttsx3

# Try importing cvzone modules with error handling
try:
    from cvzone.HandTrackingModule import HandDetector
    from cvzone.ClassificationModule import Classifier
except ImportError:
    messagebox.showerror("Import Error", "cvzone module not found. Please install it using: pip install cvzone")
    exit(1)


class ASLDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Detection and Translation")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

        # App state variables
        self.running = True
        self.finalText = ""
        self.wordText = ""
        self.startDetectTime = 0
        self.detecting = False
        self.captured = False
        self.lastDetectedTime = 0
        self.predictedLetter = ""
        self.space_added = False
        self.last_prediction = ""

        # Configuration parameters
        self.offset = 20
        self.imageSize = 300
        self.detectGap = 1.0
        self.captureDelay = 1.5  # Reduced from 2.0 for better responsiveness
        self.labels = [chr(i) for i in range(65, 91)]  # A-Z

        # Set up the UI components
        self.setup_ui()

        # Set up the model and camera
        self.setup_model()

        # Set up TTS
        self.setup_tts()

        # Start the video processing thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Start the UI update loop
        self.update_ui()

    def setup_ui(self):
        """Set up all UI components"""
        # Create a gradient background
        self.canvas = tk.Canvas(self.root, width=1200, height=700, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.draw_gradient(self.canvas, "#3a86ff", "#000000", 120)

        # Main container
        self.main_frame = ttk.Frame(self.canvas)
        self.canvas.create_window(0, 0, anchor="nw", window=self.main_frame, width=1200, height=700)

        # Configure custom styles
        self.style = ttk.Style()
        self.style.configure("Title.TLabel", font=('Helvetica', 24, 'bold'), foreground="#ffffff", background="#3a86ff")
        self.style.configure("SubTitle.TLabel", font=('Helvetica', 16), foreground="#ffffff", background="#3a86ff")
        self.style.configure("TButton", font=('Helvetica', 12))

        # Title and subtitle
        ttk.Label(self.main_frame, text="🖐️ ASL Detection and Translation", style="Title.TLabel").pack(pady=(20, 5))
        ttk.Label(self.main_frame, text="Hold hand signs in frame for recognition", style="SubTitle.TLabel").pack(
            pady=(0, 20))

        # Main container for video and controls
        self.container = ttk.Frame(self.main_frame)
        self.container.pack(padx=20, pady=10, fill='both', expand=True)

        # Video panel on the left
        self.video_frame = ttk.Frame(self.container)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)

        self.video_panel = tk.Label(self.video_frame, bg="black", width=640, height=480, bd=2, relief="groove")
        self.video_panel.pack(padx=10, pady=10)

        self.progress_bar = ttk.Progressbar(self.video_frame, orient='horizontal', length=640, mode='determinate')
        self.progress_bar.pack(fill='x', padx=10, pady=(0, 10))

        # Controls on the right
        self.right_frame = ttk.Frame(self.container)
        self.right_frame.grid(row=0, column=1, sticky="n", padx=10)

        # Live prediction
        self.prediction_frame = ttk.Frame(self.right_frame, relief="groove", borderwidth=2)
        self.prediction_frame.pack(fill="x", pady=10)

        ttk.Label(self.prediction_frame, text="Live Prediction:", font=("Helvetica", 14)).pack(anchor="w", padx=10,
                                                                                               pady=(10, 0))
        self.live_pred = ttk.Label(self.prediction_frame, text="—", font=("Helvetica", 42), foreground="#00ffcc")
        self.live_pred.pack(pady=10)

        # Text accumulation
        ttk.Label(self.right_frame, text="Captured Letters:", font=("Helvetica", 14)).pack(anchor="w", pady=(10, 0))
        self.text_frame = ttk.Frame(self.right_frame)
        self.text_frame.pack(fill="both", expand=True, pady=10)

        self.captured_text = tk.Text(self.text_frame, height=8, width=30, font=("Helvetica", 16), wrap='word', bd=2,
                                     relief="groove")
        self.captured_text.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(self.text_frame, orient="vertical", command=self.captured_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.captured_text.configure(yscrollcommand=self.scrollbar.set)

        # Buttons for actions
        self.buttons_frame = ttk.Frame(self.right_frame)
        self.buttons_frame.pack(fill="x", pady=10)

        ttk.Button(self.buttons_frame, text="Add Space", command=self.add_space).pack(side="left", padx=5, pady=5,
                                                                                      fill="x", expand=True)
        ttk.Button(self.buttons_frame, text="Clear Text", command=self.clear_text).pack(side="left", padx=5, pady=5,
                                                                                        fill="x", expand=True)

        self.buttons_frame2 = ttk.Frame(self.right_frame)
        self.buttons_frame2.pack(fill="x", pady=(0, 10))

        ttk.Button(self.buttons_frame2, text="Save Text", command=self.save_text).pack(side="left", padx=5, pady=5,
                                                                                       fill="x", expand=True)
        ttk.Button(self.buttons_frame2, text="Speak Text", command=lambda: self.speak_text(self.finalText)).pack(
            side="left", padx=5, pady=5, fill="x", expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def draw_gradient(self, canvas, color1, color2, steps):
        """Draw a vertical gradient on the canvas"""
        r1, g1, b1 = self.root.winfo_rgb(color1)
        r2, g2, b2 = self.root.winfo_rgb(color2)

        height = 700
        for i in range(steps):
            nr = int(r1 + (r2 - r1) * i / steps) // 256
            ng = int(g1 + (g2 - g1) * i / steps) // 256
            nb = int(b1 + (b2 - b1) * i / steps) // 256
            color = f"#{nr:02x}{ng:02x}{nb:02x}"
            canvas.create_rectangle(0, i * (height // steps), 1200, (i + 1) * (height // steps), outline="", fill=color)

    def setup_model(self):
        """Set up the camera and ML model"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera. Please check your camera connection.")
                self.exit_app()
                return

            # Initialize hand detector and classifier
            self.detector = HandDetector(maxHands=1, detectionCon=0.8)

            model_path = "Models/keras_model.h5"
            labels_path = "Models/labels.txt"

            # Check if model files exist
            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                messagebox.showerror("Model Error",
                                     f"Model files not found at {model_path} or {labels_path}. Please check the file locations.")
                self.exit_app()
                return

            self.classifier = Classifier(model_path, labels_path)
            self.status_var.set("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Setup Error", f"Error setting up the model: {str(e)}")
            self.exit_app()

    def setup_tts(self):
        """Set up text to speech engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            voices = self.engine.getProperty('voices')
            # Set a more natural voice if available
            if len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)  # Often the second voice is female and clearer
        except Exception as e:
            messagebox.showwarning("TTS Warning", f"Could not initialize text-to-speech: {str(e)}")

    def process_video(self):
        """Process video frames in a separate thread"""
        while self.running:
            success, img = self.cap.read()
            if not success:
                continue

            # Create copies for processing and display
            imgOutput = img.copy()

            # Find hands in the image
            hands, _ = self.detector.findHands(img)

            now = time.time()

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Create a white background image for normalization
                imgWhite = np.ones((self.imageSize, self.imageSize, 3), np.uint8) * 255

                # Ensure crop is within image boundaries
                y1 = max(0, y - self.offset)
                y2 = min(img.shape[0], y + h + self.offset)
                x1 = max(0, x - self.offset)
                x2 = min(img.shape[1], x + w + self.offset)

                # Crop and process the hand image
                try:
                    imgCrop = img[y1:y2, x1:x2]
                    aspectRatio = h / w

                    # Resize while maintaining aspect ratio
                    if aspectRatio > 1:
                        k = self.imageSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, self.imageSize))
                        wGap = (self.imageSize - wCal) // 2
                        imgWhite[:, wGap:wGap + wCal] = imgResize
                    else:
                        k = self.imageSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (self.imageSize, hCal))
                        hGap = (self.imageSize - hCal) // 2
                        imgWhite[hGap:hGap + hCal, :] = imgResize

                    # Get prediction
                    prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                    self.predictedLetter = self.labels[index]

                    # Update UI with prediction
                    self.last_prediction = self.predictedLetter

                    # Draw bounding box and prediction
                    cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(imgOutput, self.predictedLetter, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error processing hand image: {str(e)}")
                    self.predictedLetter = "—"

                # Start detecting if not already
                if not self.detecting:
                    self.startDetectTime = now
                    self.detecting = True
                    self.captured = False

                # Calculate progress and capture letter if needed
                progress = min((now - self.startDetectTime) / self.captureDelay, 1.0)
                self.progress = progress * 100

                if not self.captured and progress >= 1.0:
                    self.finalText += self.predictedLetter
                    # Use threading for TTS to avoid UI freezing
                    threading.Thread(target=self.speak_text, args=(self.predictedLetter,)).start()
                    self.captured = True

                self.lastDetectedTime = now
            else:
                # Reset detection if no hand is found for a while
                if self.detecting and (now - self.lastDetectedTime > self.detectGap):
                    self.detecting = False
                    self.captured = False
                    self.predictedLetter = "—"
                    self.progress = 0

            # Convert to RGB for tkinter display
            self.current_frame = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)

    def update_ui(self):
        """Update the UI with the latest processed frame and data"""
        if hasattr(self, 'current_frame') and self.running:
            imgPIL = Image.fromarray(self.current_frame)
            imgTk = ImageTk.PhotoImage(image=imgPIL)
            self.video_panel.imgTk = imgTk
            self.video_panel.config(image=imgTk)

            # Update prediction label
            if hasattr(self, 'predictedLetter'):
                self.live_pred.config(text=self.predictedLetter)

            # Update progress bar
            if hasattr(self, 'progress'):
                self.progress_bar['value'] = self.progress

            # Update text display
            if self.finalText != self.captured_text.get("1.0", tk.END).strip():
                self.captured_text.delete("1.0", tk.END)
                self.captured_text.insert(tk.END, self.finalText)
                self.captured_text.see(tk.END)

        # Call this method again after 15ms
        if self.running:
            self.root.after(15, self.update_ui)

    def add_space(self):
        """Add a space to the accumulated text"""
        self.finalText += " "
        self.space_added = True
        self.captured_text.delete("1.0", tk.END)
        self.captured_text.insert(tk.END, self.finalText)

    def clear_text(self):
        """Clear all accumulated text"""
        self.finalText = ""
        self.captured_text.delete("1.0", tk.END)
        self.status_var.set("Text cleared")

    def save_text(self):
        """Save accumulated text to a file"""
        if not self.finalText:
            messagebox.showinfo("Info", "No text to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save ASL Text"
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.finalText)
                self.status_var.set(f"Text saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save file: {str(e)}")

    def speak_text(self, text):
        """Speak the given text"""
        try:
            self.status_var.set("Speaking...")
            self.engine.say(text)
            self.engine.runAndWait()
            self.status_var.set("Ready")
        except Exception as e:
            print(f"TTS Error: {str(e)}")

    def exit_app(self):
        """Clean up resources and exit the application"""
        self.running = False
        try:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ASLDetectionApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"An unexpected error occurred: {str(e)}")





























