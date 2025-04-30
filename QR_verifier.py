import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from qr_reader import QRReader

class QRVerifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QR Code Signature Verifier")

        self.img1_path = None
        self.img2_path = None

        self.img1_panel = None
        self.img2_panel = None
        self.qr1_panel = None
        self.qr2_panel = None
        self.canvas = None

        self.qr1_image = None
        self.qr2_image = None

        self.QR_reader = QRReader(library='pyzbar', reader_name='QRReader')

        self.setup_gui()

    # ---------------- GUI Setup ----------------
    def setup_gui(self):
        """Initialize GUI components and layout."""
        self.feature_method = tk.StringVar(self.root)
        self.feature_method.set("SIFT")  # default value
        tk.OptionMenu(self.root, self.feature_method, "SIFT", "ORB").pack()
        tk.Button(self.root, text="Load Image 1", command=self.load_and_process_image1).pack()
        tk.Button(self.root, text="Load Image 2", command=self.load_and_process_image2).pack()
        tk.Button(self.root, text="Verify", command=self.verify_images).pack()

        placeholder = Image.new('RGB', (400, 400), color='gray')
        placeholder_img = ImageTk.PhotoImage(placeholder)

        frame1 = tk.Frame(self.root)
        frame1.pack()
        self.img1_panel = tk.Label(frame1, image=placeholder_img)
        self.img1_panel.image = placeholder_img
        self.img1_panel.grid(row=0, column=0)
        self.img2_panel = tk.Label(frame1, image=placeholder_img)
        self.img2_panel.image = placeholder_img
        self.img2_panel.grid(row=0, column=1)
        self.qr_code1_label = tk.Label(frame1, text="Image 1", font=("Arial", 10, "bold"))
        self.qr_code1_label.grid(row=1, column=0)
        self.qr_code2_label = tk.Label(frame1, text="Image 2", font=("Arial", 10, "bold"))
        self.qr_code2_label.grid(row=1, column=1)

        # frame2 = tk.Frame(self.root)
        # frame2.pack()
        # self.img2_panel = tk.Label(frame2, image=placeholder_img)
        # self.img2_panel.image = placeholder_img
        # self.img2_panel.grid(row=0, column=0)
        # self.qr2_panel = tk.Label(frame2, image=placeholder_img)
        # self.qr2_panel.image = placeholder_img
        # self.qr2_panel.grid(row=0, column=1)
        # tk.Label(frame2, text="Image 2", font=("Arial", 10, "bold")).grid(row=1, column=0)
        # self.qr_code2_label = tk.Label(frame2, text="QR Code 2 Region", font=("Arial", 10, "bold"))
        # self.qr_code2_label.grid(row=1, column=1)

        self.result_label = tk.Label(self.root, text="Results will appear here", font=("Arial", 14))
        self.result_label.pack()
        self.qr_code_match_label = tk.Label(self.root, text="QR Code Match: ", font=("Arial", 14))
        self.qr_code_match_label.pack()

    # ---------------- Image Loading and Display ----------------
    def load_and_process_image1(self):
        """Load first image, extract QR region and display."""
        self.img1_path = filedialog.askopenfilename()
        if self.img1_path:
            # self.display_image(self.img1_path, self.img1_panel)
            self.image1 = cv2.imread(self.img1_path)
            self.qr1_image, qr_text, polygon = self.QR_reader.extract_qr_region(self.image1)
            self.image1 = self.draw_polygon_on_image(self.image1, polygon)
            self.display_image(self.image1, self.img1_panel)
            # self.display_qr_region(self.qr1_image, self.qr1_panel)
            self.qr_code1_label.config(text=f"QR Code: {qr_text}")

    def load_and_process_image2(self):
        """Load second image, extract QR region and display."""
        self.img2_path = filedialog.askopenfilename()
        if self.img2_path:
            self.image2 = cv2.imread(self.img2_path)
            self.qr2_image, qr_text, polygon = self.QR_reader.extract_qr_region(self.image2)
            self.image2 = self.draw_polygon_on_image(self.image2, polygon)
            self.display_image(self.image2, self.img2_panel)
            # self.display_qr_region(self.qr2_image, self.qr2_panel)
            self.qr_code2_label.config(text=f"QR Code: {qr_text}")

    def draw_polygon_on_image(self, image, polygon):
        """Draw polygon on image based on detected QR code corners."""
        if polygon is not None and len(polygon) == 4:
            for i in range(4):
                pt1 = tuple(polygon[i])
                pt2 = tuple(polygon[(i + 1) % 4])
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)
                cv2.circle(image, pt1, 5, (0, 0, 255), -1)
            
        return image

    def display_image(self, cvImage, panel):
        """Display image from file path on given panel."""
        pil_img = Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
        # resize to have 400 px height
        pil_img = pil_img.resize((int(pil_img.width * 400 / pil_img.height), 400))
        # img = Image.open(path).resize((200, 200))
        img_tk = ImageTk.PhotoImage(pil_img)
        panel.configure(image=img_tk)
        panel.image = img_tk

    def display_qr_region(self, qr_img, panel):
        """Display the extracted QR code region on given panel."""
        qr_pil = Image.fromarray(cv2.cvtColor(qr_img, cv2.COLOR_BGR2RGB)).resize((200, 200))
        qr_img_tk = ImageTk.PhotoImage(qr_pil)
        panel.configure(image=qr_img_tk)
        panel.image = qr_img_tk

    def get_qr_code_data(self, qr_image):
        """Get QR code text and bounding rect from QR image."""
        return self.QR_reader.get_qr_data(qr_image)

    # ---------------- Mask and Rectangle Utilities ----------------
    def create_mask_for_region(self, shape, region):
        """Create binary mask with region set to 0 and rest 255."""
        mask = np.ones(shape, dtype=np.uint8) * 255
        if region:
            x, y, w, h = region
            mask[y:y+h, x:x+w] = 0
        return mask

    def calculate_padding(self, rect, padding_factor=4.2):
        """Calculate padding size based on rectangle size."""
        if rect is None:
            return 0
        # rect expected as (x, y, w, h) or object with width and height attributes
        if hasattr(rect, 'width') and hasattr(rect, 'height'):
            w, h = rect.width, rect.height
        else:
            w, h = rect[2], rect[3]
        padding = int(np.sqrt(w * h) / padding_factor) // 2
        return padding

    def expand_rectangle(self, rect, padding):
        """Expand rectangle by padding on all sides."""
        if rect is None:
            return None
        if hasattr(rect, 'left') and hasattr(rect, 'top') and hasattr(rect, 'width') and hasattr(rect, 'height'):
            x = rect.left - padding
            y = rect.top - padding
            w = rect.width + 2 * padding
            h = rect.height + 2 * padding
        else:
            x, y, w, h = rect
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding
        return (x, y, w, h)

    def draw_rectangle_on_image(self, image, rect, color=(255, 0, 0), thickness=2, alpha=0.2):
        """Draw semi-transparent rectangle with border on image."""
        if rect is None:
            return

        x, y, w, h = map(int, rect)
        overlay = image.copy()
        output = image.copy()

        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

        # Blend overlay with the original image
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, image)

        # Draw rectangle border
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    # ---------------- Feature Extraction ----------------
    def extract_features(self, img_gray, mask):
        """Extract keypoints and descriptors based on selected feature method."""
        method = self.feature_method.get()
        if method == "SIFT":
            sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.09, edgeThreshold=10, sigma=1.6)
            return sift.detectAndCompute(img_gray, mask)
        elif method == "ORB":
            orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31,
                                 firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=49)
            return orb.detectAndCompute(img_gray, mask)
        else:
            messagebox.showwarning("Feature Extraction", "Unknown feature extraction method.")
            return None, None

    # ---------------- Feature Matching ----------------
    def match_descriptors(self, des1, des2):
        """Match descriptors using appropriate matcher based on feature method."""
        method = self.feature_method.get()
        if method == "SIFT":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            return good_matches
        elif method == "ORB":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        else:
            messagebox.showwarning("Feature Matching", "Unknown feature extraction method.")
            return []

    # ---------------- Verification Logic ----------------
    def verify_images(self):
        """Main verification logic comparing two images."""
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Input Error", "Please load both images.")
            return

        qr1_text, qr1_rect = self.get_qr_code_data(self.qr1_image)
        qr2_text, qr2_rect = self.get_qr_code_data(self.qr2_image)

        qr1 = self.qr1_image.copy()
        qr2 = self.qr2_image.copy()


        if qr1_rect is None or qr2_rect is None:
            self.result_label.config(text="QR code not detected in one of the images.")
            return

        # Expand QR code rectangles with padding
        qr1_padding = self.calculate_padding(qr1_rect)
        qr2_padding = self.calculate_padding(qr2_rect)

        qr1_rect_expanded = self.expand_rectangle(qr1_rect, qr1_padding)
        qr2_rect_expanded = self.expand_rectangle(qr2_rect, qr2_padding)

        # Draw rectangles on QR images for visualization
        self.draw_rectangle_on_image(qr1, qr1_rect_expanded)
        self.draw_rectangle_on_image(qr2, qr2_rect_expanded)

        # Convert QR images to grayscale
        gray1 = cv2.cvtColor(qr1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(qr2, cv2.COLOR_BGR2GRAY)

        # Update QR code match label
        self.qr_code_match_label.config(text=f"QR Code Match: {qr1_text == qr2_text}")

        # Create masks to exclude QR code regions from feature extraction
        mask1 = self.create_mask_for_region(gray1.shape, qr1_rect_expanded)
        mask2 = self.create_mask_for_region(gray2.shape, qr2_rect_expanded)

        # Extract features with mask
        kp1, des1 = self.extract_features(gray1, mask1)
        kp2, des2 = self.extract_features(gray2, mask2)

        if des1 is None or des2 is None:
            self.result_label.config(text="Could not extract features.")
            return

        # Match features
        matches = self.match_descriptors(des1, des2)

        # Display result summary
        result_text = f"Number of Good Matches: {len(matches)}"
        self.result_label.config(text=result_text)

        # Draw matches image for visualization
        match_img = cv2.drawMatches(qr1, kp1, qr2, kp2, matches, None, flags=2)
        match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(match_img_rgb)
        ax.axis('off')
        plt.tight_layout()
        plt.title("Matches between QR Codes", fontsize=14, fontweight='bold')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Remove previous canvas if any
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = QRVerifierApp(root)
    root.mainloop()
