import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        self.qr1_region = None
        self.qr2_region = None

        self.setup_gui()

    def setup_gui(self):
        tk.Button(self.root, text="Load Image 1", command=self.load_image1).pack()
        tk.Button(self.root, text="Load Image 2", command=self.load_image2).pack()
        tk.Button(self.root, text="Verify", command=self.verify_images).pack()

        placeholder = Image.new('RGB', (200, 200), color='gray')
        placeholder_img = ImageTk.PhotoImage(placeholder)

        frame1 = tk.Frame(self.root)
        frame1.pack()
        self.img1_panel = tk.Label(frame1, image=placeholder_img)
        self.img1_panel.image = placeholder_img
        self.img1_panel.grid(row=0, column=0)
        self.qr1_panel = tk.Label(frame1, image=placeholder_img)
        self.qr1_panel.image = placeholder_img
        self.qr1_panel.grid(row=0, column=1)
        tk.Label(frame1, text="Image 1", font=("Arial", 10, "bold")).grid(row=1, column=0)
        self.qr_code1_label = tk.Label(frame1, text="QR Code 1 Region", font=("Arial", 10, "bold"))
        self.qr_code1_label.grid(row=1, column=1)

        frame2 = tk.Frame(self.root)
        frame2.pack()
        self.img2_panel = tk.Label(frame2, image=placeholder_img)
        self.img2_panel.image = placeholder_img
        self.img2_panel.grid(row=0, column=0)
        self.qr2_panel = tk.Label(frame2, image=placeholder_img)
        self.qr2_panel.image = placeholder_img
        self.qr2_panel.grid(row=0, column=1)
        tk.Label(frame2, text="Image 2", font=("Arial", 10, "bold")).grid(row=1, column=0)
        self.qr_code2_label = tk.Label(frame2, text="QR Code 2 Region", font=("Arial", 10, "bold"))
        self.qr_code2_label.grid(row=1, column=1)

        self.result_label = tk.Label(self.root, text="Results will appear here", font=("Arial", 14))
        self.result_label.pack()
        self.qr_code_match_label = tk.Label(self.root, text="QR Code Match: ", font=("Arial", 14))
        self.qr_code_match_label.pack()

    def rectify_qr(self, image, polygon, size=420, margin_ratio=0.25):
        if len(polygon) != 4:
            return None
        src_pts = np.array([(pt.x, pt.y) for pt in polygon], dtype='float32')
        margin = int(size * margin_ratio)
        padded_size = size + 2 * margin
        dst_pts = np.array([
            [margin, margin],
            [margin + size - 1, margin],
            [margin + size - 1, margin + size - 1],
            [margin, margin + size - 1]
        ], dtype='float32')
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, M, (padded_size, padded_size))

    def extract_qr_region(self, image, qr_panel):

        decoded = decode(image)

        if decoded and len(decoded[0].polygon) == 4:
            qr_text = decoded[0].data.decode("utf-8")
            rectified = self.rectify_qr(image, decoded[0].polygon)

            if rectified is not None:
                qr_pil = Image.fromarray(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)).resize((200, 200))
                qr_img = ImageTk.PhotoImage(qr_pil)
                qr_panel.configure(image=qr_img)
                qr_panel.image = qr_img

                return rectified, qr_text

        messagebox.showwarning("QR Code", "QR code not detected or not suitable for rectification.")
        return None, None

    def load_image1(self):
        self.img1_path = filedialog.askopenfilename()
        if self.img1_path:
            self.display_image(self.img1_path, self.img1_panel)
            self.image1 = cv2.imread(self.img1_path)
            self.qr1_image, qr_text = self.extract_qr_region(self.image1, self.qr1_panel)
            self.qr_code1_label.config(text=f"QR Code: {qr_text}")

    def load_image2(self):
        self.img2_path = filedialog.askopenfilename()
        if self.img2_path:
            self.display_image(self.img2_path, self.img2_panel)
            self.image2 = cv2.imread(self.img2_path)
            self.qr2_image, qr_text = self.extract_qr_region(self.image2, self.qr2_panel)
            self.qr_code2_label.config(text=f"QR Code: {qr_text}")

    def display_image(self, path, panel):
        img = Image.open(path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

    def decode_qr(self, image):
        decoded = decode(image)
        if decoded:
            return decoded[0].rect, decoded[0].data.decode("utf-8")
        return None, "No QR code found"
        

    def create_mask(self, shape, region):
        mask = np.ones(shape, dtype=np.uint8) * 255
        if region:
            x, y, w, h = region
            mask[y:y+h, x:x+w] = 0
        return mask

    def extract_sift(self, img, mask):
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(img, mask)

    def match_features(self, des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        return [m for m, n in matches if m.distance < 0.75 * n.distance]

    def verify_images(self):
        if not self.img1_path or not self.img2_path:
            messagebox.showwarning("Input Error", "Please load both images.")
            return

        qr1_rect, qr1_text = self.decode_qr(self.qr1_image)
        qr2_rect, qr2_text = self.decode_qr(self.qr2_image)
        

        gray1 = cv2.cvtColor(self.qr1_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.qr2_image, cv2.COLOR_BGR2GRAY)

        if qr1_rect is None or qr2_rect is None:
            self.result_label.config(text="QR code not detected in one of the images.")
            return
        
        self.qr_code_match_label.config(text=f"QR Code Match: {qr1_text == qr2_text}")
            
        

        mask1 = self.create_mask(gray1.shape, qr1_rect)
        mask2 = self.create_mask(gray2.shape, qr2_rect)

        kp1, des1 = self.extract_sift(gray1, mask1)
        kp2, des2 = self.extract_sift(gray2, mask2)

        if des1 is None or des2 is None:
            self.result_label.config(text="Could not extract features.")
            return

        matches = self.match_features(des1, des2)

        result_text = f"Number of Good Matches: {len(matches)}"
        self.result_label.config(text=result_text)

        match_img = cv2.drawMatches(self.qr1_image, kp1, self.qr2_image, kp2, matches, None, flags=2)
        match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(match_img_rgb)
        ax.axis('off')
        # set tight layout
        plt.tight_layout()
        plt.title("Matches between QR Codes", fontsize=14, fontweight='bold')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = QRVerifierApp(root)
    root.mainloop()
