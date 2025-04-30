import cv2
import numpy as np


class QRReader:
    def __init__(self, library='pyzbar', reader_name='QRReader'):
        self.reader_name = reader_name
        self.library = library  # 'pyzbar' or 'opencv'

    def rectify_qr_code(self, image, polygon, size=420, margin_ratio=0.5):
        """
        Applies a perspective transform to the QR code region to obtain a rectified square image.
        :param image: The original image.
        :param polygon: List of 4 points defining the QR code polygon.
        :param size: Desired size of the rectified QR code image.
        :param margin_ratio: Margin ratio around the QR code in the output image.
        :return: Rectified QR code image or None if polygon is invalid.
        """
        if len(polygon) != 4:
            return None
        src_pts = np.array([(pt.x, pt.y) if hasattr(pt, 'x') else (pt[0], pt[1]) for pt in polygon], dtype='float32')
        margin = int(size * margin_ratio)
        padded_size = size + 2 * margin
        dst_pts = np.array([
            [margin, margin],
            [margin + size - 1, margin],
            [margin + size - 1, margin + size - 1],
            [margin, margin + size - 1]
        ], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified_img = cv2.warpPerspective(image, transform_matrix, (padded_size, padded_size))
        return rectified_img

    def extract_qr_region(self, image):
        """
        Detects and extracts the QR code region from the image, rectifies it, and returns the rectified image and decoded text.
        :param image: Input image.
        :return: Tuple of (rectified QR code image or original image, decoded QR text or message).
        """
        decoded, qr_text, orientation, polygon, rect = self._decode_qr_code(image)
        if decoded and polygon is not None and len(polygon) == 4:
            # Rotate image according to QR code orientation
            if orientation == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif orientation == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rectified_img = self.rectify_qr_code(image, polygon, margin_ratio=0.3)
            if rectified_img is not None:
                return rectified_img, qr_text, polygon
        return image, "No QR code detected"

    def extract_qr_region_from_file(self, image_path):
        """
        Loads an image from a file and extracts the QR code region.
        :param image_path: Path to the image file.
        :return: Rectified QR code image and decoded text.
        """
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image"
        return self.extract_qr_region(image)

    def get_qr_data(self, image):
        """
        Decodes the QR code data from the given image.
        :param image: Image containing the QR code.
        :return: Tuple of (decoded QR text or None, bounding rectangle or None).
        """
        decoded, qr_text, orientation, polygon, rect = self._decode_qr_code(image)
        if decoded:
            return qr_text, rect
        return None, None

    def _decode_qr_code(self, image):
        """
        Internal method to decode QR code from the image using the selected library.
        :param image: Input image.
        :return: Tuple (decoded: bool, qr_text: str or None, orientation: int or None, polygon: list or None, rect: tuple or None)
        """
        if self.library == 'pyzbar':
            from pyzbar.pyzbar import decode
            decoded_objects = decode(image)
            if decoded_objects:
                obj = decoded_objects[0]
                polygon = obj.polygon
                rect = obj.rect
                orientation = getattr(obj, 'orientation', 0)
                qr_text = obj.data.decode('utf-8')
                return True, qr_text, orientation, polygon, rect
            else:
                return False, None, None, None, None

        elif self.library == 'opencv':
            # Ensure image is in BGR format without alpha channel
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            qr_detector = cv2.QRCodeDetector()
            data, points, _ = qr_detector.detectAndDecode(image)
            if points is not None and data:
                # points is a numpy array of shape (4,2)
                polygon = points.reshape(4, 2)
                rect = cv2.boundingRect(polygon)
                orientation = 0  # OpenCV does not provide orientation
                return True, data, orientation, polygon, rect
            else:
                return False, None, None, None, None

        else:
            # Unsupported library
            return False, None, None, None, None