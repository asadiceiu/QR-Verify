import cv2
from pyzbar.pyzbar import decode

img = cv2.imread("test-photos/IMG_7238.png")
qr_detector = cv2.QRCodeDetector()
data, points, qr_image = qr_detector.detectAndDecode(img)

if points is not None:
    print("QR Data:", data)
    print("Points:", points)
    points = points[0] if len(points) == 1 else points
    for i in range(4):
        pt1 = tuple(points[i].astype(int))
        pt2 = tuple(points[(i + 1) % 4].astype(int))
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    cv2.imshow("Detected QR", img)
    cv2.imshow("Rectified QR", qr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No QR code found.")
