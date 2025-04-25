import cv2
import numpy as np
from pyzbar.pyzbar import decode

def recognize_qr_code(image_path):
    ext = image_path.split(".")[-1]
    if ext not in ["jpg", "jpeg", "png"]:
        print("Invalid image format. Please provide a JPG, JPEG, or PNG image.")
        return
    # Load the image
    image = cv2.imread(image_path)

    # Decode the QR code
    decoded_objects = decode(image)
    # qr code size is 4.2 cm x 4.2 cm

    # Process and print the results
    for obj in decoded_objects:
        # qr code size in pixels
        qr_code_size = obj.rect.width * obj.rect.height
        print(f"QR Code Size: {qr_code_size} pixels")
        qr_code_pixel_size = qr_code_size / (4.2 * 4.2)
        print(f"QR Code Size: {qr_code_pixel_size} pixels/cm")
        # margine should be 1 cm
        margin = round(np.sqrt(qr_code_pixel_size))
        print(f"Margin: {margin} pixels")
        #get the qr code region with the margin
        x, y, w, h = obj.rect
        # cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255),-1)
        qr_code_region = image[y - int(margin):y + h + int(margin), x - int(margin):x + w + int(margin)]

        # Save the QR code region
        qr_code_filename = image_path.replace("."+ext, "_qr_code.png")
        cv2.imwrite(qr_code_filename, qr_code_region)
        qr_code_rect = margin, margin, w, h
        keypoints = generate_sift_signature(qr_code_region=qr_code_region, qr_code_rect=qr_code_rect)
        # Save the image with keypoints
        keypoints_filename = image_path.replace("."+ext, "_keypoints.png")
        cv2.imwrite(keypoints_filename, keypoints)
        # Draw a rectangle around the QR code
        
    


    if not decoded_objects:
        print("No QR code found in the image.")

def generate_sift_signature(qr_code_region, qr_code_rect=None):
    # Load image in grayscale
    image = cv2.cvtColor(qr_code_region, cv2.COLOR_BGR2GRAY)
    mask = np.ones(image.shape, dtype=np.uint8)*255
    x, y, w, h = qr_code_rect
    mask[y:y + h, x:x + w] = 0

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(image, mask)

    print(f"Number of keypoints: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape}")

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Save the image with keypoints
    # Display the image with keypoints
    cv2.imshow("SIFT Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_with_keypoints

# Example usage
if __name__ == "__main__":
    # image_path = "photos/IMG_7173.jpeg"  # Replace with your image path
    recognize_qr_code("photos/IMG_7173.jpeg")
    recognize_qr_code("photos/IMG_7174.jpeg")
    # generate_sift_signature("photos/IMG_7173_qr_code.png")
    # generate_sift_signature("photos/IMG_7174_qr_code.png")