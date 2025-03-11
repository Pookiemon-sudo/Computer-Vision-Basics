import cv2
import json

# Global variables
drawing = False
ix, iy = -1, -1
boxes = []
labels = []
current_label = ""

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, boxes, labels, current_label
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        boxes.append((ix, iy, x, y))
        labels.append(current_label)
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.putText(img, current_label, (ix, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Image", img)

def annotate_image(image_path, label):
    global img, current_label
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return
    
    current_label = label
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)
    
    while True:
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return boxes, labels

# Example usage
image_path = "animal.jpg"  # Change to your image path
animal_label = "cat"  # Change to the correct label for the image

boxes, labels = annotate_image(image_path, animal_label)

# Save annotations
annotations = {"image": image_path, "annotations": []}
for box, label in zip(boxes, labels):
    annotations["annotations"].append({"box": box, "label": label})

with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=4)

print("Annotations saved to annotations.json")
