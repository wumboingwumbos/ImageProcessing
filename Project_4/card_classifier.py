# assuming input is oriented cropped card image
# finds all components on the card image
# compares located rank and suit to pre-stored masks
# output is card rank and suit
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
test_card_path = 'C:\\ImageProcessing\\upright_Testimage1.tif'
# test_cards_folder = 'C:\\ImageProcessing\\Project_4\\test_cards\\'
# test_cards = ['7_clubs.jpg', '6_diamonds.jpg', 'A_diamonds.jpg', 'J_hearts.jpg', 'K_clubs.jpg', 'K_diamonds.jpg', 'A_hearts.jpg']
              
rank_base_path = 'C:\\ImageProcessing\\Project_4\\rank_masks_bw\\'
suit_base_path = 'C:\\ImageProcessing\\Project_4\\suit_masks_bw\\'

def analyze_component(component, masks):
    best_index = None
    best_score = float('inf')
    binary_comp = component > 0
    for i, mask in enumerate(masks):
        resized_mask = cv2.resize(mask, (binary_comp.shape[1], binary_comp.shape[0]), interpolation=cv2.INTER_NEAREST)
        intersection = np.logical_and(binary_comp, resized_mask > 0)
        union = np.logical_or(binary_comp, resized_mask > 0)
        jaccard_index = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        score = 1 - jaccard_index  # lower score is better
        if score < best_score:
            best_score = score
            best_index = i
    # show side by side comparison of best mask and component
    comparison = np.hstack((binary_comp.astype(np.uint8) * 255, resized_mask.astype(np.uint8) * 255))
    # plt.imshow(comparison, cmap='gray'); plt.title('Component (left) vs Best Mask (right)'); plt.axis('off'); plt.show()
    return best_index, best_score
 
def find_all_components(binary_card):
    num_labels, labels_im = cv2.connectedComponents(binary_card)
    # print(f'Number of components found: {num_labels}')
    components = []
    for label in range(0, num_labels): 
        component = np.zeros_like(binary_card)
        # limit max height and width of component to avoid large background
        if np.sum(labels_im == label) > 1000 or np.sum(labels_im == label) < 50:
            continue
        component[labels_im == label] = 255
        # crop to bounding box
        ys, xs = np.where(component > 0)
        if ys.size > 0 and xs.size > 0:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            component = component[y_min:y_max+1, x_min:x_max+1]
            if component.shape[0]/component.shape[1] > 3.5 or component.shape[1]/component.shape[0] > 3.5:
                continue
            # print(f'component size: {component.shape}')
        components.append(component)
        # cv2.imshow(f'Component {label}', component)
        if len(components) == 2:
            break
    # cv2.waitKey(0)    
    return components

def dynamic_binarize(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    return binary

def classify_and_annotate(input_img):
    """Classify a cropped/oriented card image (BGR or grayscale ndarray) and
    return an annotated BGR image plus the predicted rank and suit names.
    """
    # accept either a path or an image array
    if isinstance(input_img, str):
        img = cv2.imread(input_img, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_img}")
    else:
        img = input_img.copy()

    # work on a resized copy for stable ROI coordinates
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    resized = cv2.resize(gray, (200, 300), interpolation=cv2.INTER_AREA)
    binary = dynamic_binarize(resized)
    roi = binary[5:80, 0:35]

    components = find_all_components(roi)
    if len(components) < 2:
        # return original annotated with 'Not Found'
        annotated = ensure_color(img)
        cv2.putText(annotated, 'Rank/Suit not found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        return annotated, None, None

    # load masks
    suit_masks = [cv2.imread(f'{suit_base_path}{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(4)]
    rank_masks = [cv2.imread(f'{rank_base_path}{i+1}.png', cv2.IMREAD_GRAYSCALE) for i in range(13)]

    rank = analyze_component(components[0], rank_masks)
    suit = analyze_component(components[1], suit_masks)
    suit_mapping = {0: 'Hearts', 1: 'Diamonds', 2: 'Clubs', 3: 'Spades'}
    rank_mapping = {i+1: str(i+1) for i in range(10)}
    rank_mapping.update({11: 'J', 12: 'Q', 13: 'K', 1: 'A'})

    rank_name = rank_mapping.get(rank[0]+1, str(rank[0]+1))
    suit_name = suit_mapping.get(suit[0], str(suit[0]))

    final = ensure_color(resized)
    label = f"{rank_name} of {suit_name}"
    font_size = 0.5
    position = (50, 20)
    cv2.putText(final, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, (200,70,30), 2)
    return final, rank_name, suit_name

def ensure_color(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def main(image):
    # backward-compatible entry point: accept path or image ndarray
    final, rank_name, suit_name = classify_and_annotate(image)
    # if final is not None:
    #     cv2.imshow(f"{rank_name} of {suit_name}", image)
    #     cv2.waitKey(0)
    return final, rank_name, suit_name
if __name__ == "__main__":
    test_card_folder = 'C:\\ImageProcessing\\Project_4\\test_cards\\'
    for filename in os.listdir(test_card_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            test_card_path = os.path.join(test_card_folder, filename)
            test_card = cv2.imread(test_card_path, cv2.IMREAD_COLOR)
            final, rank_name, suit_name = main(test_card)
            cv2.imshow(f"{rank_name} of {suit_name}", final)
    cv2.waitKey(0)
