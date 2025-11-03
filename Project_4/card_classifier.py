# assuming input is oriented cropped card image
# finds all components on the card image
# compares located rank and suit to pre-stored masks
# output is card rank and suit
import cv2
import numpy as np
import matplotlib.pyplot as plt

# test_card_path = 'C:\\ImageProcessing\\Project_4\\test_cards\\A_spades.jpg'
test_cards_folder = 'C:\\ImageProcessing\\Project_4\\test_cards\\'
test_cards = ['7_clubs.jpg', '6_diamonds.jpg', 'A_diamonds.jpg', 'J_hearts.jpg', 'K_clubs.jpg', 'K_diamonds.jpg', 'A_hearts.jpg']
              
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

def main():
    for test_card_path in [test_cards_folder + name for name in test_cards]:
        original = cv2.imread(test_card_path, cv2.IMREAD_GRAYSCALE)
        # scale to fixed pixel dimensions
        test_card = cv2.resize(original, (200, 300), interpolation=cv2.INTER_AREA)
        test_card = dynamic_binarize(test_card)
        roi = test_card[5:80, 0:35]
        # mask inner region to avoid border artifacts
        # plt.imshow(roi, cmap='gray'); plt.title('ROI'); plt.axis('off'); plt.show()
        suit_masks = []  # load suit masks from suit_base_path
        rank_masks = []  # load rank masks from rank_base_path

        for i in range(4):  # assuming 4 suits
            suit_mask = cv2.imread(f'{suit_base_path}{i}.png', cv2.THRESH_BINARY)
            suit_masks.append(suit_mask)
        for i in range(13):  # assuming 13 ranks
            rank_mask = cv2.imread(f'{rank_base_path}{i+1}.png', cv2.THRESH_BINARY)
            rank_masks.append(rank_mask)
        # show ROI boxes on test card
        # cv2.imshow('Test Card', test_card)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        components = find_all_components(roi)

        rank = analyze_component(components[0], rank_masks)
        suit = analyze_component(components[1], suit_masks)
        suit_mapping = {0: 'Hearts', 1: 'Diamonds', 2: 'Clubs', 3: 'Spades'}
        rank_mapping = {i+1: str(i+1) for i in range(10)}
        rank_mapping.update({11: 'J', 12: 'Q', 13: 'K', 1: 'A'})
        # compare predicted rank with mask
        # plt.imshow(components[0], cmap='gray'); plt.title('Detected Rank Component'); plt.axis('off'); plt.show()
        # plt.imshow(rank_masks[rank[0]], cmap='gray'); plt.title('Best Matching Rank Mask'); plt.axis('off'); plt.show()
        print(f'Detected Card: {rank_mapping[rank[0]+1]} of {suit_mapping[suit[0]]} (Rank Score: {1-rank[1]:.3f}, Suit Score: {1-suit[1]:.3f})')
        # cv2.imshow('Masked Test Card', roi)
        # cv2.waitKey(0)
        plt.imshow(original, cmap='gray'); plt.title(rank_mapping[rank[0]+1] + " of " + suit_mapping[suit[0]]); plt.axis('off'); plt.show()
if __name__ == "__main__":
    main()