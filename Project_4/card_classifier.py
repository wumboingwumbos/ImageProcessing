# assuming input is oriented cropped card image
# locates rank and suit on card image
# compares located rank and suit to pre-stored masks
# output is card rank and suit
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def classify_card(card_image, rank_masks, suit_masks, rank_names=None, suit_names=None):
    # convert to black and white
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # initial guess at rank and suit locations to start connected component analysis
    # suit position
    suit_roi = bw[20:120, 10:60]
    # rank position
    rank_roi = bw[10:110, 10:60]
    # Connected components analysis to isolate rank and suit
    num_labels_suit, labels_im_suit = cv2.connectedComponents(suit_roi)
    num_labels_rank, labels_im_rank = cv2.connectedComponents(rank_roi)
    # Extract the largest component as the suit
    suit_component = np.zeros_like(suit_roi)
    suit_component[labels_im_suit == 1] = 255
    # Extract the largest component as the rank
    rank_component = np.zeros_like(rank_roi)  
    rank_component[labels_im_rank == 1] = 255
    # Resize components to match mask sizes
    suit_component_resized = cv2.resize(suit_component, (suit_masks[0].shape[1], suit_masks[0].shape[0]))
    rank_component_resized = cv2.resize(rank_component, (rank_masks[0].shape[1], rank_masks[0].shape[0]))
    # Compare suit component to suit masks
    best_suit = None
    best_suit_score = float('inf')
    
    # Jaccard similarity for better matching
    for i, suit_mask in enumerate(suit_masks):
        intersection = np.logical_and(suit_component_resized > 0, suit_mask > 0)
        union = np.logical_or(suit_component_resized > 0, suit_mask > 0)
        jaccard_index = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        score = 1 - jaccard_index  # lower score is better
        if score < best_suit_score:
            best_suit_score = score
            best_suit = i  # index of the best matching suit

    # Compare rank component to rank masks
    best_rank = None
    best_rank_score = float('inf')
    for i, rank_mask in enumerate(rank_masks):
        intersection = np.logical_and(rank_component_resized > 0, rank_mask > 0)
        union = np.logical_or(rank_component_resized > 0, rank_mask > 0)
        jaccard_index = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        score = 1 - jaccard_index  # lower score is better
        if score < best_rank_score:
            best_rank_score = score
            best_rank = i  # index of the best matching rank

    # Map indices back to names if name lists were provided
    rank_result = best_rank
    suit_result = best_suit
    if rank_names is not None and best_rank is not None and best_rank < len(rank_names):
        rank_result = rank_names[best_rank]
    if suit_names is not None and best_suit is not None and best_suit < len(suit_names):
        suit_result = suit_names[best_suit]

    return rank_result, suit_result


def main():
    # Load pre-stored rank and suit masks
    rank_masks = []
    rank_names = []
    suit_masks = []
    suit_names = []

    # Load all rank masks from the rank_masks/ directory. Filenames (without extension)
    # are used as the rank identifiers (e.g. 'A', '2', 'J').
    for filepath in sorted(glob.glob(os.path.join('rank_masks', '*.png'))):
        name = os.path.splitext(os.path.basename(filepath))[0]
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: failed to load rank mask: {filepath}")
            continue
        rank_masks.append(mask)
        rank_names.append(name)

    # Load all suit masks from the suit_masks/ directory. Filenames are used as identifiers
    # (e.g. 'C' -> clubs, 'D' -> diamonds).
    for filepath in sorted(glob.glob(os.path.join('suit_masks', '*.png'))):
        name = os.path.splitext(os.path.basename(filepath))[0]
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: failed to load suit mask: {filepath}")
            continue
        suit_masks.append(mask)
        suit_names.append(name)
    # Load test cropped and oriented card image
    card_image = cv2.imread('test_card.png')

    rank_name, suit_name = classify_card(card_image, rank_masks, suit_masks, rank_names, suit_names)
    print(f'Classified Card: Rank {rank_name}, Suit {suit_name}')

if __name__ == "__main__":
    main()