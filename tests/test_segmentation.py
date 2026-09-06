from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocessing import banana_mask, fill_mask_holes


def test_fill_mask_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    # Punch a void/hole in the center
    mask[45:55, 45:55] = 0

    assert np.count_nonzero(mask[45:55, 45:55]) == 0
    filled = fill_mask_holes(mask)
    assert np.all(filled[20:80, 20:80] == 255)
    assert np.all(filled[:20, :] == 0)


def test_specular_glare_hole_filling():
    # Yellow banana with a bright white specular reflection highlight in the middle
    image = np.full((224, 224, 3), 245, dtype=np.uint8)
    image[50:170, 50:170] = (235, 205, 35)  # Fruit
    image[100:120, 100:120] = (255, 255, 255)  # Specular glare reflection

    mask = banana_mask(image)
    # The specular reflection center must be solid foreground (255), not a hollow hole
    assert mask[110, 110] == 255
    # Outer background should remain background (0)
    assert mask[10, 10] == 0


def test_tightly_cropped_banana_mask():
    # Banana filling almost the entire crop (touching borders), with small white corners
    image = np.full((160, 200, 3), (230, 200, 40), dtype=np.uint8)
    image[:20, :20] = 245
    image[:20, -20:] = 245

    mask = banana_mask(image)
    # Center of fruit must be detected as fruit
    assert mask[80, 100] == 255
    # Fruit should comprise the majority of the mask
    assert float(np.count_nonzero(mask)) / mask.size > 0.85
    # Corners should be excluded
    assert mask[5, 5] == 0


def test_multi_banana_cluster_preservation():
    # Two distinct banana fingers in a cluster
    image = np.full((224, 224, 3), 245, dtype=np.uint8)
    image[30:80, 40:180] = (235, 205, 35)   # Finger 1
    image[120:170, 40:180] = (235, 205, 35)  # Finger 2

    mask = banana_mask(image)
    # Both fingers should be retained
    assert mask[55, 100] == 255
    assert mask[145, 100] == 255
    # The gap between them should be background
    assert mask[100, 100] == 0


def test_adaptive_otsu_segmentation_clean_backgrounds():
    for bg_color in [(245, 245, 245), (15, 15, 15), (200, 180, 220)]:
        image = np.full((224, 224, 3), bg_color, dtype=np.uint8)
        image[60:160, 60:160] = (235, 205, 35)  # Yellow banana

        mask = banana_mask(image)
        assert mask[110, 110] == 255
        assert mask[10, 10] == 0
        ratio = float(np.count_nonzero(mask)) / mask.size
        assert 0.15 < ratio < 0.30
