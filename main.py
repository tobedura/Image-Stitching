import numpy as np
import cv2 as cv
import glob

# Load all images from data/ (order-agnostic)
image_paths = sorted(glob.glob('data/*.jpg'))
images = [cv.imread(p) for p in image_paths]
assert all(img is not None for img in images), 'Cannot read images'
n = len(images)

fdetector = cv.BRISK_create()
fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')

# Pre-compute features once per image
features = [fdetector.detectAndCompute(img, None) for img in images]

def match_pair(i, j):
    """Match images[i] and images[j]. Returns (inlier_count, dx).
    dx > 0 means images[j] is positioned to the right of images[i]."""
    kpts_i, desc_i = features[i]
    kpts_j, desc_j = features[j]
    knn = fmatcher.knnMatch(desc_i, desc_j, k=2)
    good = [m for m, nn in knn if m.distance < 0.75 * nn.distance]
    if len(good) < 4:
        return 0, 0.0
    pts_i = np.float32([kpts_i[m.queryIdx].pt for m in good])
    pts_j = np.float32([kpts_j[m.trainIdx].pt for m in good])
    H, mask = cv.findHomography(pts_j, pts_i, cv.RANSAC, 5.0)
    if H is None:
        return 0, 0.0
    inlier_idx = mask.ravel().astype(bool)
    avg_x_i = float(pts_i[inlier_idx, 0].mean())
    dx = avg_x_i - images[i].shape[1] / 2
    return int(inlier_idx.sum()), dx

# Pairwise matching
inliers = np.zeros((n, n), dtype=int)
dx = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            inliers[i, j], dx[i, j] = match_pair(i, j)

print('Pairwise inlier counts:')
print(inliers)

# Reference image = most total inlier matches (likely the central one)
ref = int(inliers.sum(axis=1).argmax())
positions = [0.0 if i == ref else dx[ref, i] for i in range(n)]
order = sorted(range(n), key=lambda i: positions[i])
print(f'Reference: {image_paths[ref]}')
print(f'Order (left to right): {[image_paths[i] for i in order]}')

def gaussian_pyr(img, levels):
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(cv.pyrDown(pyr[-1]))
    return pyr

def laplacian_pyr(img, levels):
    gp = gaussian_pyr(img, levels)
    lp = [gp[-1]]
    for i in range(levels - 1, 0, -1):
        up = cv.pyrUp(gp[i], dstsize=(gp[i - 1].shape[1], gp[i - 1].shape[0]))
        lp.append(gp[i - 1] - up)
    return lp[::-1]

def multi_band_blend(img1, img2, mask, levels=5):
    """Blend img1 (where mask~1) and img2 (where mask~0) using Laplacian pyramids."""
    lp1 = laplacian_pyr(img1, levels)
    lp2 = laplacian_pyr(img2, levels)
    mp = gaussian_pyr(mask, levels)

    blended = []
    for l1, l2, m in zip(lp1, lp2, mp):
        m3 = m[..., None]
        blended.append(l1 * m3 + l2 * (1.0 - m3))

    result = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        result = cv.pyrUp(result, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        result = result + blended[i]
    return np.clip(result, 0, 255).astype(np.uint8)


def stitch_pair(left, right):
    kpts_l, desc_l = fdetector.detectAndCompute(left, None)
    kpts_r, desc_r = fdetector.detectAndCompute(right, None)
    knn = fmatcher.knnMatch(desc_l, desc_r, k=2)
    match = [m for m, nn in knn if m.distance < 0.75 * nn.distance]

    pts_l = np.float32([kpts_l[m.queryIdx].pt for m in match])
    pts_r = np.float32([kpts_r[m.trainIdx].pt for m in match])
    H, _ = cv.findHomography(pts_r, pts_l, cv.RANSAC)
    canvas_w = left.shape[1] + right.shape[1]
    canvas_h = max(left.shape[0], right.shape[0])

    # Place each image on the same canvas (no overwrite — both kept for blending)
    warped_right = cv.warpPerspective(right, H, (canvas_w, canvas_h))
    canvas_left = np.zeros((canvas_h, canvas_w, 3), dtype=left.dtype)
    canvas_left[:left.shape[0], :left.shape[1]] = left

    # Distance-based mask: high inside each image, fades to 0 at borders
    mask_l = (cv.cvtColor(canvas_left, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    mask_r = (cv.cvtColor(warped_right, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    dist_l = cv.distanceTransform(mask_l, cv.DIST_L2, 5)
    dist_r = cv.distanceTransform(mask_r, cv.DIST_L2, 5)
    mask = dist_l / (dist_l + dist_r + 1e-6)

    # Pyramid depth based on canvas size
    min_dim = min(canvas_h, canvas_w)
    levels = max(2, min(5, int(np.log2(min_dim)) - 3))

    return multi_band_blend(canvas_left, warped_right, mask, levels=levels)

# Stitch in determined order: leftmost as base, append each next image to the right
img_merged = images[order[0]]
for k in range(1, len(order)):
    img_merged = stitch_pair(img_merged, images[order[k]])

cv.imshow('Planar Image Stitching', img_merged)
cv.waitKey(0)
cv.destroyAllWindows()
