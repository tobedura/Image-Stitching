import numpy as np
import cv2 as cv

img1 = cv.imread('assets/20260427_134715.jpg')
img2 = cv.imread('assets/20260427_134721.jpg')
img3 = cv.imread('assets/20260427_134726.jpg')
assert (img1 is not None) and (img2 is not None) and (img3 is not None), 'Cannot read the given images'

fdetector = cv.BRISK_create()
fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')


def crop_black(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    coords = cv.findNonZero(gray)
    if coords is None:
        return img
    x, y, w, h = cv.boundingRect(coords)
    return img[y:y + h, x:x + w]


def stitch_pair(left, right, debug_name):
    kpts_l, desc_l = fdetector.detectAndCompute(left, None)
    kpts_r, desc_r = fdetector.detectAndCompute(right, None)

    # Lowe's ratio test
    knn = fmatcher.knnMatch(desc_l, desc_r, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    print(f'[{debug_name}] keypoints: {len(kpts_l)} / {len(kpts_r)}, good matches: {len(good)} / {len(knn)}')

    if len(good) < 4:
        raise RuntimeError(f'[{debug_name}] not enough good matches')

    pts_l = np.float32([kpts_l[m.queryIdx].pt for m in good])
    pts_r = np.float32([kpts_r[m.trainIdx].pt for m in good])

    H, inlier_mask = cv.findHomography(pts_r, pts_l, cv.RANSAC, 5.0)
    inliers = int(inlier_mask.sum())
    print(f'[{debug_name}] RANSAC inliers: {inliers}/{len(good)} ({100*inliers/len(good):.1f}%)')

    vis = cv.drawMatches(left, kpts_l, right, kpts_r, good, None,
                         matchesMask=inlier_mask.ravel().tolist(),
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(f'debug_matches_{debug_name}.jpg', vis)

    # Compute output canvas that fits both warped right and the left image
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    corners_r = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(corners_r, H)
    all_corners = np.concatenate(
        (warped_corners, np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2))
    )
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    merged = cv.warpPerspective(right, translation @ H, (canvas_w, canvas_h))
    merged[-ymin:-ymin + h_l, -xmin:-xmin + w_l] = left
    return merged


print('--- Stitching img1 + img2 ---')
m12 = stitch_pair(img1, img2, '1_2')
m12 = crop_black(m12)
cv.imwrite('debug_stitched_12.jpg', m12)

print('--- Stitching (img1+img2) + img3 ---')
m123 = stitch_pair(m12, img3, '12_3')
m123 = crop_black(m123)
cv.imwrite('debug_stitched_123.jpg', m123)

cv.imshow('Stitched 1+2', cv.resize(m12, None, fx=0.4, fy=0.4))
cv.imshow('Stitched 1+2+3', cv.resize(m123, None, fx=0.4, fy=0.4))
cv.waitKey(0)
cv.destroyAllWindows()
