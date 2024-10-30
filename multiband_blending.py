import cv2
import numpy as np
def pyramid_blending(img1, img2, mask, levels=6):
    # Generate Gaussian pyramids for both images and the mask
    g_pyr_img1 = [img1]
    g_pyr_img2 = [img2]
    g_pyr_mask = [mask]

    for i in range(levels):
        img1 = cv2.pyrDown(g_pyr_img1[-1])
        img2 = cv2.pyrDown(g_pyr_img2[-1])
        mask = cv2.pyrDown(g_pyr_mask[-1])
        g_pyr_img1.append(img1)
        g_pyr_img2.append(img2)
        g_pyr_mask.append(mask)

    # Generate Laplacian pyramids for both images
    l_pyr_img1 = [g_pyr_img1[-1]]
    l_pyr_img2 = [g_pyr_img2[-1]]

    for i in range(levels - 1, 0, -1):
        size = (g_pyr_img1[i - 1].shape[1], g_pyr_img1[i - 1].shape[0])
        laplacian_img1 = cv2.subtract(g_pyr_img1[i - 1], cv2.resize(cv2.pyrUp(g_pyr_img1[i]), size))
        laplacian_img2 = cv2.subtract(g_pyr_img2[i - 1], cv2.resize(cv2.pyrUp(g_pyr_img2[i]), size))
        l_pyr_img1.append(laplacian_img1)
        l_pyr_img2.append(laplacian_img2)

    # Blend the two Laplacian pyramids with the Gaussian mask pyramid
    blended_pyr = []
    for l_img1, l_img2, g_mask in zip(l_pyr_img1, l_pyr_img2, g_pyr_mask[::-1]):
        size = (l_img1.shape[1], l_img1.shape[0])  # Ensure mask is same size as images
        g_mask_resized = cv2.resize(g_mask, size)
        blended = l_img1 * g_mask_resized + l_img2 * (1.0 - g_mask_resized)
        blended_pyr.append(blended)

    # Reconstruct the image from the blended pyramid
    blended_image = blended_pyr[0]
    for i in range(1, levels):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        blended_image = cv2.add(cv2.resize(cv2.pyrUp(blended_image), size), blended_pyr[i])

    return blended_image
# Load the two images
img1 = cv2.imread('IMG_0632.jpg')
img2 = cv2.imread('IMG_0633.jpg')

# Create a mask (0 for img1 and 1 for img2 in the blending region)
mask = np.zeros_like(img1[:, :, 0], dtype=np.float32)
mask[:, img1.shape[1]//2:] = 1

# Convert mask to three channels
mask = cv2.merge([mask, mask, mask])

# Perform blending
result = pyramid_blending(img1, img2, mask, levels=6)

# Save and show the result
cv2.imwrite('blended_image.jpg', result)
cv2.imshow('Blended Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()