import numpy as np
import cv2
from scipy.ndimage import convolve

def compute_gradient(img):
    # Compute the gradient of the image using Sobel operator
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y

def compute_divergence(grad_x, grad_y):
    # Compute the divergence of the gradient
    div_x = np.roll(grad_x, -1, axis=1) - grad_x
    div_y = np.roll(grad_y, -1, axis=0) - grad_y
    return div_x + div_y

def llt_denoise(img, alpha, lambda_param, max_iter=100, tol=1e-5):
    h, w = img.shape
    u = img.astype(np.float64)
    u_old = u.copy()
    
    for i in range(max_iter):
        # Compute the gradient of the current estimate
        grad_x, grad_y = compute_gradient(u)
        
        # Compute the divergence of the gradient
        div = compute_divergence(grad_x, grad_y)
        
        # Update the estimate using the proximal operator
        u = (u_old + lambda_param * (img - u)) / (1 + lambda_param)
        
        # Apply the proximal operator for the divergence term
        u += alpha * div
        
        # Check for convergence
        if np.linalg.norm(u - u_old) < tol:
            break
        
        u_old = u.copy()
    
    return u

if __name__ == "__main__":
    # Read the noisy image
    img = cv2.imread('Images\lena.png', cv2.IMREAD_GRAYSCALE)
    
    # Set parameters
    alpha = 0.1
    lambda_param = 0.01
    
    # Denoise the image using LLT model
    denoised_img = llt_denoise(img, alpha, lambda_param)
    
    # Save or display the denoised image
    cv2.imwrite('denoised_image.png', denoised_img)
    cv2.imshow('Denoised Image', denoised_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
