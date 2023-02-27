import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import skimage.transform
import cv2

def generate_cloud(im_size_h, im_size_w,k = 2,):
    # Initialize the white noise pattern
    base_pattern = np.random.uniform(0,255, (im_size_h//2, im_size_w//2))

    # Initialize the output pattern
    turbulence_pattern = np.zeros((im_size_h, im_size_w))

    # Create cloud pattern
    power_range = [k**i for i in range(2, int(np.log2(min(im_size_h, im_size_w))))]
    
    for p in power_range:
        quadrant = base_pattern[:p, :p]
        upsampled_pattern = skimage.transform.resize(quadrant, (im_size_h, im_size_w), mode='reflect')
        turbulence_pattern += upsampled_pattern / float(p)

    turbulence_pattern /= sum([1 / float(p) for p in power_range])    
    return turbulence_pattern


def add_cloud(file_name,k):
    #file_name = '20_00019.png'
#     img = mpimg.imread(file_name) * 255
    img = cv2.imread(file_name)
    im_size_h, im_size_w = np.shape(img)[:2]
        
    # Generate cloud map
    cloud_map = generate_cloud(im_size_h, im_size_w,k)
    fourground_map = (255 - cloud_map) / 255
    #plt.imsave('cloud.png',cloud_map)
    # add cloud to original image
    res = np.zeros((np.shape(img)))
    print( img[:,:,0])
    res[:,:,0] = img[:,:,0] * fourground_map + cloud_map
    res[:,:,1] = img[:,:,1] * fourground_map + cloud_map
    res[:,:,2] = img[:,:,2] * fourground_map + cloud_map
    
    #print(np.max(res))
    #print(np.min(res))
    #plt.imsave(file_name.replace('.tif', '_cloud.png'), res)
    
    return cloud_map, res.astype(np.uint8),fourground_map


generate_cloud(256, 256,2)
