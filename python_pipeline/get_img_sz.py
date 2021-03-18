import my_conv2

def get_img_sz(img, filters):
    img_size = {}
    for i in range(1, filters.size):
        img_size[i] = my_conv2(img, filters[i]).shape
    
    return img_size