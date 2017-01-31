from global_head_file import *
def getData(work_dir, classification_num, start, batch_size, shuffle = False, data_augmentation = True): #data shuffling is not yet implemented
    trainX, trainY = [], []
    dirs = os.listdir(work_dir)
    for dir in dirs:
        index = int(dir) #used to define the one hot vector
        dir_full_path = work_dir + dir
        imageNames = (os.listdir(dir_full_path))
        for j in range(start, start + batch_size):
            file_num = int(random.uniform(0, len(imageNames) - 1))#this would pick up image randomly to ensure most data are used intead of using fixed part of the samples.
            imageName = imageNames[file_num]
            imageName = os.path.join(dir_full_path, imageName)
            img = cv2.imread(imageName)
            img_resize = cv2.resize(img, (image_size, image_size))
            if color_channels == 1:
                img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                img_resize = np.expand_dims(img_resize, 2)
            #cv2.imshow('resized_image', img_resize)
            #cv2.waitKey(0)
            if data_augmentation == True: #if data augmentation is on then augment the data using random transformation. otherwise use original image
                for iii in range(data_augmentation_factor):
                    img_distorted_resize = image_distortion(img_resize)
                    img_distorted_resize = np.expand_dims(img_distorted_resize, 2)
                    trainX.append(img_distorted_resize)
                    placeholder_temp = np.copy(classification_output_placeholder)
                    placeholder_temp[index] = 1
                    trainY.append(placeholder_temp)
            else:
                trainX.append(img_resize)
                placeholder_temp[index] = 1
                trainY.append(placeholder_temp)
    return trainX, trainY

def image_distortion(img): 
    distored_image = translate_img(img)
    distored_image = adj_contrast_brightness_noise(distored_image)
    return distored_image

def translate_img(img):
    offsetx = (random.uniform(-image_size/10, image_size/10))
    offsety = (random.uniform(-image_size/10, image_size/10))
    trans_mat = np.array([[1., 0., offsetx],[0., 1., offsety]])
    translated_image = cv2.warpAffine(img, trans_mat, (image_size, image_size))
    return translated_image

def adj_contrast_brightness_noise(img):
    #original_img = np.copy(img)
    contrast_ratio = random.gauss(1, 0.3)
    brightness_adjust = random.uniform(-50, 50)
    noise_mask = np.zeros_like(img)
    noise = cv2.randn(noise_mask, - 20, 20)
    img = img.astype(float)
    img *= contrast_ratio
    img += brightness_adjust
    img += noise
    img = np.clip(img, 0, 255) #make sure the range is in [0, 255]
    img = img.astype(np.uint8)
    #cv2.imshow('dist',img)
    #cv2.waitKey(0)
    #cv2.imshow('dist',original_img)
    #cv2.waitKey(0)
    return img

