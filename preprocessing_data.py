from global_head_file import *
def getData(work_dir, classification_num, start, batch_size, shuffle = False):
    trainX, trainY = [], []
    dirs = os.listdir(work_dir)
    for dir in dirs:
        index = int(dir) #used to define the one hot vector
        dir_full_path = work_dir + dir
        imageNames = (os.listdir(dir_full_path))
        for j in range(start, start + batch_size):
            imageName = imageNames[j]
            imageName = os.path.join(dir_full_path, imageName)
            img = cv2.imread(imageName)
            #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #imgGrayFilter = cv2.GaussianBlur(imgGray, (3,3), 0)
            img_resize = cv2.resize(img,(image_size, image_size))
            #cv2.imshow('resized_image', img_resize)
            #cv2.waitKey(0)
            trainX.append(img_resize)
            placeholder_temp = np.copy(classification_output_placeholder)
            placeholder_temp[index] = 1
            trainY.append(placeholder_temp)
    return trainX, trainY

def image_distortion(img): #the function is not yet implemented
    distored_image = np.copy(img)
    return distored_image

