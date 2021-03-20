import tensorflow as tf
import cv2
import numpy as np

class GANDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, 
                 data, 
                 image_path,
                 batch_size=1, 
                 dim=(256, 256), 
                 n_channels=3,
                 shuffle=True
                ):
        '''
           Initialization
        '''
        self.data  = data
        self.indices = []
        self.image_path = image_path
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(batch)

        return X, y

    def on_epoch_end(self):
        '''
           Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            self.indices = self.indexes

    def _loadImage(self,image_path):
        'Load Target image'
        image_y = cv2.imread(image_path)
        image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB)
        image_y = cv2.resize(image_y,self.dim)
        img_gray = cv2.cvtColor(image_y, cv2.COLOR_RGB2GRAY)
        img_invert = cv2.bitwise_not(img_gray)
        img_smoothing = cv2.GaussianBlur(img_invert, (21, 21),sigmaX=0, sigmaY=0)
        image_x = cv2.divide(img_gray, 255 - img_smoothing, scale=self.dim[0])
        image_x = cv2.cvtColor(image_x,cv2.COLOR_GRAY2RGB)    
        # scale from [0,255] to [-1,1]
        image_x = (image_x  - 127.5) / 127.5
        image_y = (image_y  - 127.5) / 127.5
        return image_x, image_y
    
    def __data_generation(self,batch):
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))#,dtype=np.uint8)
        y = np.empty((self.batch_size, *self.dim, self.n_channels))#,dtype=np.uint8)
        
        for i, id_ in enumerate(batch):
            X[i,], y[i,] = self._loadImage(self.image_path + self.data[id_])
            
        return X, y

class GANDataGeneratorXY(tf.keras.utils.Sequence):
    
    def __init__(self, 
                 data_1, 
                 data_2,
                 image_path,
                 batch_size=1, 
                 dim=(256, 256), 
                 n_channels=3,
                 shuffle=True
                ):
        '''
           Initialization
        '''
        self.data_1  = data_1
        self.data_2  = data_2
        self.indices = []
        self.image_path = image_path
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data_1) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(batch)

        return X, y

    def on_epoch_end(self):
        '''
           Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.data_1))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            self.indices = self.indexes

    def _loadImage(self,image_path1,image_path2):
        
        'Load Target image'
        image_x = cv2.imread(image_path1)
        image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        image_x = cv2.resize(image_x,self.dim)

        image_y = cv2.imread(image_path2)
        image_y = cv2.cvtColor(image_y, cv2.COLOR_BGR2RGB)
        image_y = cv2.resize(image_y,self.dim)

        # scale from [0,255] to [-1,1]
        image_x = (image_x  - 127.5) / 127.5
        image_y = (image_y  - 127.5) / 127.5
        return image_x, image_y
    
    def __data_generation(self,batch):
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))#,dtype=np.uint8)
        y = np.empty((self.batch_size, *self.dim, self.n_channels))#,dtype=np.uint8)
        
        for i, id_ in enumerate(batch):
            X[i,], y[i,] = self._loadImage(self.image_path + self.data_1[id_],
                                           self.image_path + self.data_2[id_])
            
        return X, y