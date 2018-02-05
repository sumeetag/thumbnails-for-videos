import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes, prev, nex):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:   
        count = 0
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            try:
                if count < prev:
                    count += 1
                    continue
                if count > nex:
                    break
                count += 1
                #print (fl)
                image = cv2.imread(fl)
                image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image = np.multiply(image, 1.0 / 255.0)
                images.append(image)
                label = np.zeros(len(classes))
                label[index] = 1.0
                labels.append(label)
                flbase = os.path.basename(fl)
                img_names.append(flbase)
                cls.append(fields)
            except:
                pass
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    
    return images, labels, img_names, cls


class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0
    print self._images.shape

  def update(self, images, labels, img_names, cls):
    self._num_examples += images.shape[0]
    img = np.concatenate((self._images, images))
    lab = self._labels = np.concatenate((self._labels, labels))
    img_name = self._img_names = np.concatenate((self._img_names, img_names))
    cl = self._cls = np.concatenate((self._cls, cls))
    self._images = img
    self._labels = lab
    self._img_names = img_name
    self._cls = cl
    
  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()
    
  flag = 1
  prev = -1
  nex = 5000
  buffer = 5000
  while True:
      
      images = None
      images, labels, img_names, cls = load_train(train_path, image_size, classes, prev, nex)
      images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  
        
      if images.shape[0] == 0:
        break

      if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

      validation_images = images[:validation_size]
      validation_labels = labels[:validation_size]
      validation_img_names = img_names[:validation_size]
      validation_cls = cls[:validation_size]

      train_images = images[validation_size:]
      train_labels = labels[validation_size:]
      train_img_names = img_names[validation_size:]
      train_cls = cls[validation_size:]

      if flag:
          data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
          data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)
          flag = 0
      
      else:
          data_sets.train.update(train_images, train_labels, train_img_names, train_cls)
          data_sets.valid.update(validation_images, validation_labels, validation_img_names, validation_cls)
      prev += buffer
      nex += buffer
        
      print data_sets.train.num_examples
    
  
  return data_sets


