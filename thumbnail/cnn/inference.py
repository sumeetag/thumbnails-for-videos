def video_input(vid):
    import glob, os, os.path
    import cv2
    delete_jpg('./cnn/video/')
    
    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    count = 0
    
    while success:
        success,image = vidcap.read()
        if count % 30 == 0:
            cv2.imwrite("./cnn/video/frame%d.jpg" % count, image)     # save frame as JPEG file
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1
    print (count)
    for filename in os.listdir("./cnn/video/"):
        if (filename.endswith(".jpg")):
            print("--- %s seconds ---" % (time.time() - start_time))
            predict("./cnn/video/" + filename)
            #d_cap.relevance_score("./cnn/video/" + filename, u'elephant ride')
            
def delete_jpg(mydir):
    import glob, os, os.path
    filelist = glob.glob(os.path.join(mydir, "*.jpg"))
    for f in filelist:
        os.remove(f)
        
def predict(image_path):
    import tensorflow as tf
    import numpy as np
    import os,glob,cv2
    import sys,argparse
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from PIL import Image
        
    

    # First, pass the path of the image
    cwd = os.getcwd()
    dir_path = cwd #os.path.dirname(os.path.realpath(__file__))
    filename = dir_path +'/' +image_path
    image_size=256
    num_channels=3
    images = []
    # Reading the image using OpenCV
    #img = Image.open(filename)

    #img2 = img.rotate(180)
    #img2.save(filename)
    image = cv2.imread(filename)
    
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ## Let us restore the saved model 
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./cnn/model/good-bad-1-model.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./cnn/model/'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 2)) 


    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    print(result)
    #print_img(image_path)
    
def print_img(image_path):
    #S%pylab inline
    import tensorflow as tf
    import numpy as np
    import os,glob,cv2
    import sys,argparse
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img=mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()
    
   
import time
import d_cap
start_time = time.time()
vid = './cnn/video/sample1.mp4'
video_input(vid)
print("--- %s seconds ---" % (time.time() - start_time))