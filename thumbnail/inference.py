import tensorflow as tf
import glob, os, os.path
import cv2
import numpy as np
import sys,argparse
import matplotlib.pyplot as plt
from PIL import Image
import time
import json
import spacy
import heapq
nlp = spacy.load('en')


def cnn_model_init():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./cnn/model/good-bad-2-model.meta')
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
    
    return sess, saver, graph, y_pred, x, y_true


def video_input(vid, temp1, cnn_count):
    
    sess, saver, graph, y_pred, x, y_true = cnn_model_init()
    

    vidcap = cv2.VideoCapture(vid)
    success,image = vidcap.read()
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    print fps
    print frame_count
    
    while success:
        success,image = vidcap.read()
        if count % fps == 0:
            cv2.imwrite(temp1 + "/frame%d.jpg" % count, image)     # save frame as JPEG file
            #print "Asd"
        if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            break
        count += 1
        
    
    cnn_list = []
    delete_list = []
    delete_jpg(temp1 + "./")
    
    for filename in os.listdir(temp1):
        if (filename.endswith(".jpg")):
            score, image_file = predict(temp1 + "/" + filename, sess, saver, graph, y_pred, x, y_true) 

            if len(cnn_list) == cnn_count:
                top = heapq.heappop(cnn_list)
                
                if top[0] < score[0][0]:
                    heapq.heappush(cnn_list, (score[0][0], image_file))
                    delete_list.append(top[1])
                else:
                    heapq.heappush(cnn_list, top)       
                    delete_list.append(image_file)
            else:
                heapq.heappush(cnn_list, (score[0][0], image_file))
     
    #print cnn_list
    sess.close()
    delete_temp(delete_list)
    return cnn_list
    
def delete_temp(delete_list):
    for img in delete_list:
        #print "rm -r " + img
        os.system("rm -r " + img)
    pass

def delete_jpg(mydir):
    filelist = glob.glob(os.path.join(mydir, "*.jpg"))
    for f in filelist:
        img = Image.open(f)
        img2 = img.rotate(90)
        img2.save(f)


def predict(image_path,sess, saver, graph, y_pred, x, y_true):
       
    # First, pass the path of the image
    cwd = os.getcwd()
    dir_path = cwd 
    filename = dir_path +'/' +image_path
    image_size=128
    num_channels=3
    images = []
    
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
    y_test_images = np.zeros((1, 2)) 

    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    return result, filename

    
def print_img(image_path):
    #S%pylab inline
    
    img=mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()
    
    
def densecap(image):
    command = "th run_model.lua -input_dir " + image 
    os.system(command)
    pass
 
def relevance_score(temp1, head, cap_count, dense_count):
    title = nlp(head)
    
    with open("./vis/data/results.json", "r") as f:
        datastore = json.load(f)
    
    cwd = os.getcwd()
    dir_path = cwd 
    
    
    rel_list = []
    for i in xrange (len(datastore[u'results'])):
        name = datastore[u'results'][i][u'img_name']
        name = dir_path +'/' + temp1 + "/" + name

        score = 0
        for j in datastore[u'results'][i][u'captions'][:cap_count]:
            cap = nlp(j)
            score += title.similarity(cap)       
        score /= (cap_count * 1.0)
        
        if len(rel_list) == dense_count:
            
            top = heapq.heappop(rel_list)

            if top[0] < score:
                heapq.heappush(rel_list, (score, name))
       
            else:
                heapq.heappush(rel_list, top)       
              
        else:
            heapq.heappush(rel_list, (score, name))

    return rel_list


def infer():
    start_time = time.time()
    cnn_count = 10
    cap_count = 5
    dense_count = 10
    alpha = 0.3
    beta = 0.7
    final_count = 5
    title = unicode(sys.argv[2], "utf-8")
    #title = u'Seahawks vs. Rams (Week 1) | Lockett vs. Austin Mini Replay | NFL FILMS'
    
    temp1 = "candidate_thumbnail"
    #vid = './cnn/video/sample4.mp4'
    vid = sys.argv[1] 
    #vid = './test_video/sample3.mp4'
    
    

    if not os.path.exists(temp1):
        os.makedirs(temp1)

    cnn = video_input(vid, temp1, cnn_count)
    #print cnn

    densecap(temp1)
    rel = relevance_score(temp1, title, cap_count, dense_count)

    cnn_d = {}
    for i in cnn:
        cnn_d[i[1]] = i[0]


    final = []

    for i,j in rel:
        if j in cnn_d:
            score = (alpha * cnn_d[j]) + (beta * i) 
            if len(final) == final_count:
                top = heapq.heappop(final)
                if top[0] < score:
                    heapq.heappush(final, (score, j))
                else:
                    heapq.heappush(final, top)       
            else:
                heapq.heappush(final, (score, j))


    final.sort(reverse=True)
    for i in final:
        print i

    print("--- %s seconds ---" % (time.time() - start_time))


    cwd = os.getcwd()
    dir_path = cwd 
    #print dir_path
    os.system("rm -r " + dir_path + "/vis/data/")
    
    os.makedirs(dir_path + "/vis/data/")


if __name__ == '__main__':
    infer()
