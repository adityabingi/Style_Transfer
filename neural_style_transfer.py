import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

from tensorflow.keras.applications import vgg19
import tensorflow.contrib.eager as tfe

# Enable to run in eager mode
tf.enable_eager_execution()

class Config:

    """
        Initializing config variables required for the Style Transfer
    """
    
    # VGG Preprocessing mean for BGR channels
    VGG_MEAN = np.array([103.939, 116.779, 123.68])
    
    # Input Image Shape for VGG Network
    IMAGE_SHAPE = (512, 512, 3)
    
    # Weighting factors for content and style
    CONTENT_LOSS_WEIGHT = 1e3
    STYLE_LOSS_WEIGHT   = 1
    
    # Layers used for content extraction
    CONTENT_LAYERS = ['block4_conv2']
    
    # Layers used for style extraction
    STYLE_LAYERS = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    
    # Results stored here
    OUTPUT_DIR = "results/"



def preprocess(img):

    """ preprocessing for vgg-19 network"""
    
    img = img.resize((Config.IMAGE_SHAPE[1], Config.IMAGE_SHAPE[0]),Image.ANTIALIAS)
    
    img = np.array(img, dtype=np.float32)

    img = img[:,:,::-1]
    
    img[:,:,0] -= Config.VGG_MEAN[0]
    img[:,:,1] -= Config.VGG_MEAN[1]
    img[:,:,2] -= Config.VGG_MEAN[2]
    
    img = np.expand_dims(img, axis=0)
    
    return img



def load_and_process(img_path):
    
    img = Image.open(img_path)
    
    img_tensor = preprocess(img)
    
    return img_tensor



def deprocess(processed_img):

    """ Deprocessing required for the style transfer output image"""
    
    img = processed_img.copy()
    
    if(len(img.shape)==4):  
        img = np.squeeze(img, axis=0)
        
    if len(img.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    img[:,:,0] += Config.VGG_MEAN[0]
    img[:,:,1] += Config.VGG_MEAN[1]
    img[:,:,2] += Config.VGG_MEAN[2]
    
    img = img[:,:,::-1]
    
    img = np.clip(img, 0, 255).astype('uint8')
    
    return img 


def save_img(img_arr, save_path):
    
    result = Image.fromarray(img_arr)
    
    result.save(save_path)



def get_model(layer_names):

    """Define model using pretrained vgg19"""
    
    vgg = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=Config.IMAGE_SHAPE)
    
    vgg.trainable = False 
    
    outputs = [vgg.get_layer(name).output for name in layer_names]
    
    model = tf.keras.Model(vgg.input, outputs)
    
    return model



def content_loss(target_content, pastiche_content):

	""" Content loss from Gatys et al. https://arxiv.org/pdf/1508.06576.pdf"""
    
    return tf.reduce_mean(tf.square(target_content-pastiche_content))

    
def gram_matrix(input_tensor):
    
    channels = int(input_tensor.shape[-1])
    
    input_tensor = tf.squeeze(input_tensor, axis=0)
    
    input_tensor = tf.transpose(input_tensor, [2,0,1])
    
    input_tensor = tf.reshape(input_tensor, [channels, -1])
    
    n = input_tensor.shape[-1]
    
    gram_mat = tf.matmul(input_tensor, input_tensor, transpose_b = True)
    
    return gram_mat/tf.cast(n, dtype=tf.float32)



def style_loss(gram_target, gram_pastiche):

	""" Gram Style loss from Gatys et al. https://arxiv.org/pdf/1508.06576.pdf"""
    
    return tf.reduce_mean(tf.square(gram_target-gram_pastiche))


class StyleTransfer:

    def __init__(self, content_path, style_path):
    
        self.style_layers = Config.STYLE_LAYERS

        self.content_layers = Config.CONTENT_LAYERS

        self.num_style_layers = len(self.style_layers)

        self.num_content_layers = len(self.content_layers)

        self.model = get_model(self.style_layers + self.content_layers)

        self.model.trainable = False

        self.min_vals = -Config.VGG_MEAN

        self.max_vals = 255 - Config.VGG_MEAN

        self.content_image = load_and_process(content_path)

        self.style_image =  load_and_process(style_path) 

        # pastiche: initial noise image to which style is transfered
        # initialize noise image with content image for fast optimization
          
        self.pastiche = tfe.Variable(self.content_image, dtype=tf.float32) 
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate= 5, beta1 =0.99, 
                                                  epsilon=1e-1)
            
    def compute_target_features(self):

        self.target_style = (self.model(self.style_image))[ :self.num_style_layers]

        self.target_content = (self.model(self.content_image))[self.num_style_layers: ]

        self.gram_target_style = [gram_matrix(style_layer) for style_layer in self.target_style]
            
        
    def compute_total_loss(self):


        pastiche_outputs = self.model(self.pastiche)

        pastiche_style, pastiche_content = (pastiche_outputs[ :self.num_style_layers],
                                                      pastiche_outputs[self.num_style_layers:])

        gram_pastiche_style = [gram_matrix(style_layer) for style_layer in pastiche_style]

        total_style_loss = 0

        total_content_loss = 0

        for i in range(self.num_style_layers):

            total_style_loss += (style_loss(self.gram_target_style[i], 
                                                   gram_pastiche_style[i]))*(1/self.num_style_layers)

        for i in range(self.num_content_layers):

            total_content_loss += (content_loss(self.target_content[i], 
                                                       pastiche_content[i]))*(1/self.num_content_layers)

        total_loss =(Config.STYLE_LOSS_WEIGHT * total_style_loss)+ (Config.CONTENT_LOSS_WEIGHT
                                                                             *total_content_loss)


        return total_loss, total_style_loss, total_content_loss

        
    def compute_grads(self):

        with tf.GradientTape() as tape:

            loss = self.compute_total_loss()
          
        total_loss = loss[0]

        return tape.gradient(total_loss, self.pastiche), loss
      
          
    def optimize_pastiche(self, num_epochs):

    	""" Image Optimization loop of pastiche (initial noise image) to which style and content are transfered"""

        self.compute_target_features()
        
        best_loss, best_img = float('inf'), None

        for i in range(num_epochs):
          
            grads, all_loss = self.compute_grads()

            total_loss, style_loss, content_loss = all_loss

            self.optimizer.apply_gradients([(grads, self.pastiche)])

            clipped = tf.clip_by_value(self.pastiche, self.min_vals, self.max_vals)

            self.pastiche.assign(clipped)
                
            if(total_loss<best_loss):

                best_loss = total_loss

                best_img = deprocess(self.pastiche.numpy())
                     
            if( i% 10 ==0):
                print("After {} epochs total_loss: {}   style_loss: {}   content_loss: {} ".format(
                                  i, total_loss, style_loss, content_loss))
                    
            if i%100 == 0:

                track_img = deprocess(self.pastiche.numpy())

                save_img(track_img, Config.OUTPUT_DIR + ('iteration_{}.jpg'.format(i)))
                                           
        save_img(best_img, Config.OUTPUT_DIR + 'final_best_img.jpg')
        
                         
        return best_loss


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path", nargs=1)
    parser.add_argument("--style_path", nargs=1)

    args = parser.parse_args()
    
    style_transfer = StyleTransfer(args.content_path[0], args.style_path[0])
    
    best_loss = style_transfer.optimize_pastiche(1000)

    print("Final total loss after 1000 iteration is: {}".format(best_loss))


if __name__=='__main__':
    main()

