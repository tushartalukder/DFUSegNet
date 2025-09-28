

import tensorflow as tf
from tensorflow.keras import layers, backend as K
from keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer
import tf_clahe

# --- Basic and Residual Convolutional Blocks ---
def conv2d(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu',dilation_rate=1, name=None):
    init = RandomNormal(stddev=0.02)
    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding,dilation_rate=1, kernel_initializer=init)(x)
    return x


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu',dilation_rate=1, name=None):
    init = RandomNormal(stddev=0.02)
    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding,dilation_rate=1, kernel_initializer=init)(x)
    x = layers.BatchNormalization(axis=3)(x, training=True)
    if activation:
        x = layers.Activation(activation, name=name)(x)
    return x

def MultiResBlock(U, inp, alpha=1.67):
    W = alpha * U
    shortcut = inp
    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) + int(W*0.5), 1, 1, activation=None, padding='same')
    
    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3, activation='relu', padding='same')
    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3, activation='relu', padding='same')
    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3, activation='relu', padding='same')

    out = layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = layers.BatchNormalization(axis=3)(out, training=True)
    
    out = layers.add([shortcut, out])
    out = layers.Activation('relu')(out)
    out = layers.BatchNormalization(axis=3)(out, training=True)
    return out

# --- Attention and Gating Mechanisms ---

def gating_signal(input, out_size, batch_norm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x, training=True)
    x = layers.ReLU()(x)
    return x

def repeat_elem(tensor, rep):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
    
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = K.int_shape(theta_x)
    
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3), 
                                        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), 
                                        padding='same')(phi_g)
    
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.ReLU()(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi1 = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)
    
    upsample_psi = repeat_elem(upsample_psi1, shape_x[3])
    y = layers.multiply([upsample_psi, x])
    
    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result, training=True)
    return result_bn, upsample_psi1

class PAM_Module(tf.keras.layers.Layer):
    """Position Attention Module (PAM)"""
    def __init__(self, in_dim,name='b', **kwargs):
        super(PAM_Module, self).__init__(**kwargs)
        self.chanel_in = in_dim

        # Define the query, key, and value convolutions with layer names
        self.query_conv = layers.Conv2D(filters=in_dim//4, kernel_size=1, padding='same', name=name+'query_conv')
        self.query_conv1 = layers.Conv2D(filters=in_dim//4, kernel_size=3, padding='same', name=name+'query_conv1')
        self.query_conv2 = layers.Conv2D(filters=in_dim//4, kernel_size=3, padding='same', dilation_rate=(3, 3),name=name+'query_conv2')
        self.query_conv3 = layers.Conv2D(filters=in_dim//4, kernel_size=3, padding='same', dilation_rate=(6, 6),name=name+'query_conv3')
        self.query_conv4 = layers.Conv2D(filters=in_dim, kernel_size=1, padding='same', name=name+'query_conv4')
        self.key_conv = layers.Conv2D(filters=in_dim, kernel_size=3, padding='same', name=name+'key_conv')
        
        self.key_conv1 = layers.Conv2D(filters=in_dim, kernel_size=3, padding='same', name=name+'key_conv1')
        self.value_conv = layers.Conv2D(filters=in_dim, kernel_size=1, name=name+'value_conv')
        
        # Gamma is a trainable parameter
        self.gamma = self.add_weight(name=name+'gamma', shape=(1,), initializer='zeros', trainable=True)

        # Softmax for attention calculation
        self.softmax = layers.Softmax(axis=-1, name=name+'attention_softmax')
    
    def call(self, x):
        """
        Parameters:
        ----------
            x : input feature maps (batch_size, height, width, channels)
        
        Returns:
        ----------
            out : attention value + input feature
            attention: (batch_size, height*width, height*width)
        """
        m_batchsize = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        C = tf.shape(x)[3]
        
        # Compute query, key, and value
        proj_query1 = self.query_conv(x)
        proj_query2 = self.query_conv1(x)
        proj_query3 = self.query_conv2(x)
        proj_query4 = self.query_conv3(x)
        proj_query = tf.concat([proj_query1,proj_query2,proj_query3,proj_query4],axis=-1)
        proj_query = self.query_conv4(proj_query)
        proj_key = self.key_conv(x)
        proj_key = self.key_conv(proj_key)
        proj_value = self.value_conv(x)

        # Reshape and transpose for attention calculation
        proj_query = tf.transpose(proj_query, perm=[0, 2, 1, 3])  # (batch_size, height*width, channels)
        energy = tf.einsum('bijc,bkjc->bikc', proj_key, proj_query)
        
        # Calculate energy and attention map
        attention = self.softmax(energy)  # (batch_size, height*width, height*width)
        
        # Compute the output with attention applied to value
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        out = tf.einsum('bijc,bkjc->bikc', attention, proj_value)  # (batch_size, channels, height*width)
        
        # Apply the gamma scaling and add the input feature map
        out = self.gamma * out + x
        return out

    def get_config(self):
        config = super(PAM_Module, self).get_config()
        config.update({
            'in_dim': self.chanel_in,
        })
        return config


# --- Specialized Blocks (RFB, Tokenized MLP) ---

def BasicConv2D(inputs, out_planes, kernel_size, stride=1, padding='same', dilation=1):
    conv = layers.Conv2D(filters=out_planes, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, use_bias=False)(inputs)
    bn = layers.BatchNormalization()(conv, training=True)
    relu = layers.ReLU()(bn)
    return relu

def RFBModified(inputs, out_channel):
    relu = layers.ReLU()(inputs)
    branch0 = BasicConv2D(relu, out_channel, kernel_size=1)
    
    conv1_3x3 = BasicConv2D(BasicConv2D(BasicConv2D(relu, out_channel, 1), out_channel, (1,3)), out_channel, (3,1))
    conv1_3x3_dil = BasicConv2D(conv1_3x3, out_channel, 3, dilation=3)
    
    conv2_3x3 = BasicConv2D(BasicConv2D(BasicConv2D(relu, out_channel, 1), out_channel, (1,5)), out_channel, (5,1))
    conv2_3x3_dil = BasicConv2D(conv2_3x3, out_channel, 3, dilation=5)

    conv3_3x3 = BasicConv2D(BasicConv2D(BasicConv2D(relu, out_channel, 1), out_channel, (1,7)), out_channel, (7,1))
    conv3_3x3_dil = BasicConv2D(conv3_3x3, out_channel, 3, dilation=7)

    branches_concat = layers.Concatenate(axis=-1)([branch0, conv1_3x3_dil, conv2_3x3_dil, conv3_3x3_dil])
    conv_cat = BasicConv2D(branches_concat, out_channel, 3)
    conv_res = BasicConv2D(relu, out_channel, 1)
    
    return layers.ReLU()(conv_cat + conv_res)


def aggregation(x1, x2, x3, x4, x5):
    channel=32
    upsample = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
    x1_1 = x1
    x2_1 = BasicConv2D(upsample(x1),channel, 3, padding='same')*x2
    x3_1 = BasicConv2D(upsample(upsample(x1)),channel, 3, padding='same')*BasicConv2D(upsample(x2),channel, 3, padding='same')*x3
    x4_1 = BasicConv2D(upsample(upsample(upsample(x1))),channel, 3, padding='same')*BasicConv2D(upsample(upsample(x2)),channel, 3, padding='same')* \
            BasicConv2D(upsample(x3),channel, 3, padding='same') * x4
    x5_1 = BasicConv2D(upsample(upsample(upsample(upsample(x1)))),channel, 3, padding='same')*BasicConv2D(upsample(upsample(upsample(x2))),channel, 3, padding='same')* \
            BasicConv2D(upsample(upsample(x3)),channel, 3, padding='same') * BasicConv2D(upsample(x4),channel, 3, padding='same')*x5
    
    
    x2_2 = tf.concat([x2_1, BasicConv2D(upsample(x1_1),channel, 3, padding='same')], axis=-1)
    x2_2 = BasicConv2D(x2_2,2*channel, 3, padding='same')
    
    x3_2 = tf.concat([x3_1, BasicConv2D(upsample(x2_2),2*channel, 3, padding='same')], axis=-1)
    x3_2 = BasicConv2D(x3_2,3*channel, 3, padding='same')    

    x4_2 = tf.concat([x4_1, BasicConv2D(upsample(x3_2),3*channel, 3, padding='same')], axis=-1)
    x4_2 = BasicConv2D(x4_2,4*channel, 3, padding='same')  

    x5_2 = tf.concat([x5_1, BasicConv2D(upsample(x4_2),4*channel, 3, padding='same')], axis=-1)
    x5_2 = BasicConv2D(x5_2,5*channel, 3, padding='same') 

    x = BasicConv2D(x5_2,5 * channel, 3, padding='same')
    x = tf.keras.layers.Conv2D(1, 1)(x)
    
    return x
    
class TokenizedMLPBlock(layers.Layer):
    def __init__(self, embedding_dim, mlp_hidden_dim, **kwargs):
        super(TokenizedMLPBlock, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.tokenize = layers.Conv2D(embedding_dim, kernel_size=3, padding='same')
        self.mlp1 = layers.Dense(mlp_hidden_dim)
        self.dwconv1 = layers.DepthwiseConv2D(kernel_size=3, padding='same')
        self.activation = layers.Activation('gelu')
        self.mlp2 = layers.Dense(embedding_dim) # Corrected to project back to embedding_dim
        self.ln = layers.LayerNormalization()
        self.reproject = layers.Dense(embedding_dim)

    def call(self, inputs):
        tokens_w = self.tokenize(inputs)
        y = tf.roll(tokens_w, shift=1, axis=2)
        y = self.mlp1(y)
        y = self.dwconv1(y)
        y = self.activation(y)
        y = tf.roll(y, shift=1, axis=1)
        y = self.mlp2(y) # Corrected this part
        y = self.activation(y)
        y = self.ln(y)
        y = self.reproject(y)
        return layers.Add()([tokens_w, y])

    def get_config(self):
        config = super(TokenizedMLPBlock, self).get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
        })
        return config


def rgb_to_hsv(rgb_image):
    # Ensure the input is in float32 and scaled between 0 and 1
#     rgb_image = tf.cast(rgb_image, dtype=tf.float32) / 255.0

    r, g, b = tf.split(rgb_image, num_or_size_splits=3, axis=-1)
    
    # Calculate max, min, and difference (chroma)
    max_val = tf.reduce_max(rgb_image, axis=-1, keepdims=True)
    min_val = tf.reduce_min(rgb_image, axis=-1, keepdims=True)
    chroma = max_val - min_val

    # Compute Value (V)
    v = max_val

    # Compute Saturation (S)
    s = tf.where(max_val == 0, tf.zeros_like(max_val,dtype =tf.float32), chroma / max_val)

    # Compute Hue (H)
    zero_tensor = tf.zeros_like(r,dtype =tf.float32)
    
    h = tf.where(tf.equal(chroma, 0), zero_tensor, tf.where(
        tf.equal(max_val, r), (g - b) / chroma % 6,
        tf.where(tf.equal(max_val, g), (b - r) / chroma + 2, (r - g) / chroma + 4)
    ))

    h = h / 6.0  # Normalize hue to range [0, 1]
    
    # Stack the H, S, V components back together
    hsv_image = tf.concat([h, s, v], axis=-1)
#     hsv_image = (hsv_image-tf.reduce_min(hsv_image))/(tf.reduce_max(hsv_image)-tf.reduce_min(hsv_image))
    return hsv_image

def hsv_to_rgb(hsv_image):
    # Ensure the input is in float32 and scaled between 0 and 1
    h, s, v = tf.split(hsv_image, num_or_size_splits=3, axis=-1)

    c = v * s
    x = c * (1 - tf.abs((h * 6) % 2 - 1))
    m = v - c

    # Calculate the RGB prime values depending on the hue sector
    zeros = tf.zeros_like(h,dtype =tf.float32)
    
    r1 = tf.where((h < 1/6), c, tf.where((h < 2/6), x, tf.where((h < 3/6), zeros, tf.where((h < 4/6), zeros, tf.where((h < 5/6), x, c)))))
    g1 = tf.where((h < 1/6), x, tf.where((h < 2/6), c, tf.where((h < 3/6), c, tf.where((h < 4/6), x, tf.where((h < 5/6), zeros, zeros)))))
    b1 = tf.where((h < 1/6), zeros, tf.where((h < 2/6), zeros, tf.where((h < 3/6), x, tf.where((h < 4/6), c, tf.where((h < 5/6), c, x)))))
    
    # Add m to get the final RGB values
    r = (r1 + m)
    g = (g1 + m)
    b = (b1 + m)
    
    # Stack the R, G, B components back together
    rgb_image = tf.concat([r, g, b], axis=-1)

    return rgb_image



class CombinedImageProcessingLayer(Layer):
    def __init__(self, **kwargs):
        super(CombinedImageProcessingLayer, self).__init__(**kwargs)
        
        # Learnable parameters for HSV adjustments
        self.hue_delta = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="hue_delta")
        self.saturation_factor = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="saturation_factor")
        self.value_factor = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="value_factor")
        self.edge_enhance_factor = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="edge_enhance_factor")
        

        # Learnable parameters for Gaussian sharpening
        self.sharpen_factor = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="sharpen_factor")
        self.blur_kernel_size = 5
        self.blur_sigma = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="blur_sigma")

#         self.blur_sigma = tf.Variable(1.0, trainable=True, dtype=tf.float32, name="blur_sigma")

    def gaussian_blur(self, image):
        def gaussian_kernel(size: int, sigma: float):
            size = int(size)
            x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
            x = tf.square(x)
            kernel = tf.exp(-x / (2.0 * sigma**2))
            kernel = kernel / tf.reduce_sum(kernel)
            kernel = tf.reshape(kernel, (size, 1))
            return kernel

        gaussian_filter_x = gaussian_kernel(self.blur_kernel_size, self.blur_sigma)
        gaussian_filter_y = tf.transpose(gaussian_filter_x)

        gaussian_filter_x = tf.expand_dims(gaussian_filter_x, axis=-1)
        gaussian_filter_x = tf.expand_dims(gaussian_filter_x, axis=-1)
        gaussian_filter_x = tf.tile(gaussian_filter_x, [1, 1, 1, 3])
        
        gaussian_filter_y = tf.expand_dims(gaussian_filter_y, axis=-1)
        gaussian_filter_y = tf.expand_dims(gaussian_filter_y, axis=-1)
        gaussian_filter_y = tf.tile(gaussian_filter_y, [1, 1, 1, 3])

        blurred_image_x = tf.nn.conv2d(image, gaussian_filter_x, strides=[1, 1, 1, 1], padding="SAME")
        blurred_image = tf.nn.conv2d(blurred_image_x, gaussian_filter_y, strides=[1, 1, 1, 1], padding="SAME")

        return blurred_image

    def call(self, inputs):

        hsv_image = rgb_to_hsv(inputs)
        H = hsv_image[..., 0]
        S = hsv_image[..., 1]
        V = hsv_image[..., 2]

        H = tf.clip_by_value(H * self.hue_delta, 0.0, 1.0)
        S = tf.clip_by_value(S * self.saturation_factor, 0.0, 1.0)
        V = tf.clip_by_value(V * self.value_factor, 0.0, 1.0)
        
        adjusted_hsv = tf.stack([H, S, V], axis=-1)
        hsv_adjusted_image = hsv_to_rgb(adjusted_hsv)

        # --- Edge Enhancement ---
        sobel_x = tf.image.sobel_edges(inputs)[..., 0]
        sobel_y = tf.image.sobel_edges(inputs)[..., 1]
        edge_magnitude = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))
        enhanced_edges = edge_magnitude * self.edge_enhance_factor
        enhanced_edges_rgb = tf.tile(enhanced_edges, [1, 1, 1, 1])
        enhanced_edges_rgb = tf.clip_by_value(enhanced_edges_rgb+inputs, 0.0, 1.0)

        blurred_image = self.gaussian_blur(inputs)
        sharpened_image = inputs + self.sharpen_factor * (inputs - blurred_image)
        sharpened_image = tf.clip_by_value(sharpened_image, 0.0, 1.0)
        clahe_image = tf_clahe.clahe(inputs)
        output_image = tf.concat([hsv_adjusted_image, enhanced_edges_rgb, sharpened_image,clahe_image], axis=-1)

        return output_image

    def get_config(self):
        config = super(CombinedImageProcessingLayer, self).get_config()
        return config
