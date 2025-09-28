import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Concatenate, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, DenseNet201
from src.model_parts import MultiResBlock, PAM_Module, TokenizedMLPBlock, RFBModified, aggregation, BasicConv2D, gating_signal, attention_block
from src.utils import hed, boundary_enhancer
from src.losses import structure_loss, focal_loss, bce_dice_loss
from src.model_parts import conv2d_bn, conv2d, repeat_elem
# Assuming CombinedImageProcessingLayer is in model_parts
from src.model_parts import CombinedImageProcessingLayer 

def define_generator(height, width, n_channels,name_prefix='b_'):
    inputs = Input((height, width, n_channels))
    encoder = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
    encoder1 = DenseNet201(include_top=False, weights="imagenet", input_tensor=inputs)
    m1_inputs = CombinedImageProcessingLayer()(inputs)
    m2_inputs = tf.image.resize(m1_inputs, (256,256))
    m3_inputs = tf.image.resize(m1_inputs, (128,128))
    m4_inputs  = tf.image.resize(m1_inputs, (64,64))
    m5_inputs  = tf.image.resize(m1_inputs, (32,32))

    ## encoder stage 1
    m1_512 = MultiResBlock(32, m1_inputs)
    m1_512 = PAM_Module(in_dim=m1_512.shape[-1],name='512')(m1_512)
    m1_512_p = MaxPooling2D(pool_size=(2, 2))(m1_512)
    # m1_512 = ResPath(32, 4, m1_512)

    ## encoder stage 2
    s2 = encoder.get_layer("block2_conv2").output 
    e2 = encoder1.get_layer("conv1/relu").output
    m2_256 = MultiResBlock(32, m2_inputs)
    hed2 = hed(m2_256)
    m1_256 = Concatenate()([m1_512_p,m2_256,s2,e2,hed2])
    m1_256 = MultiResBlock(32*2, m1_256)
    m1_256 = PAM_Module(in_dim=m1_256.shape[-1],name='256')(m1_256)
    m1_256_p = MaxPooling2D(pool_size=(2, 2))(m1_256)
    m2_256_p = MaxPooling2D(pool_size=(2, 2))(m2_256)
    # m1_256 = ResPath(32*2, 3, m1_256)

    ## encoder stage 3
    s3 = encoder.get_layer("block3_conv2").output
    e3 = encoder1.get_layer("pool2_conv").output

    m3_128 = MultiResBlock(32, m3_inputs)
    # hed31 = hed(m3_128)
    m2_128 = Concatenate()([m2_256_p,m3_128])
    m2_128 = MultiResBlock(32*2, m2_128)
    hed32 = hed(m2_128)
    m1_128 = Concatenate()([m1_256_p,m2_128,s3,e3,hed32])
    m1_128 = MultiResBlock(32*4, m1_128)
    m1_128 = PAM_Module(in_dim=m1_128.shape[-1],name='128')(m1_128)
    m1_128_p = MaxPooling2D(pool_size=(2, 2))(m1_128)
    m2_128_p = MaxPooling2D(pool_size=(2, 2))(m2_128)
    m3_128_p = MaxPooling2D(pool_size=(2, 2))(m3_128)
    # m1_128 = ResPath(32*4, 2, m1_128)

    ## encoder stage 4
    s4 = encoder.get_layer("block4_conv2").output
    e4 = encoder1.get_layer("pool3_conv").output

    m4_64 = MultiResBlock(32, m4_inputs)
    # hed41 = hed(m4_64)
    m3_64 = Concatenate()([m3_128_p,m4_64])
    m3_64 = MultiResBlock(32*2, m3_64)
    # hed42 = hed(m3_64)
    m2_64 = Concatenate()([m2_128_p,m3_64])
    m2_64 = MultiResBlock(32*4, m2_64)
    hed43 = hed(m2_64)
    m1_64 = Concatenate()([m1_128_p,m2_64,s4,e4,hed43])
    m1_64 = MultiResBlock(32*8, m1_64)
    m1_64 = PAM_Module(in_dim=m1_64.shape[-1],name='64')(m1_64)
    m1_64_p = MaxPooling2D(pool_size=(2, 2))(m1_64)
    m2_64_p = MaxPooling2D(pool_size=(2, 2))(m2_64)
    m3_64_p = MaxPooling2D(pool_size=(2, 2))(m3_64)
    m4_64_p = MaxPooling2D(pool_size=(2, 2))(m4_64)
    # m1_64 = ResPath(32*8, 1, m1_64)

    ## encoder stage 5

    s5 = encoder.get_layer("block5_conv2").output
    e5 = encoder1.get_layer("pool4_conv").output

    m5_32 = MultiResBlock(32, m5_inputs)
    # hed51 = hed(m5_32)
    m4_32 = Concatenate()([m4_64_p,m5_32])
    m4_32 = MultiResBlock(32*2, m4_32)
    # hed52 = hed(m4_32)
    m3_32 = Concatenate()([m3_64_p,m4_32])
    m3_32 = MultiResBlock(32*4, m3_32)
    # hed53 = hed(m3_32)
    m2_32 = Concatenate()([m2_64_p,m3_32])
    m2_32 = MultiResBlock(32*8, m2_32)
    hed54 = hed(m2_32)
    m1_32 = Concatenate()([m1_64_p,m2_32,s5,e5,hed54])
    m1_32 = MultiResBlock(32*16, m1_32)
    m1_32 = PAM_Module(in_dim=m1_32.shape[-1],name='32')(m1_32)

    m2_32 = MultiResBlock(32*8, m1_32)
    m3_32 = MultiResBlock(32*4, m2_32)
    m4_32 = MultiResBlock(32*2, m3_32)
    m5_32 = MultiResBlock(32, m4_32)

    
    m1_32 = TokenizedMLPBlock(32*16, 512)(m1_32)
    m1_64 = TokenizedMLPBlock(32*8, 32*8)(m1_64)
    m1_128 = TokenizedMLPBlock(32*4, 32*4)(m1_128)
    m1_256 = TokenizedMLPBlock(32*2, 32*2)(m1_256)
    m1_512 = TokenizedMLPBlock(32, 32)(m1_512)



    
    
    m1_512z,map_512 = boundary_enhancer(m1_512,m1_256)
    m1_256z,map_256 = boundary_enhancer(m1_256,m1_128)
    m1_128z,map_128 = boundary_enhancer(m1_128,m1_64)
    m1_64z,map_64 = boundary_enhancer(m1_64,m1_32)
    
    

    m5_map = conv2d_bn(m5_32, 1, 1, 1, activation='sigmoid')

    
    x1_rfb=RFBModified(m1_512,32)
    x2_rfb=RFBModified(m1_256,32)
    x3_rfb=RFBModified(m1_128,32)
    x4_rfb=RFBModified(m1_64,32)
    x5_rfb=RFBModified(m1_32,32)

    ra5_feat = aggregation(x1_rfb,x2_rfb,x3_rfb,x4_rfb,x5_rfb)
    crop_5 = tf.image.resize(ra5_feat, [32,32])
    x = -1 * (tf.math.sigmoid(crop_5)) + 1
    x = tf.keras.layers.Multiply()([x, m1_32])
    
    gating_8 = gating_signal(x, 512, batch_norm=True)
    att_8,att_map64 = attention_block(m1_64z, gating_8, 512)
    
    x = BasicConv2D(x, 32*16, 1)
    x = BasicConv2D(x, 32*16, 5, padding='same')
    x = BasicConv2D(x, 32*16, 5, padding='same')
    x_ = BasicConv2D(x, 32*16, 5, padding='same')
    ra4_feat = BasicConv2D(x_, 1, 1)
    x = tf.keras.layers.Add()([ra4_feat, crop_5])
    crop_4 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = -1 * (tf.math.sigmoid(crop_4)) + 1
    x = tf.keras.layers.Multiply()([x, att_8])
    

    gating_32 = gating_signal(x, 256, batch_norm=True)
    att_32,att_map128 = attention_block(m1_128z, gating_32, 256)
    
    x = BasicConv2D(x, 32*8, 1)
    x_ = BasicConv2D(x_, 32*8, 3, padding='same')
    x_ = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x_)
    x = tf.keras.layers.Concatenate()([x,x_])
    x = BasicConv2D(x, 32*8, 3, padding='same')
    x_ = BasicConv2D(x, 32*8, 3, padding='same')
    ra3_feat = BasicConv2D(x_, 1, 3, padding='same')
    x = tf.keras.layers.Add()([ra3_feat, crop_4])
    guide_64 = tf.keras.activations.sigmoid(x)
    guide_64 = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(guide_64)
    crop_3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = -1 * (tf.math.sigmoid(crop_3)) + 1
    x = tf.keras.layers.Multiply()([x, att_32])


    gating_64 = gating_signal(x, 128, batch_norm=True)
    att_64,att_map256 = attention_block(m1_256z, gating_64, 128)
    
    x = BasicConv2D(x, 32*4, 1)
    x_ = BasicConv2D(x_, 32*4, 3, padding='same')
    x_ = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x_)
    x = tf.keras.layers.Concatenate()([x,x_])
    x = BasicConv2D(x, 32*4, 3, padding='same')
    x_ = BasicConv2D(x, 32*4, 3, padding='same')
    ra2_feat = BasicConv2D(x_, 1, 3, padding='same')
    x = tf.keras.layers.Add()([ra2_feat, crop_3])
    guide_128 = tf.keras.activations.sigmoid(x)
    guide_128 = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(guide_128)
    crop_2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = -1 * (tf.math.sigmoid(crop_2)) + 1
    x = tf.keras.layers.Multiply()([x, att_64])
   

    gating_128 = gating_signal(x, 64, batch_norm=True)
    att_128,att_map512 = attention_block(m1_512z, gating_128, 64)
    
    x = BasicConv2D(x, 32*2, 1)
    x_ = BasicConv2D(x_, 32*2, 3, padding='same')
    x_ = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x_)
    x = tf.keras.layers.Concatenate()([x,x_])
    x = BasicConv2D(x, 32*2, 3, padding='same')
    x_ = BasicConv2D(x, 32*2, 3, padding='same')
    ra1_feat = BasicConv2D(x_, 1, 3, padding='same')
    x = tf.keras.layers.Add()([ra1_feat, crop_2])
    guide_256 = tf.keras.activations.sigmoid(x)
    guide_256 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(guide_256)
    crop_1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = -1 * (tf.math.sigmoid(crop_1)) + 1
    x = tf.keras.layers.Multiply()([x, att_128])
    
    x = BasicConv2D(x, 32, 1)
    x_ = BasicConv2D(x_, 32, 3, padding='same')
    x_ = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x_)
    x = tf.keras.layers.Concatenate()([x,x_])
    x = BasicConv2D(x, 32, 3, padding='same')
    x = BasicConv2D(x, 32, 3, padding='same')
    ra0_feat = BasicConv2D(x, 1, 3, padding='same')
    x = tf.keras.layers.Add()([ra0_feat, crop_1])
    final_map = tf.keras.activations.sigmoid(x)
    
    model = Model(inputs, [final_map,final_map,guide_256,guide_128,guide_64,map_512,map_256,map_128,map_64])

    opt = Adam(lr=0.00005, beta_1=0.5)
    
    model.compile(loss=[structure_loss,focal_loss,bce_dice_loss,bce_dice_loss,bce_dice_loss,bce_dice_loss,bce_dice_loss,bce_dice_loss,bce_dice_loss],optimizer=opt, loss_weights=[25,15,15,10,5,10,7,5,3])
    return model