import argparse
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
#import custom functions
import custom_functions as cf

parser = argparse.ArgumentParser(description='Get functional annotations.')
parser.add_argument('--input_file', type = str, default = "./work_dir/temp.h5", help = 'Input file in hdf5 format')
parser.add_argument('--work_dir', type = str, default = "./work_dir/", help = 'Work directory')
parser.add_argument('--mode', type = str, default = "fast", help = 'Prediction mode')
parser.add_argument('--seq_len', type = int, default = 131072, help = 'Input sequence length')
parser.add_argument('--batch_size', type = int, default = 16, help = 'Batch size')
parser.add_argument('--gpu', type = str, default = "-1", help = 'GPU id, start from 0. -1 for CPU')

args = parser.parse_args()
# select GPU
if args.gpu != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

model_type=sys.argv[1]
category=sys.argv[2]
data_part=sys.argv[3]
prefix_in=sys.argv[5]


data_path="./"
data_prefix=prefix_in+"_"
#data_prefix="pre_"
output_path="/lustre/grp/bitcap/wangy/SV/SVEN/small_variants/output2/anno/"
data_indicator="X"
#data_indicator="alt"
output_prefix=prefix_in

#define conv_block functions
def conv_block(inputs, filters=128, kernel_size=1, strides=1, 
    dilation_rate=1, l2_scale=0, dropout=0, pool_size=1, bn_momentum=0.9,
    residual=False, kernel_initializer='he_normal', padding='same',norm_gamma=None):

    current = inputs
    #activation
    current = layers.Activation('gelu')(current)
    #convolution
    current=layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_scale)
    )(current)
    #batch normalization
    current=layers.BatchNormalization(momentum=bn_momentum,gamma_initializer=norm_gamma)(current)
    # dropout
    if dropout > 0:
        current = layers.Dropout(dropout)(current)
    # residual add
    if residual:
        current = layers.Add()([inputs,current])
    # Pool
    if pool_size > 1:
        current = layers.MaxPool1D(pool_size=pool_size,padding=padding)(current)
    #return
    return current

def dilated_residual(inputs, filters, kernel_size=3, rate_mult=2, dropout=0, repeat=1, **kwargs):

    current = inputs
    # initialize dilation rate
    dilation_rate = 1

    for ri in range(repeat):
        rep_input = current
        # dilate
        current = conv_block(current,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            norm_gamma='ones',
            **kwargs)
        # return
        current = conv_block(current,
            filters=rep_input.shape[-1],
            dropout=dropout,
            norm_gamma='zeros',
            **kwargs)
        # residual add
        current = layers.Add()([rep_input,current])
        #update dilation rate
        dilation_rate=int(np.round(float(dilation_rate)*rate_mult))
    #current
    return current

def Rconv_block(inputs,filters,kernel_size=1,residual=True):
    current=inputs
    current=layers.BatchNormalization(momentum=0.9)(current)
    current = layers.Activation('gelu')(current)
    current=layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    if residual:
        current= layers.Add()([inputs,current])
    return current

def tf_sep_f1_model(input_length):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = cf.StochasticReverseComplement()(current)
    #shift sequence
    current = cf.StochasticShift(3)(current)
    #first conv block
    current=layers.Conv1D(
        filters=128,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current,filters=128)
    current = layers.MaxPool1D(pool_size=2,padding="same")(current)
    filter_num=192
    for x in range(6):
        current = Rconv_block(current,filters=filter_num,kernel_size=5,residual=False)
        current = Rconv_block(current,filters=filter_num)
        current = layers.MaxPool1D(pool_size=2,padding="same")(current)
        filter_num=int(filter_num*1.15)
    
    #dr block
    current=dilated_residual(current, filters=384, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current=layers.Cropping1D(cropping=64)(current)
    current=conv_block(current, filters=512, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(1)(current)
    current = layers.Activation('softplus')(current)
    outputs = cf.SwitchReverse()([current,reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="tf_sep_f1_model")
    #model.summary()
    return model

def acc_build_in_model(input_length=131_072):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #first conv block
    current=layers.Conv1D(
        filters=384,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current,filters=384)
    current = layers.MaxPool1D(pool_size=2,padding="same")(current)
    filter_num=384
    for x in range(6):
        current = Rconv_block(current,filters=filter_num,kernel_size=5,residual=False)
        current = Rconv_block(current,filters=filter_num)
        current = layers.MaxPool1D(pool_size=2,padding="same")(current)
        filter_num=int(filter_num*1.1496)
    
    #dr block
    current=dilated_residual(current, filters=766, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current=layers.Cropping1D(cropping=64)(current)
    current=conv_block(current, filters=1536, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(684)(current)
    outputs = layers.Activation('softplus')(current)

    model = keras.Model(inputs=sequence, outputs=outputs, name="acc_hol_model")
    return model

def acc_hol_model(input_length):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = cf.StochasticReverseComplement()(current)
    #shift sequence
    current = cf.StochasticShift(3)(current)
    #first conv block
    current=layers.Conv1D(
        filters=384,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current,filters=384)
    current = layers.MaxPool1D(pool_size=2,padding="same")(current)
    filter_num=384
    for x in range(6):
        current = Rconv_block(current,filters=filter_num,kernel_size=5,residual=False)
        current = Rconv_block(current,filters=filter_num)
        current = layers.MaxPool1D(pool_size=2,padding="same")(current)
        filter_num=int(filter_num*1.1496)
    
    #dr block
    current=dilated_residual(current, filters=766, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current=layers.Cropping1D(cropping=64)(current)
    current=conv_block(current, filters=1536, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(684)(current)
    current = layers.Activation('softplus')(current)
    outputs = cf.SwitchReverse()([current,reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="acc_hol_model")
    return model

def tf_his_sep_model(input_length):
    #acc model
    acc_model=acc_build_in_model()
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = cf.StochasticReverseComplement()(current)
    #shift sequence
    current = cf.StochasticShift(3)(current)
    acc_input=current
    #first conv block
    current=layers.Conv1D(
        filters=192,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current,filters=192)
    current = layers.MaxPool1D(pool_size=2,padding="same")(current)
    filter_num=192
    for x in range(6):
        current = Rconv_block(current,filters=filter_num,kernel_size=5,residual=False)
        current = Rconv_block(current,filters=filter_num)
        current = layers.MaxPool1D(pool_size=2,padding="same")(current)
        filter_num=int(filter_num*1.1496)

    #dr block
    current=dilated_residual(current, filters=384, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current=layers.Cropping1D(cropping=64)(current)
    acc_labels_sr=acc_model(acc_input)
    current =layers.Concatenate()([current,acc_labels_sr])
    current=conv_block(current, filters=512, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(1)(current)
    current = layers.Activation('softplus')(current)
    outputs = cf.SwitchReverse()([current,reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="tf_his_sep_model")
    return model

def his_hol_model(input_length):
    #acc model
    acc_model=acc_build_in_model()
    
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = cf.StochasticReverseComplement()(current)
    #shift sequence
    current = cf.StochasticShift(3)(current)
    acc_input=current
    acc_labels_sr=acc_model(acc_input)
    #first conv block
    current=layers.Conv1D(
        filters=384,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current,filters=384)
    current = layers.MaxPool1D(pool_size=2,padding="same")(current)
    filter_num=384
    for x in range(6):
        current = Rconv_block(current,filters=filter_num,kernel_size=5,residual=False)
        current = Rconv_block(current,filters=filter_num)
        current = layers.MaxPool1D(pool_size=2,padding="same")(current)
        filter_num=int(filter_num*1.1496)

    #dr block
    current=dilated_residual(current, filters=766, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=12)
    #conv block
    current=layers.Cropping1D(cropping=64)(current)
    
    current =layers.Concatenate()([current,acc_labels_sr])
    current=conv_block(current, filters=1536, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(1976)(current)
    current = layers.Activation('softplus')(current)
    outputs = cf.SwitchReverse()([current,reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="his_hol_model")

    return model

def tf_hol_model(input_length):
    #input sequence
    sequence = keras.Input(shape=(input_length, 4), name='sequence')
    current = sequence
    #rc sequence
    current, reverse_bool = cf.StochasticReverseComplement()(current)
    #shift sequence
    current = cf.StochasticShift(3)(current)
    #first conv block
    current=layers.Conv1D(
        filters=768,
        kernel_size=15,
        strides=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(0)
    )(current)
    current = Rconv_block(current,filters=768)
    current = layers.MaxPool1D(pool_size=2)(current)
    filter_num=768
    for x in range(6):
        current = Rconv_block(current,filters=filter_num,kernel_size=5,residual=False)
        current = Rconv_block(current,filters=filter_num)
        current = layers.MaxPool1D(pool_size=2)(current)
        filter_num=int(filter_num*1.1494)

    #dr block
    current=dilated_residual(current, filters=1536, kernel_size=3, rate_mult=1.5, dropout=0.3, repeat=11)
    #conv block
    current=layers.Cropping1D(cropping=64)(current)
    current=conv_block(current, filters=1536, kernel_size=1,dropout=0.05)

    current = layers.Activation('gelu')(current)

    current = layers.Dense(2540)(current)
    current = layers.Activation('softplus')(current)
    outputs = cf.SwitchReverse()([current,reverse_bool])
    
    model = keras.Model(inputs=sequence, outputs=outputs, name="tf_hol_model")
    return model

def run_hol_prediction(category, out_dir, data_x):
    model_params_file = "./model_params/class_oriented_" + category + ".h5"
    if category == "acc":
        model = acc_hol_model(args.seq_len)
        feature_num = 684
    elif category == "tf":
        model = tf_hol_model(args.seq_len)
        feature_num = 2540
    elif category == "his":
        model = his_hol_model(args.seq_len)
        feature_num = 1976
    else:
        raise ValueError("Category not found.")
    # load weights
    model.load_weights(model_params_file)
    # predict
    pred_file = out_dir + "class_" + category + ".npy"
    if args.gpu == "-1":
        print("Predicting with CPUs...")
        y_pred = model.predict(data_x, batch_size=args.batch_size, verbose=0)
    else:
        print("Predicting with GPU %s..." % args.gpu)
        y_pred = np.empty([data_x.shape[0], 896, feature_num], dtype=np.float32)
        for y in range(data_x.shape[0]):
            predictions = model(data_x[y].reshape((1, args.seq_len, 4)))
            y_pred[y] = predictions
    if category == "tf":
        y_pred = y_pred[:, :, 684:]
    # save
    np.save(pred_file, y_pred)


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data_x  = h5py.File(args.input_file, 'r')['seq'][:]
    # Create output directory
    out_dir = args.work_dir + "annotations/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Run prediction
    if args.mode == "fast":
        print("Predicting with fast mode...")
        run_hol_prediction("acc", out_dir, data_x)
        run_hol_prediction("tf", out_dir, data_x)
        run_hol_prediction("his", out_dir, data_x)
    else:
        print("Coming soon...")
    print("Success: get functional annotations.")
    