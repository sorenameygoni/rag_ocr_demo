import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
import cv2


VOCAB = r" !" + r'"' + r"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`acdehilmnrs{|}~·"


char_to_num = {c: i + 1 for i, c in enumerate(VOCAB)}
num_to_char = {i: c for c, i in char_to_num.items()}

def decode_batch_preds(logits, input_lengths, beam_width=50, top_paths=1):

    if not tf.is_tensor(logits):
        logits_tf = tf.convert_to_tensor(logits, dtype=tf.float32)
    else:
        logits_tf = tf.cast(logits, tf.float32)

    if not tf.is_tensor(input_lengths):
        input_len_tf = tf.convert_to_tensor(input_lengths, dtype=tf.int32)
    else:
        input_len_tf = tf.cast(input_lengths, tf.int32)

    t = tf.shape(logits_tf)[1]
    input_len_tf = tf.clip_by_value(input_len_tf, 0, t)

    probs = tf.nn.softmax(logits_tf, axis=-1)
    greedy = (beam_width <= 1)
    
    decoded, _ = K.ctc_decode(probs, input_length=input_len_tf,
                              greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    
    decoded_np = [d.numpy() for d in decoded]
    B = decoded_np[0].shape[0]

    out = []
    for i in range(B):
        per = []
        for p in range(top_paths):
            seq = decoded_np[p][i]
            chars = []
            for idx in seq:
                idx = int(idx)
                if idx == -1: break
                if idx == 0:  continue 

                chars.append(num_to_char.get(idx, ''))
            per.append(''.join(chars))
        out.append(per if top_paths > 1 else per[0])
    return out

def preprocess_image(image_path_or_bytes):

    img = tf.io.read_file(image_path_or_bytes)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    

    target_height = 48
    img_shape = tf.shape(img)
    h = tf.cast(img_shape[0], tf.float32)
    w = tf.cast(img_shape[1], tf.float32)
    

    new_width = tf.cast(w * (target_height / h), tf.int32)
    img = tf.image.resize(img, [target_height, new_width])
    

    img = tf.image.resize_with_pad(img, target_height, 1024)
    

    img = tf.cast(img, tf.float32) / 255.0
    
    img = tf.expand_dims(img, axis=0) 
    return img



class TemporalAdditiveAttention(layers.Layer):
    def __init__(self, hidden_dim, attn_dim=64, dropout=0.0, use_layernorm=True, gamma_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.dense_W = layers.Dense(attn_dim, use_bias=True, kernel_initializer="glorot_uniform")
        self.dense_v = layers.Dense(1, use_bias=False, kernel_initializer="glorot_uniform")
        self.use_dropout = True if (dropout and dropout > 0.0) else False
        self.dropout_layer = layers.Dropout(dropout) if self.use_dropout else None
        self.use_layernorm = use_layernorm
        self.layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-5, dtype="float32") if use_layernorm else None
        self.gamma_init = gamma_init
        self.gamma = self.add_weight(name="gamma", shape=(), initializer=tf.keras.initializers.Constant(self.gamma_init), trainable=True)

    def call(self, rnn_out, training=None):
        proj = tf.tanh(self.dense_W(rnn_out))
        scores = self.dense_v(proj)
        attn_weights = tf.nn.softmax(scores, axis=1)
        attended = rnn_out * attn_weights
        if self.use_dropout:
            attended = self.dropout_layer(attended, training=training)
        out = rnn_out + self.gamma * attended
        if self.use_layernorm and self.layernorm is not None:
            out = self.layernorm(out)
        return out, attn_weights

def conv3x3(filters, stride=(1, 1), name=None):
    return layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", use_bias=False, name=name, kernel_initializer="he_normal")

def conv1x1(filters, stride=(1, 1), name=None):
    return layers.Conv2D(filters, kernel_size=1, strides=stride, padding="same", use_bias=False, name=name, kernel_initializer="he_normal")

class BasicBlock(layers.Layer):
    expansion = 1
    def __init__(self, filters, stride=(1, 1), downsample=None, name_prefix="bb", **kwargs):
        super().__init__(**kwargs)
        self.conv1 = conv3x3(filters, stride=stride, name=f"{name_prefix}_conv1")
        self.bn1 = layers.BatchNormalization(name=f"{name_prefix}_bn1", epsilon=1e-5, dtype="float32")
        self.relu = layers.Activation("relu")
        self.conv2 = conv3x3(filters, stride=(1, 1), name=f"{name_prefix}_conv2")
        self.bn2 = layers.BatchNormalization(name=f"{name_prefix}_bn2", epsilon=1e-5, dtype="float32")
        self.downsample = downsample

    def call(self, x, training=None):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        if self.downsample is not None:
            identity = self.downsample(x, training=training)
        out = layers.add([out, identity])
        out = self.relu(out)
        return out

class ResNet34Backbone(tf.keras.Model):
    def __init__(self, stem_conv_stride=(2, 1), stem_pool_stride=(2, 1), **kwargs):
        super().__init__(**kwargs)
        self.stem_conv = layers.Conv2D(64, 7, strides=stem_conv_stride, padding="same", use_bias=False, kernel_initializer="he_normal", name="conv1")
        self.stem_bn = layers.BatchNormalization(name="bn_conv1", epsilon=1e-5, dtype="float32")
        self.stem_relu = layers.Activation("relu")
        self.stem_pool = layers.MaxPool2D(pool_size=3, strides=stem_pool_stride, padding="same", name="pool1")
        self.inplanes = 64
        self.layer_cfg = [3, 4, 6, 3]
        self._make_layers()

    def _make_layer(self, filters, blocks, stride=(1, 1), name_prefix="layer"):
        layers_list = []
        if stride != (1, 1) or self.inplanes != filters * BasicBlock.expansion:
            downsample = tf.keras.Sequential([
                conv1x1(filters * BasicBlock.expansion, stride=stride, name=f"{name_prefix}_0_down_conv"),
                layers.BatchNormalization(name=f"{name_prefix}_0_down_bn", epsilon=1e-5, dtype="float32"),
            ], name=f"{name_prefix}_0_downsample")
        else:
            downsample = None
        layers_list.append(BasicBlock(filters, stride=stride, downsample=downsample, name_prefix=f"{name_prefix}_0"))
        self.inplanes = filters * BasicBlock.expansion
        for i in range(1, blocks):
            layers_list.append(BasicBlock(filters, stride=(1, 1), downsample=None, name_prefix=f"{name_prefix}_{i}"))
        return tf.keras.Sequential(layers_list, name=name_prefix)

    def _make_layers(self):
        self.layer1 = self._make_layer(64, self.layer_cfg[0], stride=(1, 1), name_prefix="conv2")
        self.layer2 = self._make_layer(128, self.layer_cfg[1], stride=(2, 2), name_prefix="conv3")
        self.layer3 = self._make_layer(256, self.layer_cfg[2], stride=(2, 2), name_prefix="conv4")
        self.layer4 = self._make_layer(512, self.layer_cfg[3], stride=(2, 2), name_prefix="conv5")

    def call(self, x, training=None):
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_relu(x)
        x = self.stem_pool(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        return x

class CRNN_ResNet34_LSTM_Attn(tf.keras.Model):
    def __init__(self, num_classes, rnn_hidden=256, rnn_layers=1, bidirectional=True, dropout=0.0, attn_dim=64, attn_dropout=0.0, gamma_init=0.0, use_layernorm=True, backbone_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = ResNet34Backbone(**backbone_kwargs)
        self.reduce_conv = layers.Conv1D(rnn_hidden, kernel_size=1, name="reduce_conv", dtype="float32")
        self.rnn_stack = []
        for i in range(rnn_layers):
            lstm = layers.LSTM(rnn_hidden, return_sequences=True, dropout=dropout if (rnn_layers > 1 and i > 0) else 0.0, recurrent_dropout=0.0)
            if bidirectional:
                lstm = layers.Bidirectional(lstm, merge_mode="concat")
            self.rnn_stack.append(lstm)
        self.fc_in = rnn_hidden * (2 if bidirectional else 1)
        self.attn = TemporalAdditiveAttention(hidden_dim=self.fc_in, attn_dim=attn_dim, dropout=attn_dropout, use_layernorm=use_layernorm, gamma_init=gamma_init)
        self.final_bn = layers.BatchNormalization(epsilon=1e-5, dtype="float32")
        self.fc = layers.Dense(num_classes, kernel_initializer="glorot_uniform", dtype="float32")

    def call(self, inputs, return_attn=False, training=None):
        
        x = tf.cast(inputs, tf.float32)
        imagenet_mean = tf.constant([0.485, 0.456, 0.406], shape=(1, 1, 1, 3), dtype=x.dtype)
        imagenet_std = tf.constant([0.229, 0.224, 0.225], shape=(1, 1, 1, 3), dtype=x.dtype)
        x = (x - imagenet_mean) / imagenet_std
        
        f = self.backbone(x, training=training)
        f = tf.reduce_mean(f, axis=1)
        f = self.reduce_conv(f, training=training)
        r = f
        for layer in self.rnn_stack:
            r = layer(r, training=training)
        attn_out, attn_weights = self.attn(r, training=training)
        stabilized_attn_out = self.final_bn(attn_out, training=training)
        logits = self.fc(stabilized_attn_out)
        if return_attn:
            return logits, attn_weights
        return logits


def load_ocr_model(weights_path):
    num_classes = len(VOCAB) + 1 
    model = CRNN_ResNet34_LSTM_Attn(num_classes=num_classes)
    

    dummy_input = tf.zeros((1, 48, 1024, 3), dtype=tf.float32)
    model(dummy_input)
    

    model.load_weights(weights_path)
    return model

def predict_ocr(model, image_path):
    processed_img = preprocess_image(image_path)
    
    logits = model(processed_img, training=False)
    
    feat_len = tf.shape(logits)[1] 
    input_lens = tf.expand_dims(feat_len, axis=0)
    

    text_list = decode_batch_preds(logits, input_lens, beam_width=10)
    return text_list[0]