from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

# """Create a rpn layer
#     Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
#             Keep the padding 'same' to preserve the feature map's size
#     Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
#             classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
#             regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
# Args:
#     base_layers: vgg in here
#     num_anchors: 9 in here

# Returns:
#     [x_class, x_regr, base_layers]
#     x_class: classification for whether it's an object
#     x_regr: bboxes regression
#     base_layers: vgg in here
# """
def rpn_layer(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]