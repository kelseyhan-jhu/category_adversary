
l
data	DataInput"&
shape:???????????"1
_output_shapes
:???????????
?
conv1_1Convdata"
group"1
_output_shapes
:???????????@"
strides
"
use_bias("
pads

    "
kernel_shape
@
K
relu1_1Reluconv1_1"1
_output_shapes
:???????????@
?
conv1_2Convrelu1_1"1
_output_shapes
:???????????@"
pads

    "
kernel_shape
@@"
use_bias("
group"
strides

K
relu1_2Reluconv1_2"1
_output_shapes
:???????????@
?
pool1Poolrelu1_2"
pads

      "
pooling_typeAVG"
kernel_shape
"1
_output_shapes
:???????????@"
strides

?
conv2_1Convpool1"
strides
"
pads

    "
kernel_shape	
@?"2
_output_shapes 
:????????????"
group"
use_bias(
L
relu2_1Reluconv2_1"2
_output_shapes 
:????????????
?
conv2_2Convrelu2_1"
pads

    "
use_bias("
group"
strides
"2
_output_shapes 
:????????????"
kernel_shape

??
L
relu2_2Reluconv2_2"2
_output_shapes 
:????????????
?
pool2Poolrelu2_2"
strides
"0
_output_shapes
:?????????@@?"
pooling_typeAVG"
pads

      "
kernel_shape

?
conv3_1Convpool2"
kernel_shape

??"
strides
"
use_bias("0
_output_shapes
:?????????@@?"
pads

    "
group
J
relu3_1Reluconv3_1"0
_output_shapes
:?????????@@?
?
conv3_2Convrelu3_1"
use_bias("0
_output_shapes
:?????????@@?"
pads

    "
group"
strides
"
kernel_shape

??
J
relu3_2Reluconv3_2"0
_output_shapes
:?????????@@?
?
conv3_3Convrelu3_2"
group"
use_bias("
strides
"0
_output_shapes
:?????????@@?"
kernel_shape

??"
pads

    
J
relu3_3Reluconv3_3"0
_output_shapes
:?????????@@?
?
conv3_4Convrelu3_3"
strides
"
kernel_shape

??"
use_bias("
group"
pads

    "0
_output_shapes
:?????????@@?
J
relu3_4Reluconv3_4"0
_output_shapes
:?????????@@?
?
pool3Poolrelu3_4"
kernel_shape
"
strides
"
pooling_typeAVG"0
_output_shapes
:?????????  ?"
pads

      
?
conv4_1Convpool3"
kernel_shape

??"0
_output_shapes
:?????????  ?"
strides
"
use_bias("
group"
pads

    
J
relu4_1Reluconv4_1"0
_output_shapes
:?????????  ?
?
conv4_2Convrelu4_1"
use_bias("
pads

    "
kernel_shape

??"0
_output_shapes
:?????????  ?"
strides
"
group
J
relu4_2Reluconv4_2"0
_output_shapes
:?????????  ?
?
conv4_3Convrelu4_2"
use_bias("
group"
strides
"
pads

    "0
_output_shapes
:?????????  ?"
kernel_shape

??
J
relu4_3Reluconv4_3"0
_output_shapes
:?????????  ?
?
conv4_4Convrelu4_3"
use_bias("
kernel_shape

??"
strides
"
pads

    "
group"0
_output_shapes
:?????????  ?
J
relu4_4Reluconv4_4"0
_output_shapes
:?????????  ?
?
pool4Poolrelu4_4"
strides
"
pooling_typeAVG"
kernel_shape
"
pads

      "0
_output_shapes
:??????????
?
conv5_1Convpool4"
pads

    "
group"
use_bias("
strides
"0
_output_shapes
:??????????"
kernel_shape

??
J
relu5_1Reluconv5_1"0
_output_shapes
:??????????
?
conv5_2Convrelu5_1"
group"
kernel_shape

??"0
_output_shapes
:??????????"
use_bias("
pads

    "
strides

J
relu5_2Reluconv5_2"0
_output_shapes
:??????????
?
conv5_3Convrelu5_2"
strides
"
kernel_shape

??"
pads

    "0
_output_shapes
:??????????"
group"
use_bias(
J
relu5_3Reluconv5_3"0
_output_shapes
:??????????
?
conv5_4Convrelu5_3"
pads

    "0
_output_shapes
:??????????"
group"
strides
"
kernel_shape

??"
use_bias(
J
relu5_4Reluconv5_4"0
_output_shapes
:??????????
?
pool5Poolrelu5_4"
pooling_typeAVG"
pads

      "0
_output_shapes
:??????????"
kernel_shape
"
strides
