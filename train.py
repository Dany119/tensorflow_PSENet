import time
import datetime
import os
import numpy as np
import tensorflow as tf
from utils.utils_tool import logger, cfg
import model
from utils.data_provider import data_provider

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


input_size = 512
batch_size_per_gpu = 8
num_readers = 32
learning_rate = 0.00001
max_steps = 20
moving_average_decay = 0.997
decay_rate = 0.94
decay_steps = 16
# tgpu_list = '0'
checkpoint_path = './resnet_train/'
logs_path = 'logs_mlt/'
restore = True
save_checkpoint_steps = 1000
save_summary_steps = 100
pretrained_model_path = '../Data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# gpus = list(range(len(gpu_list.split(','))))

logger.setLevel(cfg.debug)
now = datetime.datetime.now()
StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(logs_path + StyleTime)


def tower_loss(seg_maps_pred, seg_maps_gt, training_masks):
    # Build inference graph
    model_loss = model.loss(seg_maps_gt, seg_maps_pred, training_masks)
    regularization_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = model_loss + sum(regularization_losses)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


    #input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    #input_seg_maps = tf.placeholder(tf.float32, shape=[None, None, None, 6], name='input_score_maps')
    #input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
global_step = tf.Variable([0], trainable=False, name='global_step')
# add summary
tf.summary.scalar('learning_rate', learning_rate)
# opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9)
opt = tf.keras.optimizers.Adam(learning_rate)
# opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
loss_metric1 = tf.keras.metrics.Mean(name='model_loss')
loss_metric2 = tf.keras.metrics.Mean(name='total_loss')
psenet = model.PSEnet().model

# split
# input_images_split = tf.split(input_images, len(gpus))
# input_seg_maps_split = tf.split(input_seg_maps, len(gpus))
# input_training_masks_split = tf.split(input_training_masks, len(gpus))
@tf.function(experimental_relax_shapes=True)
def train_step(input_images, input_seg_maps, input_traing_masks):
    with tf.GradientTape() as tape:
        #tower_grads = []
        # for i, gpu_id in enumerate(gpus):
        #     with tf.device('/gpu:%d' % gpu_id):
        #         with tf.name_scope('model_%d' % gpu_id) as scope:
        # iis = input_images_split[i]
        # isegs = input_seg_maps_split[i]
        # itms = input_training_masks_split[i]
        seg_maps_pred = psenet(input_images)
        total_loss, model_loss = tower_loss(seg_maps_pred, input_seg_maps, input_traing_masks)
        gradients = tape.gradient(total_loss, psenet.trainable_variables)
        opt.apply_gradients(zip(gradients, psenet.trainable_variables))
        # tower_grads.append(grads)
        # grads = average_gradients(tower_grads)
        # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        variable_averages.apply(tf.compat.v1.trainable_variables())

        loss_metric1.update_state(model_loss)
        loss_metric2.update_state(total_loss)
# batch norm updates


#gpu_options=tf.GPUOptions(allow_growth=True)
#gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
if restore:
    logger.info('continue training from previous checkpoint')
    psenet.load_weights(checkpoint_path)
else:
    psenet.load_weights(pretrained_model_path, by_name=True)

# data_generator = data_provider.get_batch(num_workers=num_readers,
#                                          input_size=input_size,
#                                          batch_size=batch_size_per_gpu)
# #                                         batch_size=batch_size_per_gpu * len(gpus))
data_generator = data_provider.generator(input_size=input_size,
                                         batch_size=batch_size_per_gpu)

summary_writer = tf.summary.create_file_writer(logs_path + StyleTime)
start = time.time()
for step in range(max_steps):
    data = next(data_generator)
    input_images = tf.cast(data[0], dtype=tf.float32)
    input_images = model.mean_image_subtraction(input_images)

    input_seg_maps = tf.cast(data[2], dtype=tf.float32)
    input_training_masks = tf.cast(data[3], dtype=tf.float32)
    train_step(input_images, input_seg_maps, input_training_masks)
    model_mean_loss = loss_metric1.result()
    total_mean_loss = loss_metric2.result()
    with summary_writer.as_default():
        tf.summary.scalar('learning_rate', learning_rate, step=step)
        tf.summary.scalar('model_loss', model_mean_loss, step=step)
        tf.summary.scalar('total_loss', total_mean_loss, step=step)

    if step == 0:
        psenet.summary()

    if step != 0 and step % decay_steps == 0:
        learning_rate = learning_rate * decay_rate

    if step % 10 == 0:
        avg_time_per_step = (time.time() - start)/10
        avg_examples_per_second = (10 * batch_size_per_gpu)/(time.time() - start)
        start = time.time()
        logger.info('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
            step, model_mean_loss, total_mean_loss, avg_time_per_step, avg_examples_per_second))

    if (step + 1) == max_steps:
        psenet.save_weights(checkpoint_path)
