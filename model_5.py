import tensorflow as tf
import numpy as np
from glob import glob
from ops import *
from utils import *
from six.moves import xrange
import time


class MODEL(object):
    def __init__(self, sess, patch_size=40, batch_size=128,
                 output_size=40, input_c_dim=1, output_c_dim=1,
                 sigma=1.4, clip_b=0.025, lr=0.001, epoch=12,
                 ckpt_dir='./checkpoint', sample_dir='./sample', test_save_dir='./test',
                 dataset='BSD400', testset='Set12', evalset='Set12'):
        self.sess = sess
        self.is_gray = (input_c_dim == 1)
        self.batch_size = 180
        self.patch_sioze = 180
        self.output_size = output_size
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.sigma = sigma
        self.clip_b = clip_b
        self.lr = lr
        self.numEpoch = epoch
        self.ckpt_dir = ckpt_dir
        self.trainset = dataset
        self.testset = testset
        self.evalset = evalset
        self.sample_dir = sample_dir
        self.test_save_dir = test_save_dir
        self.epoch = epoch
        # Fixed params
        self.save_every_epoch = 2
        self.eval_every_epoch = 5
        # Adam setting (default setting)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.epsilon = 1e-8

        self.build_model()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.patch_sioze, self.patch_sioze, self.input_c_dim],
                                name='noisy_image')
        self.X_ = tf.placeholder(tf.float32, [None, self.patch_sioze, self.patch_sioze, self.input_c_dim],
                                 name='clean_image')
        # layer 1
        with tf.variable_scope('conv1'):
            layer_1_output = self.layer(self.X, [3, 3, self.input_c_dim, 64], useBN=False)
        # layer 2
        with tf.variable_scope('conv2'):
            layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
        # layer 3
        with tf.variable_scope('conv3'):
            layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64])
        # layer 4
        with tf.variable_scope('conv4'):
            layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64])
        # layer 5
        with tf.variable_scope('conv5'):
            self.Y = self.layer(layer_4_output, [3, 3, 64, self.output_c_dim], useBN=False, useReLU=False)
        # L2 loss
        self.Y_ = self.X - self.X_  # noisy image - clean image
        self.loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_step = optimizer.minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        # create this init op after all variables specified
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        print("[*] Initialize model successfully...")

    def conv_layer(self, inputdata, weightshape, b_init, stridemode):
        # weights
        W = tf.get_variable('weights', weightshape,
                            initializer=tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
        b = tf.get_variable('biases', [1, weightshape[-1]], initializer=tf.constant_initializer(b_init))
        # convolutional layer
        return tf.add(tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME"), b)  # SAME with zero padding

    def bn_layer(self, logits, output_dim, b_init=0.0):
        alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer=
        tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)))
        beta = tf.get_variable('bn_beta', [1, output_dim], initializer=
        tf.constant_initializer(b_init))
        return batch_normalization(logits, alpha, beta, isCovNet=True)

    def layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True, useReLU=True):
        logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode)
        if useReLU == False:
            output = logits
        else:
            if useBN:
                output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
            else:
                output = tf.nn.relu(logits)
        return output

    def train(self):
        # init the variables
        self.sess.run(self.init)
        # get data
        eval_files = glob('./data/test/{}/*.png'.format(self.evalset))
        eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
        # data = load_data(filepath='./data/img_clean_pats.npy')
        data_files = glob('./data/Train400/*.png')
        data = load_images(data_files)

        # numBatch = int(data.shape[0] / self.batch_size)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        iter_num = 0
        print("[*] Start training : ")
        start_time = time.time()
        #self.evaluate(iter_num, eval_data)  # eval_data value range is 0-255
        for epoch in xrange(self.epoch):
            np.random.shuffle(data)
            for idx in xrange(len(data)):
                lvl = 0.1
                for zz in range(10):
                    batch_images = data[idx]
                    #batch_images = np.array(batch_images / 255.0, dtype=np.float32)  # normalize the data to 0-1
                    train_images = add_noise(batch_images, lvl)
                    batch_images = np.array(batch_images / 255.0, dtype=np.float32)  # normalize the data to 0-1
                    train_images = np.array(train_images / 255.0, dtype=np.float32)  # normalize the data to 0-1
                    _, loss, summary = self.sess.run([self.train_step, self.loss, merged],
                                                 feed_dict={self.X: train_images, self.X_: batch_images})
                    print("Epoch: [%2d] brightness level: [%2.2f] image: [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, lvl, idx + 1, len(data), time.time() - start_time, loss))
                    iter_num += 1
                    lvl += 0.2
                    writer.add_summary(summary, iter_num)
            #if np.mod(epoch + 1, self.eval_every_epoch) == 0:
            #    self.evaluate(epoch, eval_data)  # eval_data value range is 0-255
            # save the model
            if np.mod(epoch + 1, self.save_every_epoch) == 0:
                self.save(iter_num)
        print("[*] Finish training.")

    def save(self, iter_num):
        model_name = "MODEL5.model"
        model_dir = "%s_%s_%s" % (self.trainset,
                                  self.batch_size, self.patch_sioze)
        checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print("[*] Saving model...")
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=iter_num)

    def sampler(self, image):
        # set reuse flag to True
        # tf.get_variable_scope().reuse_variables()
        self.X_test = tf.placeholder(tf.float32, image.shape, name='noisy_image_test')
        # layer 1 (adpat to the input image)
        with tf.variable_scope('conv1', reuse=True):
            layer_1_output = self.layer(self.X_test, [3, 3, self.input_c_dim, 64], useBN=False)
        # layer 2 to 4
        with tf.variable_scope('conv2', reuse=True):
            layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64])
        with tf.variable_scope('conv3', reuse=True):
            layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64])
        with tf.variable_scope('conv4', reuse=True):
            layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64])
        # layer 5
        with tf.variable_scope('conv5', reuse=True):
            self.Y_test = self.layer(layer_4_output, [3, 3, 64, self.output_c_dim], useBN=False, useReLU=False)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        model_dir = "%s_%s_%s" % (self.trainset, self.batch_size, self.patch_sioze)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def forward(self, noisy_image):
        # assert noisy_image is range 0-1
        self.sampler(noisy_image)
        return self.sess.run(self.Y_test, feed_dict={self.X_test: noisy_image})

    def test(self):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        
        #test_files = glob('./data/test/{}/*.jpg'.format(self.testset))
        test_files = glob('./data/test/Set12/*.png')
        
        #print(test_files)
        print("Following files are loaded for testing...")
        for fname in test_files:
            print(fname)
        print()
        if self.load(self.ckpt_dir):
            print(" [*] Load weights SUCCESS...")
        else:
            print(" [!] Load weights FAILED...")

        res = open("./Results/{}.txt".format(self.sigma), "w")

        psnr_sum, psnr_sum_x = 0, 0
        print("[*] " + 'brightness level: ' + str(self.sigma) + " start testing...")
        for idx in xrange(len(test_files)):
        #self.sigma = 0.1
        #for idx in range(10):
            test_data = load_images(test_files[idx])
            #test_data = load_images(test_files[0])
            # test_data = np.array(test_data / 255.0, dtype=np.float32)
            noisy_image = add_noise(test_data, self.sigma)  # ndarray

            noisy_image = np.array(noisy_image / 255.0, dtype=np.float32)  # normalize the data to 0-1
            test_data = np.array(test_data / 255.0, dtype=np.float32)  # normalize the data to 0-1

            predicted_noise = self.forward(noisy_image)
            output_clean_image = noisy_image - predicted_noise
            
            groundtruth = np.clip(255 * test_data, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            predicted_noise = np.clip(255 * predicted_noise, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            psnr_x = cal_psnr(groundtruth, noisy_image)

            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            psnr_sum_x += psnr - psnr_x
            save_images(os.path.join(self.test_save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(self.test_save_dir, 'denoised%d.png' % idx), outputimage)
            save_images(os.path.join(self.test_save_dir, 'predicted%d.png' % idx), predicted_noise)
        avg_psnr = psnr_sum / len(test_files)
        avg_psnr_x = psnr_sum_x / len(test_files)
        res.write("{} {}".format(avg_psnr, avg_psnr_x))
        print("--- Average PSNR %.2f ---" % avg_psnr)
        res.close()

    def evaluate(self, iter_num, test_data):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in xrange(len(test_data)):
            noisy_image = add_noise(test_data[idx] / 255.0, 0.5)  # ndarray
            predicted_noise = self.forward(noisy_image)
            output_clean_image = noisy_image - predicted_noise
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(self.sample_dir, 'test%d_%d.png' % (idx, iter_num)),
                        groundtruth, noisyimage, outputimage)
        avg_psnr = psnr_sum / len(test_data)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
