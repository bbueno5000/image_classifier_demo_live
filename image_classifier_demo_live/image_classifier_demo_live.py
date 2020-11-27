"""
NOTE: When dealing with high-dimensional inputs such as images, 
it is impractical to connect neurons to all neurons in the previous volume. 
Instead, we will connect each neuron to only a local region of the input volume. 
The spatial extent of this connectivity is a hyperparameter called the receptive field 
of the neuron (equivalently this is the filter size).
"""
import datetime
import math
import matplotlib.pyplot as pyplot
import numpy
import sklearn
import tensorflow
import time

class ImageClassifier:
    """
    DOCSTRING
    """
    def __init__(self):
        self.filter_size1 = 5
        self.num_filters1 = 16
        self.filter_size2 = 5
        self.num_filters2 = 36
        self.fc_size = 128
        self.data = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(
            'data/MNIST/', one_hot=True)
        print('Size of:')
        print('-Training-set:{}'.format(len(self.data.train.labels)))
        print('-Test-set:{}'.format(len(self.data.test.labels)))
        print('-Validation-set:{}'.format(len(self.data.validation.labels)))
        self.data.test.cls = numpy.argmax(self.data.test.labels, axis=1)
        img_size = 28
        img_size_flat = img_size * img_size
        img_shape = (img_size, img_size)
        num_channels = 1
        num_classes = 10

    def __call__(self):
        images = self.data.test.images[0:9]
        cls_true = self.data.test.cls[0:9]
        self.plot_images(images=images, cls_true=cls_true)
        x = tensorflow.placeholder(tensorflow.float32, shape=[None, img_size_flat], name='x')
        x_image = tensorflow.reshape(x, [-1, img_size, img_size, num_channels])
        y_true = tensorflow.placeholder(tensorflow.float32, shape=[None, 10], name='y_true')
        y_true_cls = tensorflow.argmax(y_true, dimension=1)
        layer_conv1, weights_conv1 = self.new_conv_layer(
            input=x_image,
            num_input_channels=num_channels,
            filter_size=self.filter_size1,
            num_filters=self.num_filters1,
            use_pooling=True)
        layer_conv1
        layer_conv2, weights_conv2 = self.new_conv_layer(
            input=layer_conv1,
            num_input_channels=self.num_filters1,
            filter_size=self.filter_size2,
            num_filters=self.num_filters2,
            use_pooling=True)
        layer_conv2
        layer_flat, num_features = self.flatten_layer(layer_conv2)
        layer_flat
        num_features
        layer_fc1 = self.new_fc_layer(
            input=layer_flat, num_inputs=num_features, num_outputs=self.fc_size, use_relu=True)
        layer_fc1
        layer_fc2 = self.new_fc_layer(
            input=layer_fc1, num_inputs=self.fc_size, num_outputs=num_classes, use_relu=False)
        layer_fc2
        y_pred = tensorflow.nn.softmax(layer_fc2)
        y_pred_cls = tensorflow.argmax(y_pred, dimension=1)
        cross_entropy = tensorflow.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
        cost = tensorflow.reduce_mean(cross_entropy)
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        correct_prediction = tensorflow.equal(y_pred_cls, y_true_cls)
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
        session = tensorflow.Session()
        session.run(tensorflow.global_variables_initializer())
        train_batch_size = 64
        total_iterations = 0
        test_batch_size = 256
        self.print_test_accuracy()
        self.optimize(num_iterations=1)
        self.print_test_accuracy()
        self.optimize(num_iterations=99)
        self.print_test_accuracy(show_example_errors=True)
        self.optimize(num_iterations=900)
        self.print_test_accuracy(show_example_errors=True)
        self.optimize(num_iterations=9000)
        self.print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)
        image1 = self.data.test.images[0]
        self.plot_image(image1)
        image2 = self.data.test.images[13]
        self.plot_image(image2)
        self.plot_conv_weights(weights=weights_conv1)
        self.plot_conv_layer(layer=layer_conv1, image=image1)
        self.plot_conv_layer(layer=layer_conv1, image=image2)
        self.plot_conv_weights(weights=weights_conv2, input_channel=0)
        self.plot_conv_weights(weights=weights_conv2, input_channel=1)
        self.plot_conv_layer(layer=layer_conv2, image=image1)
        self.plot_conv_layer(layer=layer_conv2, image=image2)
        session.close()

    def flatten_layer(self, layer):
        """
        DOCSTRING
        """
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tensorflow.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    def plot_images(self, images, cls_true, cls_pred=None):
        """
        DOCSTRING
        """
        assert len(images) == len(cls_true) == 9
        fig, axes = pyplot.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].reshape(img_shape), cmap='binary')
            if cls_pred is None:
                xlabel = 'True: {}'.format(cls_true[i])
            else:
                xlabel = 'True: {}, Pred: {}'.format(cls_true[i], cls_pred[i])
            ax.set_xlabel(xlabel)
            ax.set_xticks([])
            ax.set_yticks([])
        pyplot.show()

    def new_biases(self, length):
        """
        DOCSTRING
        """
        return tensorflow.Variable(tensorflow.constant(0.05, shape=[length]))

    def new_conv_layer(self, input, num_input_channels, filter_size, num_filters, use_pooling=True):
        """
        Parameters:
            input: The previous layer
            num_input_channels: number of channels in previous layer
            filter_size: width and height of each filter
            num_filters: number of filters
            use_pooling: use 2x2 max-pooling
        """
        shape = [filter_size, filter_size, num_input_channels, num_filters]
        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=num_filters)
        layer = tensorflow.nn.conv2d(
            input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
        layer += biases
        if use_pooling:
            layer = tensorflow.nn.max_pool(
                value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        layer = tensorflow.nn.relu(layer)
        return layer, weights

    def new_fc_layer(self, input, num_inputs, num_outputs, use_relu=True):
        """
        Parameters:
            input: the previous layer
            num_inputs: number inputs from previous layer
            num_outputs: number of outputs
            use_relu: use rectified linear unit (ReLU)
        """
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)
        layer = tensorflow.matmul(input, weights) + biases
        if use_relu:
            layer = tensorflow.nn.relu(layer)
        return layer

    def new_weights(self, shape):
        """
        DOCSTRING
        """
        return tensorflow.Variable(tensorflow.truncated_normal(shape, stddev=0.05))

    def optimize(self, num_iterations):
        """
        DOCSTRING
        """
        global total_iterations
        start_time = time.time()
        for i in range(total_iterations, total_iterations + num_iterations):
            x_batch, y_true_batch = self.data.train.next_batch(train_batch_size)
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
            if i % 100 == 0:
                acc = session.run(accuracy, feed_dict=feed_dict_train)
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                print(msg.format(i + 1, acc))
        total_iterations += num_iterations
        end_time = time.time()
        time_dif = end_time - start_time
        print('Time usage: ' + str(datetime.timedelta(seconds=int(round(time_dif)))))

    def plot_confusion_matrix(self, cls_pred):
        """
        This is called from print_test_accuracy() below.

        Parameters:
            cls_pred: an array of the predicted class-number for all images in the test-set
        """
        cls_true = self.data.test.cls
        cm = sklearn.metrics.confusion_matrix(y_true=cls_true, y_pred=cls_pred)
        print(cm)
        pyplot.matshow(cm)
        pyplot.colorbar()
        tick_marks = numpy.arange(num_classes)
        pyplot.xticks(tick_marks, range(num_classes))
        pyplot.yticks(tick_marks, range(num_classes))
        pyplot.xlabel('Predicted')
        pyplot.ylabel('True')
        pyplot.show()

    def plot_conv_layer(self, layer, image):
        """
        Assume layer is a TensorFlow op that outputs a 4-dim tensor
        which is the output of a convolutional layer,
        e.g. layer_conv1 or layer_conv2.
        """
        feed_dict = {x: [image]}
        values = session.run(layer, feed_dict=feed_dict)
        num_filters = values.shape[3]
        num_grids = math.ceil(math.sqrt(num_filters))
        fig, axes = pyplot.subplots(num_grids, num_grids)
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                img = values[0, :, :, i]
                ax.imshow(img, interpolation='nearest', cmap='binary')
            ax.set_xticks([])
            ax.set_yticks([])
        pyplot.show()

    def plot_conv_weights(self, weights, input_channel=0):
        """
        Assume weights are TensorFlow ops for 4-dim variables
        e.g. weights_conv1 or weights_conv2.
        """
        w = session.run(weights)
        w_min = numpy.min(w)
        w_max = numpy.max(w)
        num_filters = w.shape[3]
        num_grids = math.ceil(math.sqrt(num_filters))
        fig, axes = pyplot.subplots(num_grids, num_grids)
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                img = w[:, :, input_channel, i]
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
        pyplot.show()

    def plot_example_errors(self, cls_pred, correct):
        """
        This function is called from print_test_accuracy() below.
    
        Parameters:
            cls_pred: an array of the predicted class-number for all images in the test-set
            correct: a boolean array whether the predicted class
                equal to the true class for each image in the test-set
        """
        incorrect = correct == False
        images = self.data.test.images[incorrect]
        cls_pred = cls_pred[incorrect]
        cls_true = self.data.test.cls[incorrect]
        self.plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

    def plot_image(self, image):
        """
        DOCSTRING
        """
        pyplot.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
        pyplot.show()

    def print_test_accuracy(self, show_example_errors=False, show_confusion_matrix=False):
        """
        DOCSTRING
        """
        num_test = len(self.data.test.images)
        cls_pred = numpy.zeros(shape=num_test, dtype=numpy.int)
        i = 0
        while i < num_test:
            j = min(i + test_batch_size, num_test)
            images = self.data.test.images[i:j,:]
            labels = self.data.test.labels[i:j,:]
            feed_dict = {x: images, y_true: labels}
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
            i = j
        cls_true = self.data.test.cls
        correct = cls_true == cls_pred
        correct_sum = correct.sum()
        acc = float(correct_sum) / num_test
        msg = 'Accuracy on Test-Set: {0:.1%} ({1} / {2})'
        print(msg.format(acc, correct_sum, num_test))
        if show_example_errors:
            print('Example errors:')
            self.plot_example_errors(cls_pred=cls_pred, correct=correct)
        if show_confusion_matrix:
            print('Confusion Matrix:')
            self.plot_confusion_matrix(cls_pred=cls_pred)

if __name__ == '__main__':
    image_classifier = ImageClassifier()
    image_classifier()
