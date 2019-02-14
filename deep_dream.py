import numpy as np
from scipy.ndimage.filters import gaussian_filter
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile
import cv2
import glob


def main():
    # Step 1 - download google's pre-trained neural network
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = os.path.join(dir, 'data')
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)
    if not os.path.exists(local_zip_file):
        # Download
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

    model_fn = 'tensorflow_inception_graph.pb'

    # Step 2 - Creating Tensorflow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')  # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input': t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type ==
              'Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

# HELPER FUNCTIONS

    # Helper functions for TF Graph visualization
    # pylint: disable=unused-variable
    def recursive_dream(layer_tensor, image, repeat=3, scale=0.7, blend=0.2, iteration_n=10):
        if repeat > 0:
            sigma = 0.5
            img_blur = gaussian_filter(image, (sigma, sigma, 0.0))

            h0 = img_blur.shape[0]
            w0 = img_blur.shape[1]
            h1 = int(scale * h0)
            w1 = int(scale * w0)
            img_downscaled = cv2.resize(img_blur, (w1, h1))

            img_dream = recursive_dream(layer_tensor, img_downscaled,
                                        repeat - 1, scale, blend, iteration_n)
            img_upscaled = cv2.resize(img_dream, (w0, h0))

            image = blend * image + (1.0 - blend) * img_upscaled
            image = np.clip(image, 0, 255)

        return render_deepdream(layer_tensor, image, iter_n=20)

    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()  # pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>" % size
        return strip_def

    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()  # pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0] != '^' else '^'+rename_func(s[1:])
        return res_def

    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()

    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0" % layer)

    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input: img})
            # normalizing the gradient, so the same step size should work
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
        showarray(visstd(img))

    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))

        def wrap(f):
            out = f(*placeholders)

            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]
    resize = tffunc(np.float32, np.int32)(resize)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz), sz):
            for x in range(0, max(w-sz//2, sz), sz):
                sub = img_shift[y:y+sz, x:x+sz]
                g = sess.run(t_grad, {t_input: sub})
                grad[y:y+sz, x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    # CHALLENGE - Write a function that outputs a deep dream video
    def render_deepdreamvideo(t_obj, iteration_n=10):
        def key_func(x):
            return os.path.split(x)[-1]

        # Split video into frames
        print("Splitting Video...")
        split_video()

        images = [cv2.imread(file) for file in sorted(glob.glob(
            "E:/Python_Projects/deep_dream_challenge/input/*.png"), key=key_func)]

        # Apply Gradient ascent for each image
        print("Applying Gradient Ascent...")
        counter = 0
        for img in images:
            img = np.float32(img)
            img = recursive_dream(t_obj, img, repeat=3,
                                  scale=0.7, blend=0.2, iteration_n=iteration_n)/255.0
            img = np.uint8(np.clip(img, 0, 1)*255)
            # Store image in output
            plt.imshow(img/255.0)
            plt.axis('off')
            plt.subplots_adjust(bottom=0, top=1, left=0, right=1)
            plt.savefig(os.path.join(dir, 'output', 'frame%05d.png') %
                        counter, bbox_inches='tight', pad_inches=0, transparent=True, frameon=False)
            print("Frame %d Generated Successfully..." % counter)
            counter += 1

        print("Saving...")
        save()

    def split_video():
        # Playing video from file:
        cap = cv2.VideoCapture("E:/Python_Projects/deep_dream_challenge/test_vid.mp4")

        try:
            if not os.path.exists('input'):
                os.makedirs('input')
        except OSError:
            print('Error: Creating directory of input')

        currentFrame = 0
        ret, frame = cap.read()
        while ret:
            # Saves image of the current frame in jpg file
            strn = str(currentFrame).zfill(5)
            name = 'E:/Python_Projects/deep_dream_challenge/input/frame' + strn + '.png'
            print('Creating...' + name)
            cv2.imwrite(name, frame)
            ret, frame = cap.read()
            # To stop duplicate images
            currentFrame += 1

        # When everything done, release the capture
        cap.release()

    def save():
        try:
            if not os.path.exists('output'):
                os.makedirs('output')
        except OSError:
            print('Error: Creating directory of output')
        # Assumes ffmpeg installed in E:/ffmpeg
        os.system("E:/ffmpeg/bin/ffmpeg.exe -r 24 -start_number 0 -i E:/Python_Projects/deep_dream_challenge/output/frame%05d.png -c:v libx264 -vf E:/Python_Projects/deep_dream_challenge/output.mp4")

    def render_deepdream(t_obj, img0=img_noise,
                         iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))

            # this will usually be like 3 or 4 octaves
            # Step 5 output deep dream image via matplotlib
            # showarray(img/255.0)
            return img

    # Step 3 - Pick a layer to enhance our image
    # layer_names = ['conv2d0', 'conv2d1', 'conv2d2',
    #            'mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e',
    #            'mixed5a', 'mixed5b']
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139  # picking some feature channel to visualize

    render_deepdreamvideo(tf.square(T('conv2d2')))
    # Step 4 - Apply gradient ascent to that layer

    print("Done.")


if __name__ == '__main__':
    main()
