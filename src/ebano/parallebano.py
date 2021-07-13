import gc
import sys

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import Conv2D
import tensorflow.keras.backend as K
# from scipy import sparse
import matplotlib as mpl
import matplotlib.pyplot as plt

from time import perf_counter


class Explainer:
    def __init__(self,
                 model,
                 n_classes,
                 preprocess_input_fn,
                 input_shape,
                 layers_to_analyze=None):
        self.model = model  # e.g. vgg16 instance
        self.n_classes = n_classes  # TODO: change this to dynamically allocate the n of classes depending on model
        self.preprocess_input_fn = preprocess_input_fn  # e.g. vgg16.preprocess_input
        self.input_shape = input_shape

        # TODO: Check alternatives (e.g., BoxBlur runs in linear time!)
        #   https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#module-PIL.ImageFilter
        self.perturb_filter = ImageFilter.GaussianBlur(radius=10)

        layer_indexes = self._get_conv_layer_indexes(self.model)
        if layers_to_analyze:
            if layers_to_analyze > layer_indexes.__len__():  # TODO: try just len(...)
                raise ValueError(f"Layers to analyze must not exceed the number of convolutional layers: {layer_indexes.__len__()}.")
        else:
            layers_to_analyze = int(np.log(layer_indexes.__len__()) / np.log(2))

        layer_indexes = layer_indexes[-layers_to_analyze:]
        self.layers = [self.model.layers[li].output for li in layer_indexes]

    @staticmethod
    def _get_conv_layer_indexes(model):
        layer_indexes = []
        i = 0
        for l in model.layers:
            layer_name = str.lower(l.get_config()["name"])
            if (isinstance(l, Conv2D)) | ("conv2d" in layer_name):
                layer_indexes.append(i)
            i = i + 1
        return layer_indexes

    @staticmethod
    def _pil_to_numpy(image):
        """Convert a PIL Image instance to a numpy array"""

        x = np.asarray(image, dtype='float32')

        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], x.shape[1], 1))
        elif len(x.shape) != 3:
            raise ValueError(f"Unsupported image shape: {(x.shape,)}")

        return x

    def preprocess_images(self, images):
        X = np.empty(shape=(len(images), *self.input_shape), dtype='float32')

        for i, img in enumerate(images):
            x = self._pil_to_numpy(img)

            if x.shape != self.input_shape:
                raise ValueError(f"Image must be of shape {self.input_shape}, instead received {x.shape}")

            X[i] = x

        X = self.preprocess_input_fn(X)

        return X

    def extract_hypercolumns(self, X, n_components=30):
        # Extract feature maps. Return a list of length == len(self.layers),
        # i.e., one feature map per convolutional layer to analyze.
        feature_maps = K.function([self.model.layers[0].input], self.layers)(X)

        # print("Analysis of feature maps")
        # for i, fm in enumerate(feature_maps):
        #     print(f"Feature map {i} for layer {self.layers[i]}")
        #     print("Shape", fm.shape)
        #     print(f"Sparsity measure (n zeros/total): {(fm == 0).sum()}/{fm.size}={(fm == 0).sum()/fm.size}")

        start = perf_counter()
        # Preallocate hypercolumn array
        batch_size = feature_maps[0].shape[0]
        image_size = self.input_shape[0:2]  # e.g. (224, 224)
        n_features_hc = sum([fm.shape[-1] for fm in feature_maps])
        hc = np.empty((batch_size, n_features_hc, *image_size), dtype='float32')

        # Extract hypercolumns
        for fm in feature_maps:  # fm.shape = (batch_size, x, y, n_filters) e.g., (8, 14, 14, 512)
            # TODO: you can remove transpose altogether and iterate over arbitrary dimension -- but you pay with mental sanity
            #  see https://stackoverflow.com/questions/1589706/iterating-over-arbitrary-dimension-of-numpy-array

            fm_reshaped = fm.transpose((0, 3, 2, 1))

            for ii, fm_img in enumerate(fm_reshaped):  # fm_img.shape = (n_filters, y, x)
                fi = 0
                for fmp in fm_img:  # fmp.shape = (y, x)
                    # TODO: if too slow, maybe try https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#cv2.resize
                    upscaled = np.array(Image.fromarray(fmp).resize(image_size, Image.BILINEAR))  # (image_size[0], image_size[1])
                    hc[ii, fi] = upscaled
                    fi += 1

        hc = hc.reshape((batch_size, n_features_hc, np.prod(image_size))).transpose((0, 2, 1))  # (batch_size, 224*224 = 50176, n_features_hc)
        end = perf_counter()
        print("-- extract hypercols: ", end-start)

        start = perf_counter()
        # Dimensionality reduction of feature maps
        # TODO: use TruncatedSVD with sparse matrices instead of PCA -- possible speedup
        #  see: https://stats.stackexchange.com/questions/199501/user-segmentation-by-clustering-with-sparse-data
        hc_pca = np.empty((batch_size, hc.shape[1], n_components), dtype='float32')
        pca = PCA(n_components=n_components, copy=False)
        for i in range(batch_size):
            print(f"performing PCA {i}")
            hc_pca[i] = pca.fit_transform(hc[i])  # fit_transform accepts (n_samples=n_pixels, n_components)

        ### TRUNCATEDSVD ###
        # hc_svd = np.empty((batch_size, hc.shape[1], n_components), dtype='float32')
        # svd = TruncatedSVD(n_components=n_components)
        # for i in range(batch_size):
        #     print(f"performing SVD {i}")
        #     hc_sparse = sparse.csr_matrix(hc[i])
        #     hc_svd[i] = svd.fit_transform(hc_sparse).todense()???  # fit_transform accepts (n_samples=n_pixels, n_components)

        del hc
        gc.collect()

        end = perf_counter()
        print("-- dimensionality reduction: ", end - start)

        return hc_pca  # (batch_size, n_pixels, n_components)

    def kmeans_cluster_hypercolumns(self, hypercolumns, min_features=2, max_features=5):
        RANDOM_STATE = 42
        MAX_ITER = 300

        # Each image has (max_features-min_features+1) feature maps, i.e., one
        # for each possible number of clusters. The best clustering will be
        # computed *later*, for now we need to save all possible maps.
        feature_maps = np.empty(
            (hypercolumns.shape[0], max_features-min_features+1, *self.input_shape[0:2]),
            dtype=np.uint8
        )  # (batch_size, max_features-min_features+1, x, y)

        for i in range(len(hypercolumns)):  # iterate over batch
            start = perf_counter()

            print(f"clustering starting for image id {i}")
            hc_n = normalize(hypercolumns[i], norm='l2', axis=1)

            for n_features in range(min_features, max_features+1):
                print(f"computing explanation for {n_features} n_features")
                model = KMeans(n_clusters=n_features, max_iter=MAX_ITER, random_state=RANDOM_STATE)
                feature_map = model.fit_predict(hc_n)
                feature_maps[i, n_features-min_features] = feature_map.reshape(*self.input_shape[0:2]).astype(np.uint8)

            end = perf_counter()
            print(f"-- clustering {i} elapsed:", end - start)

        feature_maps += 1  # start labels from 1 instead of 0

        return feature_maps

    def generate_masks(self, X, feature_maps, min_features=2):
        # Each image (first dim of X) has 2 masks (perturbed images) for k=2,
        # 3 masks for k=3, ..., 5 masks for k=5. So in total, (2+3+4+5 masks).
        # This can be generalized to any k.
        # Masks are stored in a contiguous manner inside X_masks.
        # Example of content of X_masks_map and X_masks_labels (if k is between 2 and 5):
        #    X_masks_map    = [0,0,1,1,1,2,2,2,2,3,3,3,3,3]
        #    X_masks_labels = [1,2,1,2,3,1,2,3,4,1,2,3,4,5]
        # with len(X_masks_map) == len(X_masks_labels) == n_masks
        start = perf_counter()

        n_masks = 0
        X_masks_map = []
        X_masks_labels = []
        for i in range(feature_maps.shape[1]):
            n_masks += min_features + i
            X_masks_map.extend([i] * (min_features + i))
            X_masks_labels.extend([x for x in range(1, min_features + i + 1)])

        X_masks = np.empty((len(X), n_masks, *self.input_shape[0:2]), dtype=np.uint8)  # (batch_size, total num of masks, x, y)

        # 255: pixel to perturb, 0: pixel to keep unchanged
        for i in range(X_masks.shape[0]):  # Iterate over images
            for mask_i in range(X_masks.shape[1]):  # Iterate over masks
                X_masks[i, mask_i] = (feature_maps[i, X_masks_map[mask_i]] == X_masks_labels[mask_i]) * 255

        end = perf_counter()
        print(f"- generate all masks:", end - start)

        return X_masks, X_masks_map, X_masks_labels

    def perturb(self, images, X_masks):
        start = perf_counter()
        images_perturbed = []

        # i-th elem indicates the index of the original image inside the array
        # `images` corresponding images_perturbed[i]
        images_perturbed_origin_map = np.empty((len(images) * X_masks.shape[1]))

        for i in range(len(images)):
            im_full_perturbed = images[i].filter(self.perturb_filter)

            for mask_i in range(X_masks.shape[1]):
                im_perturbed = images[i].copy()

                # Create and apply mask
                im_mask = Image.fromarray(X_masks[i, mask_i], mode="L")
                im_perturbed.paste(im_full_perturbed, mask=im_mask)

                images_perturbed.append(im_perturbed)

                images_perturbed_origin_map[i*X_masks.shape[1]+mask_i] = i

        end = perf_counter()
        print(f"- blur images:", end - start)

        return images_perturbed, images_perturbed_origin_map

    def perturb_format_sublists(self, images, X_masks):
        start = perf_counter()
        images_perturbed = [[] for _ in range(len(images))]

        for i in range(len(images)):
            im_full_perturbed = images[i].filter(self.perturb_filter)

            for mask_i in range(X_masks.shape[1]):
                im_perturbed = images[i].copy()

                # Create and apply mask
                im_mask = Image.fromarray(X_masks[i, mask_i], mode="L")
                im_perturbed.paste(im_full_perturbed, mask=im_mask)

                images_perturbed[i].append(im_perturbed)

        end = perf_counter()
        print(f"- blur images:", end - start)

        return images_perturbed

    @staticmethod
    def softsign(x):
        return x / (1 + abs(x))

    @staticmethod
    def relu(x):
        if x < 0:
            return 0.
        return x

    @staticmethod
    def inf_float(x):
        if x == np.inf:
            return sys.float_info.max
        elif x == -np.inf:
            return -sys.float_info.max
        return x

    def explain_numeric(self, preds_perturbed, preds_original, cois):
        """
        preds_perturbed (batch size, num masks per image, n_classes)
        preds_original (batch size, n_classes)
        cois (batch size)
        """
        start = perf_counter()

        # One set of scores per perturbed image (batch size * num masks per image scores)
        nPIR = np.empty((*preds_perturbed.shape[0:2], ), dtype=float)
        nPIRP = np.empty((*preds_perturbed.shape[0:2], ), dtype=float)

        for i in range(preds_perturbed.shape[0]):  # Iterate over batch
            ci = cois[i]  # class of interest

            for f_i in range(preds_perturbed.shape[1]):  # Iterate over masks (interpretable features)
                nPIR_f = np.empty((self.n_classes, ))  # vector with nPIR_f computed for each possible class

                for c in range(self.n_classes):
                    p_o_c = preds_original[i, c]
                    p_f_c = preds_perturbed[i, f_i, c]  # prob of image to be labeled as ci when f_i is perturbed
                    alpha = self.inf_float((1 - p_o_c / p_f_c))
                    beta = self.inf_float((1 - p_f_c / p_o_c))
                    PIR_f_c = p_f_c * beta - p_o_c * alpha
                    nPIR_f_c = self.softsign(PIR_f_c)
                    nPIR_f[c] = nPIR_f_c

                nPIR[i, f_i] = nPIR_f[ci]  # get the nPIR for the class of interest

                # Compute nPIRP
                xi_vector = preds_original[i] * np.absolute(nPIR_f)
                xi_vector_no_ci = np.delete(xi_vector, ci)
                xi_ci = xi_vector[ci]
                xi_no_ci = np.sum(xi_vector_no_ci)
                a = self.inf_float((1 - xi_ci / xi_no_ci))
                b = self.inf_float((1 - xi_no_ci / xi_ci))
                PIRP_f_ci = xi_no_ci * b - xi_ci * a
                nPIRP_f_ci = self.softsign(PIRP_f_ci)

                nPIRP[i, f_i] = nPIRP_f_ci

        end = perf_counter()
        print(f"- numeric explanation: {end - start}")

        # TODO: compute informativeness

        return nPIR, nPIRP

    @staticmethod
    def explain_visual_old(image, X_masks, nPIR, nPIRP, cmap='RdYlGn'):
        nPIR_heatmap = np.sum(X_masks.astype(np.bool) * nPIR.reshape(-1, 1, 1), axis=0)
        nPIRP_heatmap = np.sum(X_masks.astype(np.bool) * nPIRP.reshape(-1, 1, 1), axis=0)

        norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
        colors = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)).to_rgba(nPIR_heatmap)
        mask = Image.fromarray((colors * 255).astype(np.uint8), mode="RGBA")
        blended = Image.blend(image.copy().convert("RGBA"), mask, alpha=.85)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig.tight_layout()
        ax.imshow(blended)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        cb_ax = fig.add_axes([1, 0.12, 0.05, 0.8])
        cb = mpl.colorbar.ColorbarBase(
            cb_ax,
            cmap=plt.get_cmap(cmap),
            norm=mpl.colors.Normalize(vmin=-1, vmax=1),
            orientation='vertical'
        )
        cb.set_label("nPIR")

        plt.show()

    def fit_batch(self, images, cois, min_features=2, max_features=5):
        """Fit explainer to batch of images.

        images (list of PIL images) of length n
        cois: np array of shape (n,)
        """

        MIN_FEATURES = 2
        MAX_FEATURES = 5

        X = self.preprocess_images(images)
        hypercolumns = self.extract_hypercolumns(X)
        feature_maps = self.kmeans_cluster_hypercolumns(hypercolumns)
        X_masks, X_masks_map, X_masks_labels = self.generate_masks(X, feature_maps)
        images_perturbed, images_perturbed_origin_map = self.perturb(images, X_masks)

        print(X_masks_map)
        print(images_perturbed_origin_map)
        print(images_perturbed)

        start = perf_counter()
        X_perturbed = self.preprocess_images(images_perturbed)
        end = perf_counter()
        print(f"- preprocess perturbed: ", end-start)

        start = perf_counter()
        preds_original = self.model.predict(X)
        end = perf_counter()
        print(f"- predict originals:", end-start)

        start = perf_counter()
        preds_perturbed = self.model.predict(X_perturbed)
        preds_perturbed = preds_perturbed.reshape((len(images), len(X_masks_map), -1))
        end = perf_counter()
        print(f"- predict perturbed: {end-start}")

        nPIR, nPIRP = self.explain_numeric(preds_perturbed, preds_original, cois)

        for i in range(len(images)):
            print(f"# image {i}")
            for k_i in np.unique(X_masks_map):
                print(f"* k={k_i+MIN_FEATURES}")
                mask = X_masks_map == k_i
                print(nPIR[i][mask])
                self.explain_visual_old(images[i], X_masks[i][mask], nPIR[i][mask], nPIRP[i][mask])

                # if k_i == 1:
                #     return images[i], X_masks[i][mask], nPIR[i][mask], nPIRP[i][mask]

        # debug. gc to remove
        del X, hypercolumns, feature_maps, X_masks, images_perturbed, X_perturbed, preds_original, preds_perturbed
        gc.collect()
