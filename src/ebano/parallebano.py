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

from utils import Profiling


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

        with Profiling("extract hypercolumns"):
            # Preallocate hypercolumn array
            batch_size = feature_maps[0].shape[0]
            image_size = self.input_shape[0:2]  # e.g. (224, 224)
            n_features_hc = sum([fm.shape[-1] for fm in feature_maps])
            hc = np.empty((batch_size, n_features_hc, *image_size), dtype='float32')

            # Extract hypercolumns
            for fm in feature_maps:  # fm.shape = (batch_size, x, y, n_filters) e.g., (8, 14, 14, 512)
                # TODO: you can remove transpose altogether and iterate over arbitrary dimension -- but you pay with mental sanity
                #  see https://stackoverflow.com/questions/1589706/iterating-over-arbitrary-dimension-of-numpy-array

                fm_reshaped = fm.transpose((0, 3, 1, 2))

                for ii, fm_img in enumerate(fm_reshaped):  # fm_img.shape = (n_filters, x, y)
                    fi = 0
                    for fmp in fm_img:  # fmp.shape = (y, x)
                        upscaled = np.array(Image.fromarray(fmp).resize(image_size, Image.BILINEAR))  # (image_size[0], image_size[1])
                        hc[ii, fi] = upscaled
                        fi += 1

            hc = hc.reshape((batch_size, n_features_hc, np.prod(image_size))).transpose((0, 2, 1))  # (batch_size, 224*224 = 50176, n_features_hc)

        with Profiling("dimensionality reduction"):
            # Dimensionality reduction of feature maps
            # TODO: use TruncatedSVD with sparse matrices instead of PCA -- possible speedup
            #  see: https://stats.stackexchange.com/questions/199501/user-segmentation-by-clustering-with-sparse-data
            PCA_TYPE = None  # "batch", "fit-first", None
            if PCA_TYPE == "fit-first":
                hc_pca = np.empty((batch_size, hc.shape[1], n_components), dtype='float32')
                pca = PCA(n_components=n_components, copy=False)
                print(f"fitting PCA")
                hc_pca[0] = pca.fit_transform(hc[0])
                for i in range(1, batch_size):
                    print(f"performing PCA {i}")
                    hc_pca[i] = pca.transform(hc[i])
            elif PCA_TYPE == "batch":
                # hc_pca = np.empty((batch_size * hc.shape[1], n_components), dtype='float32')
                pca = PCA(n_components=n_components, copy=False)
                hc_pca = pca.fit_transform(hc.reshape((-1, n_features_hc)))
                hc_pca = hc_pca.reshape((batch_size, hc.shape[1], n_components))
            else:
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
            with Profiling(f"clustering for image id {i}"):
                hc_n = normalize(hypercolumns[i], norm='l2', axis=1)

                for n_features in range(min_features, max_features+1):
                    print(f"computing explanation for {n_features} n_features")
                    model = KMeans(n_clusters=n_features, max_iter=MAX_ITER, random_state=RANDOM_STATE)
                    feature_map = model.fit_predict(hc_n)
                    feature_maps[i, n_features-min_features] = feature_map.reshape(*self.input_shape[0:2]).astype(np.uint8)

        feature_maps += 1  # start labels from 1 instead of 0

        return feature_maps

    def minibatchkmeans_cluster_hypercolumns(self, hypercolumns, min_features=2, max_features=5):
        RANDOM_STATE = 42
        MAX_ITER = 300
        BATCH_SIZE = 100

        # Each image has (max_features-min_features+1) feature maps, i.e., one
        # for each possible number of clusters. The best clustering will be
        # computed *later*, for now we need to save all possible maps.
        feature_maps = np.empty(
            (hypercolumns.shape[0], max_features-min_features+1, *self.input_shape[0:2]),
            dtype=np.uint8
        )  # (batch_size, max_features-min_features+1, x, y)

        for i in range(len(hypercolumns)):  # iterate over batch
            with Profiling(f"clustering for image id {i}"):
                hc_n = normalize(hypercolumns[i], norm='l2', axis=1)

                for n_features in range(min_features, max_features+1):
                    print(f"computing explanation for {n_features} n_features")
                    model = MiniBatchKMeans(n_clusters=n_features, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
                    feature_map = model.fit_predict(hc_n)
                    feature_maps[i, n_features-min_features] = feature_map.reshape(*self.input_shape[0:2]).astype(np.uint8)

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

        with Profiling("generate masks"):
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

        return X_masks, X_masks_map, X_masks_labels

    def perturb(self, images, X_masks):
        with Profiling("blur images"):
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

        return images_perturbed, images_perturbed_origin_map

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

    def explain_numeric(self, preds_perturbed, preds_original, X_masks_map, cois):
        """
        preds_perturbed (batch size, num masks per image, n_classes)
        preds_original (batch size, n_classes)
        cois (batch size)
        """
        with Profiling("numeric explanation"):
            # One set of scores per perturbed image (batch size * num masks per image scores)
            nPIR = np.empty(preds_perturbed.shape[0:2], dtype=float)
            nPIRP = np.empty(preds_perturbed.shape[0:2], dtype=float)

            # Informativeness vector: (batch_size, number of k's) e.g. (32, 4) if k=[2,5]
            info = np.empty((len(preds_perturbed), len(np.unique(X_masks_map))), dtype='float32')

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

                # informativeness: contrast of the nPIR values for each explanation
                for k_i in np.unique(X_masks_map):
                    mask = X_masks_map == k_i
                    nPIR_i_k = nPIR[i][mask]
                    info[i, k_i] = max(nPIR_i_k) - min(nPIR_i_k)

            print(info)

            # compute index of best explanation for each image, based on informativeness
            best = np.empty(len(preds_perturbed), dtype=np.uint8)
            for i in range(len(preds_perturbed)):
                best[i] = np.argmax(info[i])

        return nPIR, nPIRP, best

    @staticmethod
    def OLD_explain_visual(image, X_masks, nPIR, nPIRP, cmap='RdYlGn'):
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

    @staticmethod
    def _find_centroid(init_x, X, n_iter=10):
        new_x = init_x
        f_X = X

        for i in range(n_iter):
            dist = np.linalg.norm(f_X - new_x, axis=1)
            if f_X[dist < np.mean(dist)].__len__() > 0:
                f_X = f_X[dist < np.mean(dist)]
            else:
                break

            new_x = np.percentile(f_X, 50, interpolation="nearest", axis=0).astype(np.int64)

        return new_x

    def explain_visual(self, image, coi, X_masks, nPIR, nPIRP, cmap_features='Set3', cmap_index='RdYlGn'):
        nPIR_heatmap = np.sum(X_masks.astype(np.bool) * nPIR.reshape(-1, 1, 1), axis=0)
        feature_map_mask = np.zeros((X_masks.shape[1:3]), dtype=np.uint8)
        centroids = []

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.tight_layout()

        norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

        # Feature map
        ax[0].set_title("Feature map")
        ax[0].imshow(image)

        for f_i in range(len(X_masks)):  # iterate over feature IDs
            # feature map mask può semplicemente contenere degli integers tipo 1, 2, 3 etc
            # poi basta fare imshow con una colormap discreta che assegna automaticamente
            # un colore diverso ad ogni integer

            m = X_masks[f_i] > 0

            x_coors, y_coors = np.where(m)
            y_coor, x_coor = np.percentile(x_coors, 50).astype(int), np.percentile(y_coors, 50).astype(int)

            coors = np.array([x_coors, y_coors]).transpose(1, 0)
            coor = np.array([(x_coor, y_coor)])

            coor = self._find_centroid(coor, coors)
            y_coor, x_coor = coor[0], coor[1]

            centroids.append((x_coor, y_coor))

            feature_map_mask += ((m).astype(np.uint8)) * (f_i + 1)

        ax[0].imshow(feature_map_mask, cmap=cmap_features, alpha=0.8, interpolation='nearest')
        for f_i, (x_coor, y_coor) in enumerate(centroids):
            txt = ax[0].text(x_coor, y_coor, f"{f_i}", color="black", ha='center', va='center', fontsize=20)
            # path_effects=[mpl.patheffects.withStroke(linewidth=3, foreground="w")])

        ax[0].grid(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # nPIR/nPIRP chart
        x = np.arange(len(nPIR), dtype=np.uint8)
        width = 0.4
        ax[1].set_title(f"nPIR and nPIRP for COI: {coi}")
        rects1 = ax[1].bar(x - width / 2, nPIR, width, label="nPIR")
        rects2 = ax[1].bar(x + width / 2, nPIRP, width, label="nPIRP")
        ax[1].set_ylim([-1.2, 1.2])
        ax[1].set_xticks(x)
        ax[1].legend()
        ax[1].axhline(y=0, color="black", lw=1)
        ax[1].bar_label(rects1, fmt="%.2f", padding=3)
        ax[1].bar_label(rects2, fmt="%.2f", padding=3)
        ax[1].set_xlabel("Feature ID")

        # nPIR heatmap
        ax[2].set_title("nPIR heatmap")
        ax[2].imshow(image)
        ax[2].contourf(np.arange(224), np.arange(224), nPIR_heatmap, cmap=cmap_index, norm=norm, alpha=0.6, antialiased=True)
        ax[2].grid(False)
        ax[2].set_xticks([])
        ax[2].set_yticks([])

        colors = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_index)).to_rgba(nPIR_heatmap)
        cb_ax = fig.add_axes([1, 0.12, 0.02, 0.8])
        cb = mpl.colorbar.ColorbarBase(
            cb_ax,
            cmap=plt.get_cmap(cmap_index),
            norm=norm,
            orientation='vertical'
        )
        cb.set_label("nPIR")

        plt.show()

    def fit_batch(self, images, cois, min_features=2, max_features=5):
        """Fit explainer to batch of images.

        images (list of PIL images) of length n
        cois: np array of shape (n,)
        """

        X = self.preprocess_images(images)
        hypercolumns = self.extract_hypercolumns(X)
        feature_maps = self.kmeans_cluster_hypercolumns(hypercolumns, min_features=min_features, max_features=max_features)
        X_masks, X_masks_map, X_masks_labels = self.generate_masks(X, feature_maps)
        images_perturbed, images_perturbed_origin_map = self.perturb(images, X_masks)

        with Profiling("preprocess perturbed"):
            X_perturbed = self.preprocess_images(images_perturbed)

        with Profiling("predict originals"):
            preds_original = self.model.predict(X, batch_size=len(X))

        with Profiling("predict perturbed"):
            preds_perturbed = self.model.predict(X_perturbed, batch_size=len(X_perturbed))
            preds_perturbed = preds_perturbed.reshape((len(images), len(X_masks_map), -1))

        nPIR, nPIRP, best = self.explain_numeric(preds_perturbed, preds_original, X_masks_map, cois)

        print(best)

        for i in range(len(images)):
            print(f"# image {i}")
            for k_i in np.unique(X_masks_map):
                mask = X_masks_map == k_i
                self.explain_visual(images[i], cois[i], X_masks[i][mask], nPIR[i][mask], nPIRP[i][mask])

        # debug. gc to remove
        del X, hypercolumns, feature_maps, X_masks, images_perturbed, X_perturbed, preds_original, preds_perturbed
        gc.collect()