import sys

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import Conv2D
import tensorflow.keras.backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt
import faiss

import utils
from utils import Profiling, pil_to_ndarray, ndarray_to_pil


class Explainer:
    def __init__(self,
                 model,
                 n_classes,
                 input_shape,
                 layers_to_analyze=None):
        self.model = model
        self.n_classes = n_classes
        self.input_shape = input_shape

        self.perturb_filter = ImageFilter.GaussianBlur(radius=10)

        layer_idx = self._get_conv_layer_indexes(self.model)
        if layers_to_analyze:
            if layers_to_analyze > len(layer_idx):
                raise ValueError(
                    f"Layers to analyze must not exceed the number of convolutional layers: {len(layer_idx)}."
                )
        else:
            layers_to_analyze = int(np.log(len(layer_idx)) / np.log(2))

        layer_indexes = layer_idx[-layers_to_analyze:]
        self.layers = [self.model.layers[li].output for li in layer_indexes]

    @staticmethod
    def _get_conv_layer_indexes(model):
        layer_indexes = []

        for i, l in enumerate(model.layers):
            layer_name = str.lower(l.get_config()["name"])
            if (isinstance(l, Conv2D)) | ("conv2d" in layer_name):
                layer_indexes.append(i)

        return layer_indexes

    def get_hypercolumns(self, X, features=30, reduction="pca"):
        with Profiling("Extract layer activations"):
            # Extract activations for each layer to analyze
            activations = K.function([self.model.layers[0].input], self.layers)(X)  # list of len(self.layers)

        # Preallocate array of hypercolumns
        n_filters = sum([a.shape[-1] for a in activations])
        hc = np.empty((len(X), n_filters, *self.input_shape), dtype='float32')  # (batch size, num features, filter x, filter y)

        # Extract hypercolumns
        for img_i in range(len(X)):
            with Profiling(f"Image {img_i} hypercolumns extraction"):
                f_i = 0
                for layer_i, layer_activations in enumerate(activations):  # layer_activations.shape = (batch size, filter x, filter y, num filters) e.g., (8, 14, 14, 512)
                    layer_activations_img = layer_activations[img_i].transpose((2, 0, 1))  # (num filters, filter x, filter y)
                    for filter_activations_img in layer_activations_img:
                        scaled = np.array(Image.fromarray(filter_activations_img).resize(self.input_shape, Image.BILINEAR))
                        hc[img_i, f_i] = scaled
                        f_i += 1

        hc = hc.reshape((len(X), n_filters, np.prod(self.input_shape))).transpose((0, 2, 1))  # (batch_size, x*y (224*224), num filters)

        return hc

    def reduce_hypercolumns(self, hc, features=30, reduction="pca"):
        with Profiling("Reduce dimensionality of hypercolumns"):
            if reduction == "none" or reduction is None:
                print("No reduction to do.")
                return hc

            # Here, each image is treated as a "dataset" with np.prod(input_shape)
            # samples, each of which has `n_filters` features. We want to reduce
            # the number of these features to `features`.
            hc_r = np.empty((len(hc), np.prod(self.input_shape), features), dtype='float32')  # (batch size, x*y, features)

            if reduction == "pca":
                for i, hc_i in enumerate(hc):
                    hc_r[i] = self._hypercolumn_reduction_pca(hc_i, n_components=features)
            elif reduction == "tsvd":
                for i, hc_i in enumerate(hc):
                    hc_r[i] = self._hypercolumn_reduction_tsvd(hc_i, n_components=features)
            elif reduction == "sampletsvd":
                for i, hc_i in enumerate(hc):
                    hc_r[i] = self._hypercolumn_reduction_tsvd(hc_i, n_components=features, fit_size=0.01)
            else:
                raise ValueError(f"Unsupported dimensionality reduction: {reduction}")

            del hc

        with Profiling("Normalize (L2) hypercolumns"):
            for i in range(len(hc_r)):
                hc_r[i] = normalize(hc_r[i], norm='l2', axis=1)  # L2 normalization over features

        return hc_r

    @staticmethod
    def _hypercolumn_reduction_pca(hc, n_components):
        pca = PCA(n_components, copy=False)
        hc_pca = pca.fit_transform(hc)
        return hc_pca

    @staticmethod
    def _hypercolumn_reduction_tsvd(hc, n_components, fit_size=1.):
        tsvd = TruncatedSVD(n_components)
        if fit_size < 1:
            idx = np.random.choice(len(hc), size=int(len(hc) * fit_size), replace=False)
            tsvd.fit(hc[idx])
            hc_tsvd = tsvd.transform(hc)
        else:
            hc_tsvd = tsvd.fit_transform(hc)
        return hc_tsvd

    def cluster_hypercolumns(self,
                             hc,
                             min_features=2,
                             max_features=5,
                             clustering="minibatchkmeans",
                             seed=None,
                             use_gpu=False,
                             **kwargs):
        # Each image has (max_features-min_features+1) feature maps, i.e., one
        # for each possible number of clusters. The best clustering will be
        # computed *later*, for now we need to save all possible maps.
        feature_maps = np.empty((hc.shape[0], max_features-min_features+1, *self.input_shape),
                                dtype=np.uint8)  # (batch_size, max_features-min_features+1, x, y)

        if clustering == "faisskmeans":
            d = hc.shape[-1]  # number of features
            for i in range(len(hc)):
                with Profiling(f"Building index for image {i}"):
                    # hc_r shape = (batch size, x*y, features)
                    if use_gpu:
                        res = faiss.StandardGpuResources()  # use a single GPU
                        config = faiss.GpuIndexFlatConfig()
                        index = faiss.GpuIndexFlatL2(res, d, config)
                    else:
                        index = faiss.IndexFlatL2(d)
                    # index.add(hc[i])
                    # print(index.is_trained)
                    # print(index.ntotal)

                for n_clusters in range(min_features, max_features+1):
                    with Profiling(f"Compute explanation for {n_clusters} clusters"):
                        kmeans = faiss.Clustering(d, n_clusters)
                        kmeans.verbose = bool(1)
                        kmeans.niter = kwargs["niter"]

                        kmeans.train(hc[i], index)
                        _, I = index.search(hc[i], 1)
                        # feature_map = I.squeeze()
                        feature_maps[i, n_clusters-min_features] = I.reshape(*self.input_shape).astype(np.uint8)

        else:
            for n_clusters in range(min_features, max_features+1):
                with Profiling(f"Compute explanation for {n_clusters} clusters"):
                    if clustering == "minibatchkmeans":
                        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, max_iter=kwargs["cluster_max_iter"], batch_size=kwargs["cluster_batch_size"])
                    elif clustering == "kmeans":
                        model = KMeans(n_clusters=n_clusters, random_state=seed, max_iter=kwargs["cluster_max_iter"])
                    else:
                        raise ValueError(f"Unsupported clustering algorithm: {clustering}")

                    for i in range(len(hc)):
                        print(f"Compute explanation for image {i}")
                        feature_map = model.fit_predict(hc[i])
                        feature_maps[i, n_clusters-min_features] = feature_map.reshape(*self.input_shape).astype(np.uint8)

        feature_maps += 1  # start labels from 1 instead of 0
        return feature_maps

    def generate_perturbation_masks(self, X, feature_maps, min_features=2):
        # Each image (first dim of X) has 2 masks (perturbed images) for k=2,
        # 3 masks for k=3, ..., 5 masks for k=5. So in total, (2+3+4+5 masks).
        # This can be generalized to any k.
        # Masks are stored in a contiguous manner inside X_masks.
        # Example of content of X_masks_map and X_masks_labels (if k is between 2 and 5):
        #    X_masks_map    = [0,0,1,1,1,2,2,2,2,3,3,3,3,3]
        #    X_masks_labels = [1,2,1,2,3,1,2,3,4,1,2,3,4,5]
        # with len(X_masks_map) == len(X_masks_labels) == n_masks

        with Profiling("Generate perturbation masks"):
            n_masks = 0
            X_masks_map = []
            X_masks_labels = []
            for i in range(feature_maps.shape[1]):
                n_masks += min_features + i
                X_masks_map.extend([i] * (min_features + i))
                X_masks_labels.extend([x for x in range(1, min_features + i + 1)])

            X_masks = np.empty((len(X), n_masks, *self.input_shape), dtype=np.uint8)  # (batch_size, total num of masks, x, y)

            # 255: pixel to perturb, 0: pixel to keep unchanged
            for i in range(X_masks.shape[0]):  # Iterate over images
                for mask_i in range(X_masks.shape[1]):  # Iterate over masks
                    X_masks[i, mask_i] = (feature_maps[i, X_masks_map[mask_i]] == X_masks_labels[mask_i]) * 255

        return X_masks, X_masks_map

    def perturb(self, X, X_masks, inverse=False):
        """
        Perturbs original images, given an array of perturbation masks.
        If inverse is True, perturbs each feature individually. Otherwise,
        perturb the whole image except the masked feature.
        """

        with Profiling("Perturb images"):
            X_perturbed = np.empty((len(X) * X_masks.shape[1], *self.input_shape, 3))

            # The i-th element indicates the index of the original image (located inside
            # the array X) corresponding to the i-th element of X_perturbed
            X_perturbed_origin_map = np.empty((len(X) * X_masks.shape[1]))

            if inverse:
                X_masks_inverse = np.copy(X_masks)
                X_masks_inverse[X_masks == 255] = 0
                X_masks_inverse[X_masks == 0] = 255
                X_masks = X_masks_inverse

            for i, x in enumerate(X):
                x_pil = ndarray_to_pil(x)
                x_perturbed = x_pil.filter(self.perturb_filter)  # Fully perturbed image

                for mask_i in range(X_masks.shape[1]):
                    x_perturbed_i = x_pil.copy()

                    # Create and apply mask
                    x_mask = Image.fromarray(X_masks[i, mask_i], mode="L")
                    x_perturbed_i.paste(x_perturbed, mask=x_mask)
                    x_perturbed_i = pil_to_ndarray(x_perturbed_i)

                    X_perturbed[i * X_masks.shape[1] + mask_i] = x_perturbed_i
                    X_perturbed_origin_map[i * X_masks.shape[1] + mask_i] = i

        return X_perturbed, X_perturbed_origin_map

    def explain_numeric(self, preds_perturbed, preds_original, X_masks_map, cois):
        """
        preds_perturbed (batch size, num masks per image, n_classes)
        preds_original (batch size, n_classes)
        cois (batch size)
        """
        with Profiling("Explain numerically"):
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

            # compute index of best explanation for each image, based on informativeness
            best = np.empty(len(preds_original), dtype=np.uint8)
            for i in range(len(preds_original)):
                best[i] = np.argmax(info[i])

        return nPIR, nPIRP, best

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
        ax[2].contourf(np.arange(224), np.arange(224), nPIR_heatmap, cmap=cmap_index, norm=norm, alpha=0.6,
                       antialiased=True)
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

    def fit_batch(self,
                  X,
                  cois,
                  y=None,
                  preprocess_input_fn=None,
                  hypercolumn_features=30,
                  hypercolumn_reduction="pca",
                  clustering="minibatchkmeans",
                  min_features=2,
                  max_features=5,
                  display_plots=True,
                  return_indices=False,
                  return_results=False,
                  use_gpu=False,
                  seed=None,
                  **kwargs):
        """Fit explainer to a batch of images.

        Args:
            X: Array of (unprocessed) images to explain.
            cois: Classes of interest.
            y: Ground truth, useful for debugging purposes. Ignored otherwise.
            preprocess_input_fn: Preprocessing function to apply to the array.
            hypercolumn_features:
            hypercolumn_reduction:
            clustering:
            min_features:
            max_features:
            display_plots: If True, create and display visual explanations.
                Otherwise, only compute indices.
            return_indices:
            return_results:
            use_gpu:
            seed: Seed for reproducibility.
        """

        if preprocess_input_fn:
            X_preprocessed = preprocess_input_fn(np.copy(X))  # avoid destructive action
        else:
            X_preprocessed = X

        hc = self.get_hypercolumns(X_preprocessed, features=hypercolumn_features, reduction=hypercolumn_reduction)
        hc_r = self.reduce_hypercolumns(hc, features=hypercolumn_features, reduction=hypercolumn_reduction)
        del hc

        feature_maps = self.cluster_hypercolumns(hc_r,
                                                 min_features=min_features,
                                                 max_features=max_features,
                                                 clustering=clustering,
                                                 seed=seed,
                                                 use_gpu=use_gpu,
                                                 **kwargs)

        X_masks, X_masks_map = self.generate_perturbation_masks(X, feature_maps, min_features=min_features)
        del feature_maps

        X_perturbed, X_perturbed_origin_map = self.perturb(X, X_masks)
        X_perturbed = preprocess_input_fn(X_perturbed)

        with Profiling("Predict original images"):
            preds_original = self.model.predict(X_preprocessed, batch_size=len(X), verbose=1)

        with Profiling("Predict perturbed images"):
            preds_perturbed = self.model.predict(X_perturbed, batch_size=len(X), verbose=1)
            preds_perturbed = preds_perturbed.reshape((len(X), len(X_masks_map), -1))

        nPIR, nPIRP, best = self.explain_numeric(preds_perturbed, preds_original, X_masks_map, cois)

        if display_plots:
            with Profiling("Explain visually"):
                for i in range(len(X)):
                    print(f"# image {i}, best explanation k={best[i]+min_features}")
                    print(f"# coi (pred): {cois[i]} ground truth: {y[i]} correctly classified: {cois[i] == y[i]}")
                    k_best = best[i]
                    mask = X_masks_map == k_best
                    image_i = utils.ndarray_to_pil(X[i])
                    self.explain_visual(image_i, cois[i], X_masks[i][mask], nPIR[i][mask], nPIRP[i][mask])

        if return_indices:
            nPIR_best = []
            nPIRP_best = []
            for i in range(len(X)):
                best_mask = X_masks_map == best[i]
                nPIR_best.append(nPIR[i][best_mask])
                nPIRP_best.append(nPIRP[i][best_mask])

            return nPIR_best, nPIRP_best

        if return_results:
            # Fetch data for analysis
            # Returns (list of dicts):
            #   X_original (ndarray (x, y, n_channels)): original image
            #   truth (int): ground truth class
            #   preds (ndarray (n_classes)): predictions of original image (output of softmax)
            #   best_n_features (int): number of clusters of best explanation
            #   X_masks (ndarray (n_masks, x, y)): masks of features associated to best explanation
            #   nPIR_best, nPIRP_best: indices of best explanation
            results = []

            for i in range(len(X)):
                best_mask = X_masks_map == best[i]
                results.append({
                    "X_original": X[i],
                    "truth": y[i],
                    "preds": preds_original[i],
                    "best_n_features": best[i]+min_features,
                    "X_masks": X_masks[i][best_mask],
                    "nPIR_best": nPIR[i][best_mask],
                    "nPIRP_best": nPIRP[i][best_mask],
                })

            return results

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
