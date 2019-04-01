
from __future__ import absolute_import
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from nipy.algorithms.segmentation._segmentation import _ve_step, _interaction_energy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
NITERS = 10
NGB_SIZE = 26
BETA = 0.1
r_1 = 1
r_2 = 1
r_3 = 1

nonzero = lambda x: np.maximum(x, 1e-50)
log = lambda x: np.log(nonzero(x))


class Segmentation(object):

    def __init__(self, data, mask=None, mu=None, sigma=None,
                 ppm=None, prior=None, U=None,
                 ngb_size=NGB_SIZE, beta=BETA):
        """
        Class for multichannel Markov random field image segmentation
        using the variational EM algorithm. For details regarding the
        underlying algorithm, see:
        Roche et al, 2011. On the convergence of EM-like algorithms
        for image segmentation using Markov random fields. Medical
        Image Analysis (DOI: 10.1016/j.media.2011.05.002).
        Parameters
        ----------
        data : array-like
          Input image array
        mask : array-like or tuple of array
          Input mask to restrict the segmentation
        beta : float
          Markov regularization parameter
        mu : array-like
          Initial class-specific means
        sigma : array-like
          Initial class-specific variances
        """
        data = data.squeeze()
        if not len(data.shape) in (3, 4):
            raise ValueError('Invalid input image')
        if len(data.shape) == 3:
            nchannels = 1
            space_shape = data.shape
        else:
            nchannels = data.shape[-1]
            space_shape = data.shape[0:-1]

        self.nchannels = nchannels
        if mask is None:
            mask = np.ones(space_shape, dtype=bool)
        X, Y, Z = np.where(mask)
        XYZ = np.zeros((X.shape[0], 3), dtype='intp')
        XYZ[:, 0], XYZ[:, 1], XYZ[:, 2] = X, Y, Z
        self.XYZ = XYZ

        self.mask = mask
        self.data = data[mask]
        if nchannels == 1:
            self.data = np.reshape(self.data, (self.data.shape[0], 1))

        # By default, the ppm is initialized as a collection of
        # uniform distributions
        if ppm is None:
            nclasses = len(mu)
            self.ppm = np.zeros(list(space_shape) + [nclasses])
            self.ppm[mask] = 1. / nclasses
            self.is_ppm = False
            self.mu = np.array(mu, dtype='double').reshape(\
                (nclasses, nchannels))
            self.sigma = np.array(sigma, dtype='double').reshape(\
                (nclasses, nchannels, nchannels))
        elif mu is None:
            nclasses = ppm.shape[-1]
            self.ppm = np.asarray(ppm)
            self.is_ppm = True
            self.mu = np.zeros((nclasses, nchannels))
            self.sigma = np.zeros((nclasses, nchannels, nchannels))
        else:
            raise ValueError('missing information')
        self.nclasses = nclasses

        if prior is not None:
            self.prior = np.asarray(prior)[self.mask].reshape(\
                [self.data.shape[0], nclasses])
        else:
            self.prior = None

        self.ngb_size = int(ngb_size)
        self.set_markov_prior(beta, U=U)

    def set_markov_prior(self, beta, U=None):
        if U is not None:
            self.U = np.asarray(U).copy()
        else:  # Potts model
            U = r_1*np.ones((self.nclasses, self.nclasses))
            U[_diag_indices(self.nclasses)] = 0
            self.U = U
        self.beta = float(beta)*r_2

    def vm_step(self, freeze=()):
        classes = list(range(self.nclasses))
        for i in freeze:
            classes.remove(i)

        for i in classes:
            P = self.ppm[..., i][self.mask].ravel()
            Z = nonzero(P.sum())
            tmp = self.data.T * P.T
            mu = tmp.sum(1) / Z
            mu_ = mu.reshape((len(mu), 1))
            sigma = np.dot(tmp, self.data) / Z - np.dot(mu_, mu_.T)
            self.mu[i] = mu
            self.sigma[i] = sigma

    def log_external_field(self):
        """
        Compute the logarithm of the external field, where the
        external field is defined as the likelihood times the
        first-order component of the prior.
        """
        lef = np.zeros([self.data.shape[0], self.nclasses])

        for i in range(self.nclasses):
            centered_data = self.data - self.mu[i]
            if self.nchannels == 1:
                inv_sigma = 1. / nonzero(self.sigma[i])
                norm_factor = np.sqrt(inv_sigma.squeeze())
            else:
                inv_sigma = np.linalg.inv(self.sigma[i])
                norm_factor = 1. / np.sqrt(\
                    nonzero(np.linalg.det(self.sigma[i])))
            maha_dist = np.sum(centered_data * np.dot(inv_sigma,
                                                      centered_data.T).T, 1)
            lef[:, i] = -.5 * maha_dist/r_3
            lef[:, i] += log(norm_factor)

        if self.prior is not None:
            lef += log(self.prior)

        return lef

    def normalized_external_field(self):
        f = self.log_external_field().T
        f -= np.max(f, 0)
        np.exp(f, f)
        f /= f.sum(0)
        return f.T

    def ve_step(self):
        nef = self.normalized_external_field()
        if self.beta == 0:
            self.ppm[self.mask] = np.reshape(\
                nef, self.ppm[self.mask].shape)
        else:
            self.ppm = _ve_step(self.ppm, nef, self.XYZ,
                                self.U, self.ngb_size, self.beta)

    def run(self, niters=NITERS, freeze=()):
        if self.is_ppm:
            self.vm_step(freeze=freeze)
        for i in range(niters):
            self.ve_step()
            self.vm_step(freeze=freeze)
        self.is_ppm = True

    def map(self):
        """
        Return the maximum a posterior label map
        """
        return map_from_ppm(self.ppm, self.mask)

    def free_energy(self, ppm=None):
        """
        Compute the free energy defined as:
        F(q, theta) = int q(x) log q(x)/p(x,y/theta) dx
        associated with input parameters mu,
        sigma and beta (up to an ignored constant).
        """
        if ppm is None:
            ppm = self.ppm
        q = ppm[self.mask]
        # Entropy term
        lef = self.log_external_field()
        f1 = np.sum(q * (log(q) - lef))
        # Interaction term
        if self.beta > 0.0:
            f2 = self.beta * _interaction_energy(ppm, self.XYZ,
                                                 self.U, self.ngb_size)
        else:
            f2 = 0.0
        return f1 + f2


def _diag_indices(n, ndim=2):
    # diag_indices function present in numpy 1.4 and later.  This for
    # compatibility with numpy < 1.4
    idx = np.arange(n)
    return (idx,) * ndim


def moment_matching(dat, mu, sigma, glob_mu, glob_sigma):
    """
    Moment matching strategy for parameter initialization to feed a
    segmentation algorithm.
    Parameters
    ----------
    data: array
      Image data.
    mu : array
      Template class-specific intensity means
    sigma : array
      Template class-specific intensity variances
    glob_mu : float
      Template global intensity mean
    glob_sigma : float
      Template global intensity variance
    Returns
    -------
    dat_mu: array
      Guess of class-specific intensity means
    dat_sigma: array
      Guess of class-specific intensity variances
    """
    dat_glob_mu = float(np.mean(dat))
    dat_glob_sigma = float(np.var(dat))
    a = np.sqrt(dat_glob_sigma / glob_sigma)
    b = dat_glob_mu - a * glob_mu
    dat_mu = a * mu + b
    dat_sigma = (a ** 2) * sigma
    return dat_mu, dat_sigma


def map_from_ppm(ppm, mask=None):
    x = np.zeros(ppm.shape[0:-1], dtype='uint8')
    if mask is None:
        mask = ppm == 0
    x[mask] = ppm[mask].argmax(-1) + 1
    return x


def binarize_ppm(q):
    """
    Assume input ppm is masked (ndim==2)
    """
    bin_q = np.zeros(q.shape)
    bin_q[:q.shape[0], np.argmax(q, axis=1)] = 1
    return bin_q

def regularize(M, box):
    size = M.shape
    M.astype(int)
    N = np.zeros(size)
    N.astype(int)
    for i in range(box+1, size[0]-box-1):
        for j in range(box+1, size[1]-box-1):
            nbhd = M[i-box:i+box,j-box:j+box]
            nbhd = np.asarray(nbhd).reshape(-1)
            nbhd = nbhd.tolist()
            N[i,j] = np.bincount(nbhd).argmax()
    return N

#face = misc.imread('face.png')
face = misc.imread('circle.png')
[w,h] = face.shape
ret = np.empty((w, h, 3), dtype=np.uint8)
ret[:, :, 0] = face
ret[:, :, 1] = face
ret[:, :, 2] = face
face = ret
im = Segmentation(face, mu=[100,200], sigma=[1000,1000])
im.run()
b = im.map()
c_1 = b[:,:,1]
c_1.astype(int)
show_images([c_1*50, face])

