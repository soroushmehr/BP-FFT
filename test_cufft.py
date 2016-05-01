print "start..."
import ipdb
import numpy
np = numpy
import theano
import theano.tensor as T

from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.utils import flatten

from theano.sandbox.cuda.fftconv import cufft as rcufft
from theano.sandbox.cuda.cuda_fft import cufft
np.random.seed(seed=123)

from features.base import get_filterbanks

frame_size = 200
fb_coeff = get_filterbanks(nfilt=64, nfft=frame_size, samplerate=16000)
fb_coeff = fb_coeff.astype(theano.config.floatX)
fb_coeff = np.transpose(fb_coeff)
fb_coeff = theano.shared(fb_coeff)


# calculate real fft 2-D matrix (batch, 1-D signals) from complex fft.
# frame_size should be even. Or use //2+1
def realfft(y):
    y_fft_in = T.zeros(shape=(y.shape[0], y.shape[1], 2), dtype=theano.config.floatX)
    y_fft_in = T.set_subtensor(y_fft_in[:, :, 0], y)
    return cufft(y_fft_in)[:, :frame_size/2+1, :]


# abs/magnitude of a fft response
# y: realfft() complex output
def magnitude(y):
    return T.sqrt(y[:, :, 0]**2 + y[:, :, 1]**2)


def spectral_magnitude_log_distance_error(y_true, y_pred):
    Y_true = realfft(y_true)
    Y_pred = realfft(y_pred)
    mag_Y_true = magnitude(Y_true)
    mag_Y_pred = magnitude(Y_pred)
    mel_true = T.dot(mag_Y_true, fb_coeff)+1.0e-8
    mel_pred = T.dot(mag_Y_pred, fb_coeff)+1.0e-8
    return T.mean(T.sqr(T.log(mel_pred)-T.log(mel_true)).sum(axis=1))

lr = 3e-4
debug = 1

batch_size = mn_batch_size = 2

init_W = InitCell('rand', low=-0.5, high=0.5)
init_U = InitCell('ortho')
init_b = InitCell('zeros')
init_b_sig = InitCell('const', mean=0.6)

x = T.tensor3('x', dtype=theano.config.floatX)

x.tag.test_value = np.random.rand(2, batch_size, frame_size).astype(theano.config.floatX)

epsilonij = 0.0001
x_1 = FullyConnectedLayer(name='x_1',
                          parent=['x'],
                          parent_dim=[frame_size],
                          nout=150,
                          unit='relu',
                          init_W=init_W,
                          init_b=init_b)

theta_mu = FullyConnectedLayer(name='theta_mu',
                               parent=['x_1'],
                               parent_dim=[150],
                               nout=200,
                               unit='linear',
                               init_W=init_W,
                               init_b=init_b)

nodes = [x_1, theta_mu]

for node in nodes:
    node.initialize()

params = flatten([node.get_params().values() for node in nodes])


x_shape = x.shape
x_in = x.reshape((x_shape[0]*x_shape[1], -1))

x_1_in = x_1.fprop([x_in])
theta_mu_in = theta_mu.fprop([x_1_in])

recon = 0.5*(x_in-theta_mu_in)**2
recon_term = recon.mean()
# TODO: what should be the reconstructed signal? theta_mu or sample?

spec_recon = spectral_magnitude_log_distance_error(x_in, theta_mu_in)
spec_recon_term = spec_recon.mean()
spec_recon_term.name = 'spec_recon_term'
cost = recon_term + spec_recon_term
recon_term.name = 'recon_term'
cost.name = 'cost'


"""
spec_recon_fn = theano.function(inputs=[x],
                                outputs=[spec_recon],)
#on_unused_input=True)  # batch_size*seq_len
print "spec_recon_fn(x.tag.test_value)[0].shape:"
print spec_recon_fn(x.tag.test_value)[0].shape
"""

"""
# compare the complex fft output with real fft output
rcufft_fn = theano.function(inputs=[x], outputs=[rcufft(x_in)])
print "rcufft_fn(x.tag.test_value)[0].shape"
print rcufft_fn(x.tag.test_value)[0].shape

cufft_fn = theano.function(inputs=[x], outputs=[cufft(x_fft_in)[:, :frame_size/2+1, :]])
print "cufft_fn(x.tag.test_value)[0].shape"
print cufft_fn(x.tag.test_value)[0].shape
"""

xin_fn = theano.function(inputs=[x], outputs=[x_in, theta_mu_in])
x_in_test, theta_mu_in_test = xin_fn(x.tag.test_value)
print "x_in_test, theta_mu_in_test = xin_fn(x.tag.test_value)"
print "x shape: ", x.tag.test_value.shape
print "x_in shape:", x_in_test.shape
print "theta_mu_in shape:", theta_mu_in_test.shape
realfft_fn = theano.function(inputs=[x_in],
                             outputs=[realfft(x_in),
                                      magnitude(realfft(x_in)),
                                      T.log(magnitude(realfft(x_in))),
                                      T.log(magnitude(realfft(x_in))+1.0E-8),
                                      T.log(T.dot(magnitude(realfft(x_in)), fb_coeff)+1.0E-8)])
#print realfft_fn(x.tag.test_value)[0].shape
#print realfft_fn(x.tag.test_value)[1].shape
print "!a,b,c,d,e=realfft_fn()"
a, b, c, d, e = realfft_fn(x_in_test)
ap, bp, cp, dp, ep = realfft_fn(theta_mu_in_test)
ipdb.set_trace()

orig_params = np.array(params[0].get_value())
used_cost = cost
#used_cost = spec_recon_term
#used_cost = recon_term  # works fine.

# works, not perfectly but OK because of float32
#theano.gradient.verify_grad(fun=cufft,
#                            pt=[np.random.rand(1, 64, 2).astype(theano.config.floatX)],
#                            #n_tests=3,
#                            rng=np.random,
#                            #eps=1e-3,
#                            #abs_tol=0.01,
#                            #out_type='complex64'
#                            )


# calculates grad wrt to just weights in first layer of autoencoder, params[0]
# or W_x__x_1, with shape (200, 150)
def grad_check(i, j):
    """
    params[0].set_value(orig_params)
    #print "--------------------------------------------------"
    #print "  ->", params[0].get_value()[0, :5]
    """
    cost_specrec_par_fn = theano.function(inputs=[x],
                                          outputs=[used_cost],
                                          mode=theano.compile.MonitorMode(
                                                post_func=theano.compile.monitormode.detect_nan)\
                                                .excluding('local_elemwise_fusion', 'inplace'))

    # point1 = cost(params_old[0], x)
    # for x.tag.test_value look at line 74
    point1 = cost_specrec_par_fn(x.tag.test_value)
    print ">>>>FORWARD-PASSSS: ", point1
    #ipdb.set_trace()
    grad_cost_specrec_par_fn = theano.function(inputs=[x],
                                               outputs=[theano.grad(used_cost, params[0])],
                                               mode=theano.compile.MonitorMode(
                                                        post_func=theano.compile.monitormode.detect_nan)\
                                                        .excluding('local_elemwise_fusion', 'inplace'))

    # th_grad_cost_specrec_par = grad(cost wrt params[0])(x)
    th_grad_cost_specrec_par = np.array(grad_cost_specrec_par_fn(x.tag.test_value)[0])
    print ">>>>successss!?!", th_grad_cost_specrec_par[i, j]
    """
    ############## +
    # adding a small value to [i, j] element of params[0]
    # params[0][i, j] += epsilon
    params[0].set_value(orig_params)
    par0_val = np.array(params[0].get_value())
    par0_val[i, j] += epsilonij
    params[0].set_value(par0_val)

    #print " -->", params[0].get_value()[0, :5]
    cost_specrec_par_fn2 = theano.function(inputs=[x],
                                           outputs=[used_cost],
                                           mode=theano.compile.MonitorMode(post_func=detect_nan))

    # point2 = cost(params_new[0], x)
    # for x.tag.test_value look at line 74
    point2 = cost_specrec_par_fn2(x.tag.test_value)

    grad_cost_specrec_par_fn2 = theano.function(inputs=[x],
                                                outputs=[theano.grad(used_cost, params[0])],
                                                mode=theano.compile.MonitorMode(post_func=detect_nan))
    # th_grad_cost_specrec_par2 = grad(cost wrt params_new[0])(x)
    th_grad_cost_specrec_par2 = np.array(grad_cost_specrec_par_fn2(x.tag.test_value)[0])

    ############## -
    # subtracting a small value from [i, j] element of params[0]
    # params[0][i, j] -= epsilon
    params[0].set_value(orig_params)
    par0_val = np.array(params[0].get_value())
    par0_val[i, j] -= epsilonij
    params[0].set_value(par0_val)
    #print "--->", params[0].get_value()[0, :5]

    cost_specrec_par_fnm1 = theano.function(inputs=[x],
                                            outputs=[used_cost],
                                            mode=theano.compile.MonitorMode(post_func=detect_nan))

    # pointm1 = cost(params_new[0], x)
    # for x.tag.test_value look at line 74
    pointm1 = cost_specrec_par_fnm1(x.tag.test_value)

    # Should be compared to theano grad.
    num_grad = (point2[0] - pointm1[0])/(epsilonij*2.)

    grad_cost_specrec_par_fnm1 = theano.function(inputs=[x],
                                                 outputs=[theano.grad(used_cost, params[0])],
                                                 mode=theano.compile.MonitorMode(post_func=detect_nan))
    # th_grad_cost_specrec_parm1 = grad(cost wrt params_new[0])(x)
    th_grad_cost_specrec_parm1 = np.array(grad_cost_specrec_par_fnm1(x.tag.test_value)[0])

    print "####################", i, j, "###############"
    print "eps: ", epsilonij, "\tcost: ", used_cost.name
    print "num_grad : ", num_grad
    print "the_grad0: ", th_grad_cost_specrec_par[i, j]
    print "the_grad2: ", th_grad_cost_specrec_par2[i, j]
    print "th_gradm1: ", th_grad_cost_specrec_parm1[i, j]
    print "ratio num_grad/the_grad: ~1: ", num_grad/th_grad_cost_specrec_par[i, j]
    print "diff num_grad-the_grad:  ~0: ", num_grad - th_grad_cost_specrec_par[i, j]
    relErr = np.absolute(num_grad - th_grad_cost_specrec_par[i, j])/(np.absolute(th_grad_cost_specrec_par[i, j]) + np.absolute(num_grad))
    errHere = ""
    if relErr > 1e-2:
        errHere = "\t\t\t<-----"
    print "relErr |num_grad-the_grad|/(|the_grad|+|num_grad|)  =< 1e-2: ", relErr, errHere
    """

grad_check(0, 0)
grad_check(0, 1)
grad_check(0, 2)
grad_check(0, 3)
grad_check(0, 147)
grad_check(0, 148)
grad_check(0, 149)

grad_check(1, 1)
grad_check(1, 2)
grad_check(1, 3)
grad_check(1, 147)
grad_check(1, 148)
grad_check(1, 149)

grad_check(2, 1)
grad_check(2, 2)
grad_check(2, 3)
grad_check(2, 147)
grad_check(2, 148)
grad_check(2, 149)

grad_check(3, 1)
grad_check(3, 2)
grad_check(3, 3)
grad_check(3, 147)
grad_check(3, 148)
grad_check(3, 149)

grad_check(197, 1)
grad_check(197, 2)
grad_check(197, 3)
grad_check(197, 147)
grad_check(197, 148)
grad_check(197, 149)

grad_check(198, 1)
grad_check(198, 2)
grad_check(198, 3)
grad_check(198, 147)
grad_check(198, 148)
grad_check(198, 149)

grad_check(199, 1)
grad_check(199, 2)
grad_check(199, 3)
grad_check(199, 147)
grad_check(199, 148)
grad_check(199, 149)

grad_check(0, 0)
grad_check(1, 0)
grad_check(2, 0)
grad_check(3, 0)
grad_check(197, 0)
grad_check(198, 0)
grad_check(199, 0)
#ipdb.set_trace()
