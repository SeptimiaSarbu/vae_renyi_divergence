import os
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
#from tensorflow_probability.distributions import MultivariateNormalDiag
#from tensorflow_probability.distributions import MultivariateNormalFullCovariance
tfd = tfp.distributions

from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.special import logsumexp

import matplotlib.pyplot as plt

import pdb

from tensorflow.contrib.opt import ScipyOptimizerInterface

plt.rcParams.update({'font.size': 22})
#plt.rc('text', usetex=True)
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##########################################################################
D = 40 #dimension of the input data, x
K = 10 #dimension of the latent space, z
#q = 1.0 - 0.1

batch_size = 15#100
N_z = 100
N_z_IS = 1 * 1000
Nq = 100

dtype_var = tf.float64

W = tf.get_variable(name='W_enc', initializer=tf.truncated_normal(shape=[D,K],mean=5.01,stddev=0.01,dtype=dtype_var))
g = tf.get_variable(name='g_enc', initializer=tf.truncated_normal(shape=[K],mean=5.01,stddev=0.01,dtype=dtype_var))
log_var = tf.get_variable(name='log_var_enc', initializer=tf.truncated_normal(shape=[K],mean=5.01,stddev=0.01,dtype=dtype_var))
#the variance has to be positive, so we train log(variance), to allow for negative values

Q = tf.get_variable(name='q1_star', initializer=np.float64(1.0-1e-6), trainable=True, dtype=dtype_var)

x1 = tf.placeholder(dtype=dtype_var, shape=[None, D])
q1 = tf.placeholder(dtype=dtype_var, shape=[Nq])

q1_star = tf.placeholder(dtype=dtype_var)

def compute_qELBO(q_min, ratio_np):
    #pdb.set_trace()
    op_node_qELBO_loss_np = (1.0 - q_min) * ratio_np

    ratio_qELBO_loss_np = logsumexp(op_node_qELBO_loss_np, axis=0)  # will have dimensions=[batch_size]

    S_qELBO_loss_np = np.exp(-np.log(np.float64(N_z_IS)) + ratio_qELBO_loss_np)  # we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate

    qELBO_loss_np = (S_qELBO_loss_np - 1.0) / (1.0 - q_min)
    mean_qELBO_loss_np = np.mean(qELBO_loss_np)  # take the mean over the batch size

    return mean_qELBO_loss_np

def root_qELBO_no_exp(var, ratio_np1, eq_term):
    #pdb.set_trace()
    q_min = var
    F1 = (1.0 - q_min) * ratio_np1 # see my notebook for the notation of F1 and F2;
    # ratio_np1=means 1 sample from the batch; for each sample in the batch, we compute a q_opt and then average over the q

    F2 = 100*logsumexp(F1)  # will have dimensions=1
    # we multiply here with 100, to increase the size of the operands, to make the optimization process easier, i.e. increase the accuracy of the solution

    LHS_term = 100*np.log((1.0 - q_min) * eq_term + 1) + np.log(np.float64(N_z_IS))

    return (LHS_term - F2)**2

def root_qELBO(var, ratio_np, eq_term):
    q_min = var
    op_node_qELBO_loss_np = (1.0 - q_min) * ratio_np

    ratio_qELBO_loss_np = logsumexp(op_node_qELBO_loss_np, axis=0)  # will have dimensions=[batch_size]

    S_qELBO_loss_np = np.exp(-np.log(float(N_z_IS)) + ratio_qELBO_loss_np)  # we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate

    qELBO_loss_np = (S_qELBO_loss_np - 1.0)/( 1.0 - q_min)
    mean_qELBO_loss_np = np.mean(qELBO_loss_np)  # take the mean over the batch size

    return (mean_qELBO_loss_np-eq_term)**2

def compute_qCUBO(q_min, L_np):
    op_node_L_np = (1.0 - q_min) * L_np

    S_qCUBO_np = np.exp(op_node_L_np)

    #pdb.set_trace()
    mean_S_qCUBO_np = np.mean(S_qCUBO_np)  # take the mean over the batch size

    mean_qCUBO_np = (mean_S_qCUBO_np - 1.0)/( 2.0 * (1.0 - q_min))

    return mean_qCUBO_np
#
# def root_qCUBO(var, L_np, eq_term):
#     q_min = var
#     op_node_L_np = (1.0 - q_min) * L_np
#
#     S_qCUBO_np = np.exp(op_node_L_np)
#
#     #pdb.set_trace()
#     mean_S_qCUBO_np = np.mean(S_qCUBO_np)  # take the mean over the batch size
#
#     mean_qCUBO_np = (mean_S_qCUBO_np - 1.0)/( 2.0 * (1.0 - q_min))
#
#     return (mean_qCUBO_np-eq_term)**2

def plot_funcs2(mvn_px_np, q, q_star3_before):

    x = mvn_px_np.rvs(batch_size)
    log_px = mvn_px_np.logpdf(x)

    y_log_px = np.mean(log_px) * np.ones(shape=(Nq,))

    feed_dict = {x1: x, q1: q, q1_star: q_star3_before}

    mcubo, L_np, melbo, ratio_np = sess.run([mean_CUBO, L, mean_ELBO, ratio], feed_dict=feed_dict)

    eq_term = melbo + 0.5 * (mcubo - melbo)

    q0 = 1.0 + 1.0 / eq_term + 1e-10
    var0 = [q0]

    var_opt3 = minimize(root_qELBO, var0, args=(np.mean(ratio_np, axis=1), eq_term),
                        bounds=[(1.0 + 1.0 / eq_term, None)])

    q_star3 = var_opt3.x  # np.mean(q_star2_all)

    mqe_star = compute_qELBO(q_star3, ratio_np)
    mqc_star = compute_qCUBO(q_star3, L_np)

    feed_dict = {x1: x, q1: q, q1_star: q_star3}
    mcubo, L_np, melbo, ratio_np, lpxislse = sess.run([mean_CUBO, L, mean_ELBO, ratio, log_px_IS_logsumexp], feed_dict=feed_dict)

    err_IS = np.mean(abs(log_px - lpxislse))
    print("\n err_IS=", err_IS)

    print("\n mean_CUBO=", mcubo)
    print("\n mean_log_px=", np.mean(log_px))
    print("\n mean_ELBO=", melbo)
    print("\n mean_log_px_IS_logsumexp=", np.mean(lpxislse))
    print('\n qCUBO=', mqc_star)
    print('\n qELBO=', mqe_star)

    y_qCUBO = mqc_star * np.ones(shape=(Nq,))
    y_qELBO = mqe_star * np.ones(shape=(Nq,))

    y_mELBO = melbo * np.ones(shape=(Nq,))
    y_mCUBO = mcubo * np.ones(shape=(Nq,))
    y_mIS = np.mean(lpxislse) * np.ones(shape=(Nq,))

    plt.figure(figsize=(20, 10))

    colours = "rgbmkc"
    plt.plot(q, y_log_px, color=colours[0], label='True value - $\mathbf{(q=1.0)}$')
    plt.plot(q, y_mELBO, color=colours[2], label='mean_ELBO - $\mathbf{(q=1.0)}$')
    plt.plot(q, y_mCUBO, color=colours[1], label='mean_CUBO - $\mathbf{(q=1.0)}$')
    plt.plot(q, y_mIS, color=colours[3], label='IS of true value - $\mathbf{(q=1.0)}$')
    plt.plot(q, y_qCUBO, color=colours[4], label='qCUBO - $\mathbf{(q not eq 1)}$')
    plt.plot(q, y_qELBO, color=colours[5], label='qELBO - $\mathbf{(q not eq 1)}$')
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('log_px_logq_qx.png')

    pdb.set_trace()

    return 1
##########################################################################
##########################################################################
#p(x) = N(x;mu_x = b, cov_x = )
sig_x = 1.5

I = tf.eye(D,dtype=dtype_var)
A = tf.zeros(shape=[D,K],dtype=dtype_var)

A1 = A
diag_values = 0.1 * tf.ones(shape=[K,],dtype=dtype_var)

A = tf.linalg.set_diag(A,diag_values)
AAT = tf.matmul(A,tf.transpose(A))

b = 0.1 * tf.ones(shape=[D],dtype=dtype_var)

mu_x = b
cov_x = tf.add(I, AAT)

mvn_px = tfd.MultivariateNormalFullCovariance(loc=mu_x,covariance_matrix=cov_x)

mean_test = mvn_px.mean()
cov_test = mvn_px.covariance()

##########################################################################
#p(z) = N(z;mu_z = 0, cov_z = I)
mu_z = tf.zeros(shape=(K,),dtype=dtype_var)
std_z = 1.0 * tf.ones(shape=(K,),dtype=dtype_var)#the standard deviation element that goes on the diagonal

mvn_pz = tfd.MultivariateNormalDiag(loc=mu_z, scale_diag=std_z)

mean_z_test = mvn_pz.mean()
std_z_test = mvn_pz.stddev()

##########################################################################
#q(z|x) = N(z;mu_qzx = W*x1+g, cov_qzx = I)

mu_qzx = tf.matmul(x1, W) + g
std_qzx = tf.exp(0.5 * log_var) #var is the variance of the approximate posterior distribution
#std_qzx is the standard deviation of the approximate posterior distribution

log_var_qzx = log_var

cov_qzx = tf.exp(log_var_qzx)

mvn_qzx = tfd.MultivariateNormalDiag(loc=mu_qzx, scale_diag=std_qzx)

# ##########################################################################
# #The true posterior is given by the Bayes' theorem for Gaussian variables (see my written notes based on the book by Bishop, p.90)
# # p(z|x) = N(z;mu_pzx, cov_pzx)
Iz = tf.eye(K,dtype=dtype_var)

ATA = tf.matmul(tf.transpose(A),A)
tmp_node = tf.add(Iz,ATA)

Sig = tf.linalg.inv(tmp_node)

tmp_node2 = tf.transpose(x1-b)#the difference, x1-b, needs to be [D,batch_size]

tmp_node3 = tf.matmul(tf.transpose(A),tmp_node2)# tmp_node3 will be [K,batch_size]

tmp_node4 = tf.matmul(Sig, tmp_node3) # tmp_node4 will be [K,batch_size]

mu_pzx = tf.transpose(tmp_node4) # mu_pzx will be [batch_size,K]

cov_pzx = Sig

#mu_qzx = mu_pzx
#log_var_qzx = tf.diag_part(cov_pzx)

mvn_pzx = tfd.MultivariateNormalFullCovariance(loc=mu_pzx, covariance_matrix=cov_pzx)


#If I use mvn_qzx, instead of mvn_pzx - I want to test the values of ELBO with the true posterior, not the learned approximate posterior
##########################################################################

##########################################################################
#p(x|z) = N(x;mu_xz = A*z+b, cov_xz = I)
#pdb.set_trace()


# ##########################################################################
# #Only one sample of z
# #z_samples = mvn_qzx.sample(N_z)# produces a tensor with the dimensions: [N_z, batch_size, K]
# z_samples = mvn_qzx.sample()# produces a tensor with the dimensions: [batch_size, K]
#
# #This is for only one sample of z
# z_samples_transposed = tf.transpose(z_samples,perm=[1,0])#in this example, we are not learning the parameters of the decoder,
# # but, instead, it is fixed and it comes from the Bayesian formula, where we know p(z), p(x|z) and p(x).
# # As a result, we can analytically compute the real posterior, p(z|x), that matches the given data: p(z), p(x|z) and p(x)
# # With this algorithm, we are learning the approximate posterior, q(z|x), that hopefully will become vanishingly close to p(z|x),
# # as the training progresses
#
#
# mu_xz1 = tf.matmul(A,z_samples_transposed) # will have dimensions=[D,batch_size]
# mu_xz1 = tf.transpose(mu_xz1,perm=[1,0]) # will have dimensions=[batch_size,D]
# mu_xz = mu_xz1 + b #add the bias here; we cannot add the bias in the first line, because it won't perform broadcasting and it will give an error
#
# std_xz = 1.0 * tf.ones(shape=[D])#the standard deviation element that goes on the diagonal
#
# mvn_pxz = tfd.MultivariateNormalDiag(loc=mu_xz, scale_diag=std_xz)
# ##########################################################################

##########################################################################
#Many samples of z
z_samples = mvn_qzx.sample(N_z)# produces a tensor with the dimensions: [N_z, batch_size, K]

z_samples_transposed = tf.transpose(z_samples,perm=[2,0,1])# produces a tensor with the dimensions: [K, N_z, batch_size]

#in this example, we are not learning the parameters of the decoder,
# but, instead, it is fixed and it comes from the Bayesian formula, where we know p(z), p(x|z) and p(x).
# As a result, we can analytically compute the real posterior, p(z|x), that matches the given data: p(z), p(x|z) and p(x)
# With this algorithm, we are learning the approximate posterior, q(z|x), that hopefully will become vanishingly close to p(z|x),
# as the training progresses
z_samples = mvn_qzx.sample(N_z)# produces a tensor with the dimensions: [N_z,batch_size, K]

A1 = tf.tile(A, [N_z,1])
A2 = tf.reshape(A1,[N_z,D,K])# produces a tensor with the dimensions: [N_z,D,K]

z_samples_IS = tf.transpose(z_samples,perm=[0,2,1])# produces a tensor with the dimensions: [N_z,K,batch_size]


mu_xz1 = tf.matmul(A2, z_samples_IS) # will have dimensions=[N_z,D,batch_size]
mu_xz2 = tf.transpose(mu_xz1,perm=[0,2,1]) # will have dimensions=[N_z,batch_size,D]

mu_xz = mu_xz2 + b #add the bias here; we cannot add the bias in the first line, because it won't perform broadcasting and it will give an error

std_xz = 1.0 * tf.ones(shape=[D],dtype=dtype_var)#the standard deviation element that goes on the diagonal

mvn_pxz = tfd.MultivariateNormalDiag(loc=mu_xz, scale_diag=std_xz)

# ##########################################################################
# #New bound
# x1_tile = tf.tile(x1,[N_z,1])# produces a tensor with the dimensions: [N_z,batch_size,D]
# x2 = tf.reshape(x1_tile,[N_z,batch_size,D])# produces a tensor with the dimensions: [N_z,batch_size,D]
#
# log_pxz = mvn_pxz.log_prob(x2)
# log_pz = mvn_pz.log_prob(z_samples)
# log_qzx = mvn_qzx.log_prob(z_samples)
#
# q_node = -log_pxz - log_pz + log_qzx
# q_node = tf.transpose(q_node,perm=[1,0])# will have dimensions=[batch_size,N_z]
#
# #q11 = tf.constant(1.0 + 1e-4, dtype=dtype_var)
# #ratio_q = tf.exp(tf.einsum('i,j->ij',(1 - q11),q_node[0]))# outer product of two vectors
#
# q_loss = np.float64(1.0 - 1e-3)#2 * 1e-3
# ratio_q_loss = tf.exp(tf.multiply(1.0 - q_loss, q_node))# will have dimensions=[batch_size,N_z]
#
# S_loss = tf.reduce_mean(ratio_q_loss,axis=1)# will have dimensions=[batch_size]
# Bound_q_loss = tf.div((1 - S_loss),tf.multiply((1.0 - q_loss),S_loss))
# mean_Bound_q_loss = tf.reduce_mean(Bound_q_loss) # take the mean over the batch_size
#
# #op_node = tf.einsum('i,j->ij',(1 - q1),q_node[0])# outer product of two vectors, dimension=[Nq,N_z_IS]
# op_node2 = tf.multiply(1.0 - q_loss, q_node)# will have dimensions=[batch_size,N_z]
#
# ratio_q_loss2 = tf.reduce_logsumexp(op_node2,axis=1)# will have dimensions=[batch_size]
# S_loss2 = tf.exp(-tf.log(np.float64(N_z)) + ratio_q_loss2)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate
# Bound_q_loss2 = tf.div((1 - S_loss2),tf.multiply((1 - q_loss),S_loss2))
# mean_Bound_q_loss2 = tf.reduce_mean(Bound_q_loss2) # take the mean over the batch_size
#
# #To test
# q_loss_test = np.float64(1.0 - 0.5 * 1e-3)#best result for: D = 30, K = 10, N_z = 100 and q_loss_test = 1.0 + 1e-4
#
# q_node_test = log_pxz + log_pz - log_qzx
# q_node_test = tf.transpose(q_node_test,perm=[1,0])# will have dimensions=[batch_size,N_z]
#
# #op_node = tf.einsum('i,j->ij',(1 - q1),q_node[0])# outer product of two vectors, dimension=[Nq,N_z_IS]
# op_node2_test = tf.multiply(-1.0 + q_loss_test, q_node_test)# will have dimensions=[batch_size,N_z]
#
# ratio_q_loss2_test = tf.reduce_logsumexp(op_node2_test,axis=1)# will have dimensions=[batch_size]
#
# # S_loss2_test = tf.exp(-tf.log(float(N_z)) + ratio_q_loss2_test)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate
# #
# # #Bound_q_loss2_test = tf.div((1 - S_loss2_test),tf.multiply((1 - q_loss_test),S_loss2_test))
# # #mean_Bound_q_loss2_test = tf.reduce_mean(Bound_q_loss2_test) # take the mean over the batch_size
# # log_Bound_q_loss2_test = tf.log((1 - S_loss2_test)) - tf.log(-(1 - q_loss_test)) - tf.log(S_loss2_test)
#
# S_loss2_test = tf.exp(-tf.log(np.float64(N_z)) + ratio_q_loss2_test)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate
# #log_Bound_q_loss2_test = tf.log((1 - S_loss2_test)) - tf.log(-(1 - q_loss_test)) - tf.log(S_loss2_test)
# S_loss2_test_aux = -tf.log(np.float64(N_z)) + ratio_q_loss2_test
# log_Bound_q_loss2_test = tf.log((1 - S_loss2_test)) - tf.log(-(1 - q_loss_test)) - S_loss2_test_aux
#
#
# mean_log_Bound_q_loss2_test = tf.reduce_mean(log_Bound_q_loss2_test) # take the mean over the batch_size
#
# #Try: maximize an upper bound on the ELBO
# op_node4 = tf.multiply(1.0 - q_loss_test, q_node_test)# will have dimensions=[batch_size,N_z]
#
# ratio_q_loss4 = tf.reduce_logsumexp(op_node4,axis=1)# will have dimensions=[batch_size]
#
# S_loss4 = tf.exp(-tf.log(np.float64(N_z)) + ratio_q_loss4)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate
#
# qELBO4 = tf.div(S_loss4 - 1.0,1.0-q_loss_test)
# mean_qELBO4 = tf.reduce_mean(qELBO4)#take the mean over the batch size
#
# mean_log_Bound_q_loss2_test = tf.reduce_mean(log_Bound_q_loss2_test) # take the mean over the batch_size
#
# ##########################################################################

############################################################################################
# Define the loss
log_probs_reconst = mvn_pxz.log_prob(x1)

expectation_node = tf.reduce_mean(log_probs_reconst, axis=0)# approximate the expectation with the Monte Carlo estimate, using samples of z from q(z|x)

mean_reconstruction_loss = tf.reduce_mean(expectation_node) # take the mean over the batch_size

KL = - 0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu_qzx,2) - tf.exp(log_var_qzx), axis=1)#sum over the latent dimensions
mean_KL = tf.reduce_mean(KL) # take the mean over the batch_size

mean_ELBO = mean_reconstruction_loss - mean_KL

loss = -mean_ELBO

ELBO = expectation_node - KL
############################################################################################

##########################################################################
# Importance sample estimate of p(x)
z_samples_IS = mvn_qzx.sample(N_z_IS)# produces a tensor with the dimensions: [N_z_IS,batch_size, K]

A1 = tf.tile(A, [N_z_IS,1])
A2 = tf.reshape(A1,[N_z_IS,D,K])# produces a tensor with the dimensions: [N_z_IS,D,K]

z_samples_IS_transposed = tf.transpose(z_samples_IS,perm=[0,2,1])# produces a tensor with the dimensions: [N_z_IS,K,batch_size]


mu_xz1 = tf.matmul(A2, z_samples_IS_transposed) # will have dimensions=[N_z_IS,D,batch_size]
mu_xz2 = tf.transpose(mu_xz1,perm=[0,2,1]) # will have dimensions=[N_z_IS,batch_size,D]

mu_xz = mu_xz2 + b #add the bias here; we cannot add the bias in the first line, because it won't perform broadcasting and it will give an error

#Check that these computations for IS produce the desired result; the dimensions are ok, but check that the values are ok too (verify IS in a separate python file)

std_xz = 1.0 * tf.ones(shape=[D], dtype=dtype_var)#the standard deviation element that goes on the diagonal
mvn_pxz = tfd.MultivariateNormalDiag(loc=mu_xz, scale_diag=std_xz)

x1_tile = tf.tile(x1,[N_z_IS,1])# produces a tensor with the dimensions: [N_z_IS,batch_size,D]
x2 = tf.reshape(x1_tile,[N_z_IS,batch_size,D])# produces a tensor with the dimensions: [N_z_IS,batch_size,D]

log_pxz = mvn_pxz.log_prob(x2)
log_pz = mvn_pz.log_prob(z_samples_IS)
log_qzx = mvn_qzx.log_prob(z_samples_IS)

ratio = log_pxz + log_pz - log_qzx# will have dimensions=[N_z_IS,batch_size]
log_px_IS = tf.log(tf.reduce_mean(tf.exp(ratio),axis=0))#when ratio == 0, then tf.log gives -inf, which we can't use for training, so we need this 1e-8 as a cutoff for the log

log_px_IS_logsumexp = -tf.log(np.float64(N_z_IS)) + tf.reduce_logsumexp(ratio,axis=0)

#Try: maximize an upper bound on the ELBO
op_node_qELBO_loss = tf.multiply(1.0 - q1_star, ratio)
ratio_qELBO_loss = tf.reduce_logsumexp(op_node_qELBO_loss,axis=0)# will have dimensions=[batch_size]

S_qELBO_loss = tf.exp(-tf.log(np.float64(N_z_IS)) + ratio_qELBO_loss)# we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate

qELBO_loss = tf.div(S_qELBO_loss - 1.0,1.0-q1_star)
mean_qELBO_loss = tf.reduce_mean(qELBO_loss)#take the mean over the batch size

loss_q_star2 = -mean_qELBO_loss

############################################################################################
CUBO = 0.5 * (-tf.log(np.float64(N_z_IS)) + tf.reduce_logsumexp(2.0 * ratio,axis=0)) #standard CUBO, n=2
mean_CUBO = tf.reduce_mean(CUBO)# mean over the batch size

mean_L_CUBO = tf.reduce_mean(tf.exp(2 * CUBO))

L = 2 * CUBO# it should have dimension = [batch_size]

op_node_L = tf.einsum('i,j->ij',(1.0 - q1),L)# outer product of two vectors, dimension=[Nq,batch_size]

S_qCUBO = tf.exp(op_node_L)

mean_S_qCUBO = tf.reduce_mean(S_qCUBO,axis=1)#take the mean over the batch size

mean_qCUBO = tf.div(mean_S_qCUBO - 1.0, 2.0*(1.0 - q1))

############################################################################################
#Find optimum Q
RHS = ELBO + 0.75 * (CUBO - ELBO)
#pdb.set_trace()

F1 = tf.multiply(1.0 - Q, ratio)# will have dimensions=[N_z_IS,batch_size]

F2 = 100 * tf.reduce_logsumexp(F1,axis=0)  # will have dimensions=[batch_size]
# we multiply here with 100, to increase the size of the operands, to make the optimization process easier, i.e. increase the accuracy of the solution

LHS_term = 100 * (tf.log(tf.add(tf.multiply(1.0 - Q, RHS), 1.0)) + tf.log(tf.constant(N_z_IS,dtype=dtype_var)))

func_Q = tf.reduce_mean(tf.pow((LHS_term - F2),2))

opt_Q = tf.train.AdamOptimizer( learning_rate=1e-4,beta1=0.9,beta2=0.999)

#opt_Q_scipy = ScipyOptimizerInterface(func_Q, var_to_bounds={Q: (-np.infty, 1.0-1e-8)})
ineq = [-Q + 1.0]#>=0
opt_Q_scipy = ScipyOptimizerInterface(func_Q, inequalities=ineq, method='SLSQP')

train_step_Q = opt_Q.minimize(func_Q)

############################################################################################
############################################################################################

##########################################################################
# Define the optimization step
opt = tf.train.AdamOptimizer( learning_rate=0.12,beta1=0.9,beta2=0.999)
train_step = opt.minimize(loss)
train_step_q_star2 = opt.minimize(loss_q_star2)#uses qELBO_loss
##########################################################################

##########################################################################
# Initialize session
N_epochs = 1000
N_steps = 200
start_epoch = 1

sess = tf.InteractiveSession()

print("Initializing parameters")
sess.run(tf.global_variables_initializer())


sig_x = 1.5

I = np.eye(D)
A2_np = np.zeros(shape=(D, K))
for i in range(0, K, 1):
    A2_np[i][i] = 0.1

AAT2 = np.matmul(A2_np, A2_np.T)

b2 = 0.1*np.ones(shape=(D,))

mu_x_np = b2
cov_x_np = I + AAT2

mvn_px_np = multivariate_normal(mean=mu_x_np, cov=cov_x_np)

q = np.linspace(1.0 + 1e-6, 1.0 + 1e-4, Nq)


for N in range(1, N_epochs+1, 1):
    #pdb.set_trace()
    x_batch = mvn_px_np.rvs(batch_size)

    feed_dict = {x1: x_batch, q1: q, q1_star: 1.001}

    #fqnp, qnp = sess.run([func_Q, Q], feed_dict=feed_dict)
    #pdb.set_trace()
    #_, fqnp, qnp = sess.run([train_step_Q, func_Q, Q], feed_dict=feed_dict)


    mean_ELBO_step_np = []
    _loss_step = []
    steps = []
    gap = []

    #pdb.set_trace()
    for step in range(1, N_steps+1,1):

        #Get input data: sample x_batch from p(x)
        x_batch = mvn_px_np.rvs(batch_size)

        feed_dict = {x1: x_batch, q1: q, q1_star: 1.0-1e-4}

        if N <= start_epoch:
            q_star3 = 1.0 - 1e-4
            feed_dict = {x1: x_batch, q1: q, q1_star: q_star3}
            feed_dict3 = {x1: x_batch, q1: q, q1_star: q_star3}

            _, lqelbo, onqel, rqel = sess.run([train_step_q_star2, loss_q_star2, op_node_qELBO_loss, ratio_qELBO_loss], feed_dict=feed_dict)  # train with ELBO

            #print("\n loss_qELBO=", lqelbo)

            #pdb.set_trace()
            if step % 100 ==0:
                lqs2, lpxislse = sess.run([loss_q_star2, log_px_IS_logsumexp], feed_dict=feed_dict)  # train with ELBO
                log_px = mvn_px_np.logpdf(x_batch)

                print("\n Epoch=", N, " step=", step, " loss_q_star2=", lqs2, " log_px_IS_logsumexp=", np.mean(lpxislse))
                print("\n mean_err=", np.mean(abs(log_px - lpxislse)))

                #pdb.set_trace()
        if N > start_epoch:
            # find a value for q and train with the qELBO
            # mcubo, mean_ELBO_np, ratio_np, lpxislse = sess.run([mean_CUBO, mean_ELBO, ratio, log_px_IS_logsumexp], feed_dict=feed_dict)
            #
            # eq_term = mean_ELBO_np + 0.75 * (mcubo - mean_ELBO_np)
            #
            # q0 = 1.0 + 1.0 / eq_term + 1e-10
            # if q0 < q_star3:
            #     var0 = [q0]
            # else:
            #     var0 = [q_star3]
            #
            # var_opt3 = minimize(root_qELBO_no_exp, var0, args=(np.mean(ratio_np, axis=1), eq_term),
            #                     bounds=[(1.0 + 1.0 / eq_term, None)])
            #
            # q_star3 = var_opt3.x
            print("\n Before train_step_Q")
            pdb.set_trace()
            _, fqnp, qnp = sess.run([train_step_Q, func_Q, Q], feed_dict=feed_dict)
            #fqnp, qnp = sess.run([func_Q, Q], feed_dict=feed_dict)
            opt_Q_scipy.minimize(sess, feed_dict=feed_dict)

            q_star3 = qnp
            #pdb.set_trace()
            feed_dict3 = {x1: x_batch, q1: q, q1_star: q_star3}
            _, lqelbo, onqel = sess.run([train_step_q_star2, loss_q_star2, op_node_qELBO_loss], feed_dict=feed_dict3)  # train with ELBO

            if step % 100 ==0:
                lqs2, lpxislse = sess.run([loss_q_star2, log_px_IS_logsumexp], feed_dict=feed_dict)  # train with ELBO
                log_px = mvn_px_np.logpdf(x_batch)

                print("\n Epoch=", N, " step=", step, " loss_q_star2=", lqs2, " log_px_IS_logsumexp=", np.mean(lpxislse))
                print("\n mean_err=", np.mean(abs(log_px - lpxislse)))
                print("\n q_star3=",qnp)

    if N > 1:
        lqs2, lpxislse = sess.run([loss_q_star2, log_px_IS_logsumexp], feed_dict=feed_dict3)  # train with ELBO
        log_px = mvn_px_np.logpdf(x_batch)

        print("\n Epoch=", N, " step=", step, " loss_q_star2=", lqs2, " log_px_IS_logsumexp=", np.mean(lpxislse))
        print("\n mean_err=", np.mean(abs(log_px - lpxislse)))
        print("\n q_star3=", qnp)

        v = plot_funcs2(mvn_px_np, q, q_star3)
        #pdb.set_trace()
