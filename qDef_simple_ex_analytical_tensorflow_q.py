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


plt.rcParams.update({'font.size': 22})
#plt.rc('text', usetex=True)
#plt.rcParams['text.usetex'] = True
#plt.rcParams['text.latex.unicode'] = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##########################################################################
D = 200 #dimension of the input data, x
K = 20 #dimension of the latent space, z
#q = 1.0 - 0.1

batch_size = 25#100
N_z = 100
N_z_IS = 1 * 1000
Nq = 100

dtype_var = tf.float64

W = tf.get_variable(name='W_enc', initializer=tf.truncated_normal(shape=[D,K],mean=5.01,stddev=0.01,dtype=dtype_var))
g = tf.get_variable(name='g_enc', initializer=tf.truncated_normal(shape=[K],mean=5.01,stddev=0.01,dtype=dtype_var))
log_var = tf.get_variable(name='log_var_enc', initializer=tf.truncated_normal(shape=[K],mean=5.01,stddev=0.01,dtype=dtype_var))
#the variance has to be positive, so we train log(variance), to allow for negative values

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

# op_node_qELBO_loss = tf.multiply(1.0 - q1_star, ratio)
# ratio_qELBO_loss = tf.reduce_logsumexp(op_node_qELBO_loss,axis=1)# will have dimensions=[batch_size]
#
# S_qELBO_loss = tf.exp(-tf.log(np.float64(N_z_IS)) + ratio_qELBO_loss)# we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate
#
# qELBO_loss = tf.div(S_qELBO_loss - 1.0,1.0-q1_star)
# mean_qELBO_loss = tf.reduce_mean(qELBO_loss)#take the mean over the batch size
#
# loss_q_star2 = -mean_qELBO_loss

def compute_qCUBO(q_min, L_np):
    op_node_L_np = (1.0 - q_min) * L_np

    S_qCUBO_np = np.exp(op_node_L_np)

    #pdb.set_trace()
    mean_S_qCUBO_np = np.mean(S_qCUBO_np)  # take the mean over the batch size

    mean_qCUBO_np = (mean_S_qCUBO_np - 1.0)/( 2.0 * (1.0 - q_min))

    return mean_qCUBO_np

def root_qCUBO(var, L_np, eq_term):
    q_min = var
    op_node_L_np = (1.0 - q_min) * L_np

    S_qCUBO_np = np.exp(op_node_L_np)

    #pdb.set_trace()
    mean_S_qCUBO_np = np.mean(S_qCUBO_np)  # take the mean over the batch size

    mean_qCUBO_np = (mean_S_qCUBO_np - 1.0)/( 2.0 * (1.0 - q_min))

    return (mean_qCUBO_np-eq_term)**2

def plot_funcs2(mvn_px_np, q, q_star3_before):

    x = mvn_px_np.rvs(batch_size)
    log_px = mvn_px_np.logpdf(x)

    y_log_px = np.mean(log_px) * np.ones(shape=(Nq,))

    feed_dict = {x1: x, q1: q, q1_star: q_star3_before}

    #
    # # [qn_np, rq_np, ELBO_np, BQ_np]= sess.run(
    # #     [q_node, ratio_q, ELBO, Bound_q], feed_dict=feed_dict)
    #
    # [log_px_IS_np, ELBO_np, y_Bound_q, lpxz_np,
    #  r, qe, mean_ELBO_np, mBQl_np, mqE,
    #  lpxislse, mcubo, mqc, L_np,
    #  mqelbol, ratio_np] = sess.run(
    #     [log_px_IS, ELBO, Bound_q2, log_pxz,
    #      ratio, qELBO, mean_ELBO, mean_Bound_q_loss, mean_qELBO4,
    #      log_px_IS_logsumexp,mean_CUBO, mean_qCUBO, L,
    #      mean_qELBO_loss, ratio], feed_dict=feed_dict)

    #print("\n log_px[0]=",log_px[0])
    #print("\n ELBO[0]=", ELBO_np[0])


    # print("\n q=",q)
    #print("\n qELBO[q]=", qe)
    #print("\n mean_qCUBO[q]=", mqc)

    #mean_lpxislse = np.mean(lpxislse)

    #pos1 = np.where(mqc<=mean_lpxislse)
    #pos_select = pos1[0][0]-1
    #q_star = q[pos_select]

    #print("\n q_star=",q_star," mean_qCUBO[q_star]=",mqc[pos_select])

    # eq_term = mean_lpxislse
    # var0 = [1.001]
    # var_opt = minimize(root_qCUBO, var0, args=(L_np,eq_term))
    # q_star = var_opt.x

    # eq_term = mcubo - 0.5 * (mcubo-mean_ELBO_np)
    # #eq_term = mcubo
    # var0 = [1.001]
    # var_opt = minimize(root_qCUBO, var0, args=(L_np,eq_term))
    # q_star = var_opt.x
    # mqc_star = compute_qCUBO(q_star,L_np)
    #
    # eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
    # # # eq_term = mcubo
    # # var0 = [1.001]
    # # var_opt2 = minimize(root_qELBO, var0, args=(ratio_np, eq_term))
    # # q_star2 = var_opt2.x
    # # mqe_star = compute_qELBO(q_star2, ratio_np)

    mcubo, L_np, melbo, ratio_np = sess.run([mean_CUBO, L, mean_ELBO, ratio], feed_dict=feed_dict)

    #eq_term = melbo + 0.5 * (np.mean(lpxislse) - melbo)
    eq_term = melbo + 0.75 * (mcubo - melbo)

    q0 = 1.0 + 1.0 / eq_term + 1e-10
    var0 = [q0]

    var_opt3 = minimize(root_qELBO, var0, args=(np.mean(ratio_np, axis=1), eq_term),
                        bounds=[(1.0 + 1.0 / eq_term, None)])

    q_star3 = var_opt3.x  # np.mean(q_star2_all)
    mqe_star = compute_qELBO(q_star3, ratio_np)

    eq_term2 = mcubo - 0.5 * (mcubo - melbo)

    var_opt2 = minimize(root_qCUBO, var0, args=(L_np, eq_term))
    q_star2 = var_opt2.x

    mqc_star = compute_qCUBO(q_star2, L_np)

    # # check if eq_term<0; this minimization procedure only works for eq_term<0 -> see my notebook
    # q_star2_all = []
    # for j in range(0, batch_size, 1):
    #     ratio_np1 = ratio_np.T[j]
    #     rez = root_qELBO_no_exp(q0, ratio_np1, eq_term)
    #
    #     var_opt2 = minimize(root_qELBO_no_exp, var0, args=(ratio_np1, eq_term), bounds=[(1.0 + 1.0 / eq_term, None)])
    #
    #     q_star2_all.append(var_opt2.x)
    #
    # q_star2 = np.mean(q_star2_all)
    # mqe_star = compute_qELBO(q_star2, ratio_np)

    #pdb.set_trace()
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

    #pdb.set_trace()
    #rez = root_qCUBO(1.001, L_np)

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

    #pdb.set_trace()

    #print("\n step=", step)
    #print("\n qELBO=", qe)

    # [mE, mBQL, mBQL2,
    #  on2, rq2, S2np, qn,
    #  on2t, rq2t, S2t, qnt, mlBQ2t] = sess.run([mean_ELBO, mean_Bound_q_loss, mean_Bound_q_loss2,
    #                                            op_node2, ratio_q_loss2, S_loss2, q_node,
    #                                            op_node2_test, ratio_q_loss2_test, S_loss2_test, q_node_test,
    #                                            mean_log_Bound_q_loss2_test], feed_dict=feed_dict)

    #print("\n mean_log_Bound_q_loss2_test=", mlBQ2t)

    #y_mean_qELBO = mqE * np.ones(shape=(Nq,))  # q_loss > 1

    # print("\n Trained with q_loss_test=", q_loss_test)
    # print("\n mean_ELBO=", mean_ELBO_np)
    #
    # print("\n mean_log_px_IS=", y_log_px_IS[0])
    #
    # print("\n q = ",q_loss,", mean_Bound_q_loss=", mBQl_np)
    # print("\n q = ", q_loss_test,", mBQ2test =", -np.exp(mlBQ2t))
    # print("\n q = ", q_loss_test,", mean_qELBO =", y_mean_qELBO[0])
    # print("\n mean_log_px=", y_log_px[0])


    # y_mean_BQ_p = -np.exp(mlBQ2t) * np.ones(shape=(Nq,))# q_loss > 1
    # y_mean_BQ_n = mBQl_np * np.ones(shape=(Nq,))# q_loss < 1

    return 1
##########################################################################


def plot_funcs(mvn_px_np,q):

    x = mvn_px_np.rvs(batch_size)
    log_px = mvn_px_np.logpdf(x)

    y_log_px = np.mean(log_px) * np.ones(shape=(Nq,))

    feed_dict = {x1: x, q1: q}  # np.reshape(x,[1,D])}

    # [qn_np, rq_np, ELBO_np, BQ_np]= sess.run(
    #     [q_node, ratio_q, ELBO, Bound_q], feed_dict=feed_dict)

    [log_px_IS_np, ELBO_np, y_Bound_q, lpxz_np,
     r, qe, mean_ELBO_np, mBQl_np, mqE,
     lpxislse, mcubo, mqc, L_np] = sess.run(
        [log_px_IS, ELBO, Bound_q2, log_pxz,
         ratio, qELBO, mean_ELBO, mean_Bound_q_loss, mean_qELBO4,
         log_px_IS_logsumexp,mean_CUBO, mean_qCUBO, L], feed_dict=feed_dict)

    #print("\n log_px[0]=",log_px[0])
    #print("\n ELBO[0]=", ELBO_np[0])

    print("\n mean_CUBO=", mcubo)
    print("\n mean_log_px=", np.mean(log_px))
    print("\n mean_ELBO=", mean_ELBO_np)
    print("\n mean_log_px_IS=", np.mean(log_px_IS_np))
    print("\n mean_log_px_IS_logsumexp=", np.mean(lpxislse))

    # print("\n q=",q)
    #print("\n qELBO[q]=", qe)
    #print("\n mean_qCUBO[q]=", mqc)

    mean_lpxislse = np.mean(lpxislse)

    #pos1 = np.where(mqc<=mean_lpxislse)
    #pos_select = pos1[0][0]-1
    #q_star = q[pos_select]

    #print("\n q_star=",q_star," mean_qCUBO[q_star]=",mqc[pos_select])

    # eq_term = mean_lpxislse
    # var0 = [1.001]
    # var_opt = minimize(root_qCUBO, var0, args=(L_np,eq_term))
    # q_star = var_opt.x

    eq_term = mcubo - 0.5 * (mcubo-mean_ELBO_np)
    #eq_term = mcubo
    var0 = [1.001]
    var_opt = minimize(root_qCUBO, var0, args=(L_np,eq_term))
    q_star = var_opt.x

    mqc_star = compute_qCUBO(q_star,L_np)

    pdb.set_trace()
    #rez = root_qCUBO(1.001, L_np)

    y_qCUBO = mqc_star * np.ones(shape=(Nq,))

    y_log_px_IS = np.mean(log_px_IS_np) * np.ones(shape=(Nq,))
    y_mELBO = mean_ELBO_np * np.ones(shape=(Nq,))
    y_mCUBO = mcubo * np.ones(shape=(Nq,))
    y_mIS = np.mean(lpxislse) * np.ones(shape=(Nq,))

    y_qELBO = qe[0] * np.ones(shape=(Nq,))
    #y_qELBO = qe

    #pdb.set_trace()

    #print("\n step=", step)
    #print("\n qELBO=", qe)

    [mE, mBQL, mBQL2,
     on2, rq2, S2np, qn,
     on2t, rq2t, S2t, qnt, mlBQ2t] = sess.run([mean_ELBO, mean_Bound_q_loss, mean_Bound_q_loss2,
                                               op_node2, ratio_q_loss2, S_loss2, q_node,
                                               op_node2_test, ratio_q_loss2_test, S_loss2_test, q_node_test,
                                               mean_log_Bound_q_loss2_test], feed_dict=feed_dict)

    #print("\n mean_log_Bound_q_loss2_test=", mlBQ2t)

    y_mean_qELBO = mqE * np.ones(shape=(Nq,))  # q_loss > 1

    # print("\n Trained with q_loss_test=", q_loss_test)
    # print("\n mean_ELBO=", mean_ELBO_np)
    #
    # print("\n mean_log_px_IS=", y_log_px_IS[0])
    #
    # print("\n q = ",q_loss,", mean_Bound_q_loss=", mBQl_np)
    # print("\n q = ", q_loss_test,", mBQ2test =", -np.exp(mlBQ2t))
    # print("\n q = ", q_loss_test,", mean_qELBO =", y_mean_qELBO[0])
    # print("\n mean_log_px=", y_log_px[0])


    y_mean_BQ_p = -np.exp(mlBQ2t) * np.ones(shape=(Nq,))# q_loss > 1
    y_mean_BQ_n = mBQl_np * np.ones(shape=(Nq,))# q_loss < 1


    plt.figure(figsize=(20, 10))

    colours = "rgbmk"
    plt.plot(q, y_log_px, color=colours[0], label='True value - $\mathbf{(q=1.0)}$')
    #plt.plot(q, y_log_px_IS, color=colours[3], label='IS - $\mathbf{(q=1.0)}$')
    plt.plot(q, y_mELBO, color=colours[2], label='mean_ELBO - $\mathbf{(q=1.0)}$')
    plt.plot(q, y_mCUBO, color=colours[1], label='mean_CUBO - $\mathbf{(q=1.0)}$')
    plt.plot(q, y_mIS, color=colours[3], label='IS of true value - $\mathbf{(q=1.0)}$')
    #plt.plot(q, y_mean_BQ_n, color=colours[3], label='upper bound - mean_q-Bound - $\mathbf{(q)} < 1$')
    #plt.plot(q, y_mean_BQ_p, color=colours[1], label='lower bound - mean_q-Bound - $\mathbf{(q)} > 1$')
    plt.plot(q, y_qCUBO, color=colours[4], label='qCUBO - $\mathbf{(q not eq 1)}$')
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    # plt.savefig('log_px_logq_qx.png')

    return 1
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

##########################################################################
#New bound
x1_tile = tf.tile(x1,[N_z,1])# produces a tensor with the dimensions: [N_z,batch_size,D]
x2 = tf.reshape(x1_tile,[N_z,batch_size,D])# produces a tensor with the dimensions: [N_z,batch_size,D]

log_pxz = mvn_pxz.log_prob(x2)
log_pz = mvn_pz.log_prob(z_samples)
log_qzx = mvn_qzx.log_prob(z_samples)

q_node = -log_pxz - log_pz + log_qzx
q_node = tf.transpose(q_node,perm=[1,0])# will have dimensions=[batch_size,N_z]

#q11 = tf.constant(1.0 + 1e-4, dtype=dtype_var)
#ratio_q = tf.exp(tf.einsum('i,j->ij',(1 - q11),q_node[0]))# outer product of two vectors

q_loss = np.float64(1.0 - 1e-3)#2 * 1e-3
ratio_q_loss = tf.exp(tf.multiply(1.0 - q_loss, q_node))# will have dimensions=[batch_size,N_z]

S_loss = tf.reduce_mean(ratio_q_loss,axis=1)# will have dimensions=[batch_size]
Bound_q_loss = tf.div((1 - S_loss),tf.multiply((1.0 - q_loss),S_loss))
mean_Bound_q_loss = tf.reduce_mean(Bound_q_loss) # take the mean over the batch_size

#op_node = tf.einsum('i,j->ij',(1 - q1),q_node[0])# outer product of two vectors, dimension=[Nq,N_z_IS]
op_node2 = tf.multiply(1.0 - q_loss, q_node)# will have dimensions=[batch_size,N_z]

ratio_q_loss2 = tf.reduce_logsumexp(op_node2,axis=1)# will have dimensions=[batch_size]
S_loss2 = tf.exp(-tf.log(np.float64(N_z)) + ratio_q_loss2)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate
Bound_q_loss2 = tf.div((1 - S_loss2),tf.multiply((1 - q_loss),S_loss2))
mean_Bound_q_loss2 = tf.reduce_mean(Bound_q_loss2) # take the mean over the batch_size

#To test
q_loss_test = np.float64(1.0 - 0.5 * 1e-3)#best result for: D = 30, K = 10, N_z = 100 and q_loss_test = 1.0 + 1e-4

q_node_test = log_pxz + log_pz - log_qzx
q_node_test = tf.transpose(q_node_test,perm=[1,0])# will have dimensions=[batch_size,N_z]

#op_node = tf.einsum('i,j->ij',(1 - q1),q_node[0])# outer product of two vectors, dimension=[Nq,N_z_IS]
op_node2_test = tf.multiply(-1.0 + q_loss_test, q_node_test)# will have dimensions=[batch_size,N_z]

ratio_q_loss2_test = tf.reduce_logsumexp(op_node2_test,axis=1)# will have dimensions=[batch_size]

# S_loss2_test = tf.exp(-tf.log(float(N_z)) + ratio_q_loss2_test)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate
#
# #Bound_q_loss2_test = tf.div((1 - S_loss2_test),tf.multiply((1 - q_loss_test),S_loss2_test))
# #mean_Bound_q_loss2_test = tf.reduce_mean(Bound_q_loss2_test) # take the mean over the batch_size
# log_Bound_q_loss2_test = tf.log((1 - S_loss2_test)) - tf.log(-(1 - q_loss_test)) - tf.log(S_loss2_test)

S_loss2_test = tf.exp(-tf.log(np.float64(N_z)) + ratio_q_loss2_test)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate
#log_Bound_q_loss2_test = tf.log((1 - S_loss2_test)) - tf.log(-(1 - q_loss_test)) - tf.log(S_loss2_test)
S_loss2_test_aux = -tf.log(np.float64(N_z)) + ratio_q_loss2_test
log_Bound_q_loss2_test = tf.log((1 - S_loss2_test)) - tf.log(-(1 - q_loss_test)) - S_loss2_test_aux


mean_log_Bound_q_loss2_test = tf.reduce_mean(log_Bound_q_loss2_test) # take the mean over the batch_size

#Try: maximize an upper bound on the ELBO
op_node4 = tf.multiply(1.0 - q_loss_test, q_node_test)# will have dimensions=[batch_size,N_z]

ratio_q_loss4 = tf.reduce_logsumexp(op_node4,axis=1)# will have dimensions=[batch_size]

S_loss4 = tf.exp(-tf.log(np.float64(N_z)) + ratio_q_loss4)# we need to account for N_z, when approximating the expectation with a Monte Carlo estimate

qELBO4 = tf.div(S_loss4 - 1.0,1.0-q_loss_test)
mean_qELBO4 = tf.reduce_mean(qELBO4)#take the mean over the batch size

mean_log_Bound_q_loss2_test = tf.reduce_mean(log_Bound_q_loss2_test) # take the mean over the batch_size

##########################################################################

##########################################################################
# Define the loss
log_probs_reconst = mvn_pxz.log_prob(x1)

expectation_node = tf.reduce_mean(log_probs_reconst, axis=0)# approximate the expectation with the Monte Carlo estimate, using samples of z from q(z|x)

mean_reconstruction_loss = tf.reduce_mean(expectation_node) # take the mean over the batch_size

KL = - 0.5 * tf.reduce_sum(1 + log_var - tf.pow(mu_qzx,2) - tf.exp(log_var_qzx), axis=1)#sum over the latent dimensions
mean_KL = tf.reduce_mean(KL) # take the mean over the batch_size

mean_ELBO = mean_reconstruction_loss - mean_KL

#loss = mean_Bound_q_loss2
#loss = mean_log_Bound_q_loss2_test#we minimize this quantity, because mean_log_Bound_q_loss2_test already has a minus inside the expression

#loss = - mean_qELBO4

ELBO = expectation_node - KL
##########################################################################

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

ratio = log_pxz + log_pz - log_qzx
log_px_IS = tf.log(tf.reduce_mean(tf.exp(ratio),axis=0))#when ratio == 0, then tf.log gives -inf, which we can't use for training, so we need this 1e-8 as a cutoff for the log

log_px_IS_logsumexp = -tf.log(np.float64(N_z_IS)) + tf.reduce_logsumexp(ratio,axis=0)

CUBO = 0.5 * (-tf.log(np.float64(N_z_IS)) + tf.reduce_logsumexp(2.0 * ratio,axis=0)) #standard CUBO, n=2
mean_CUBO = tf.reduce_mean(CUBO)# mean over the batch size

mean_L_CUBO = tf.reduce_mean(tf.exp(2 * CUBO))

L = 2 * CUBO# it should have dimension = [batch_size]

op_node_L = tf.einsum('i,j->ij',(1.0 - q1),L)# outer product of two vectors, dimension=[Nq,batch_size]

S_qCUBO = tf.exp(op_node_L)

mean_S_qCUBO = tf.reduce_mean(S_qCUBO,axis=1)#take the mean over the batch size

mean_qCUBO = tf.div(mean_S_qCUBO - 1.0, 2.0*(1.0 - q1))

############################################################################################
# qCUBO-loss
op_node_L_q = tf.multiply(1.0 - q1_star,L)# q1_star is a scalar

S_qCUBO_q = tf.exp(op_node_L_q)# it should have dimension = [batch_size]

mean_S_qCUBO_q = tf.reduce_mean(S_qCUBO_q,axis=0)#take the mean over the batch size

mean_qCUBO_q = tf.div(mean_S_qCUBO_q - 1.0, 2.0*(1.0 - q1_star))

#loss_q_star = -mean_qCUBO_q # in the beginning of training == inf value; overflow in tf.exp()

log_F2 = tf.reduce_logsumexp(op_node_L_q)
log_F3 = -tf.log(np.float64(batch_size)) + log_F2 # log_F3 is equal to log(2*(1-q)*F1+1) -> see my notebook
#we can optimize log_F3, because it has the same optimum as F1; I defined, F1=mean_qCUBO (mean over the batch size)

loss_q_star = -log_F3
############################################################################################
#Try: maximize an upper bound on the ELBO
op_node_qELBO_loss = tf.multiply(1.0 - q1_star, ratio)
ratio_qELBO_loss = tf.reduce_logsumexp(op_node_qELBO_loss,axis=0)# will have dimensions=[batch_size]

S_qELBO_loss = tf.exp(-tf.log(np.float64(N_z_IS)) + ratio_qELBO_loss)# we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate

qELBO_loss = tf.div(S_qELBO_loss - 1.0,1.0-q1_star)
mean_qELBO_loss = tf.reduce_mean(qELBO_loss)#take the mean over the batch size

loss_q_star2 = -mean_qELBO_loss

############################################################################################
# Test the combination of tf.tile and tf.reshape
x_tr = tf.get_variable(name='x_tile_reshape', initializer=tf.truncated_normal(shape=[4,6],mean=15.01,stddev=0.01,dtype=dtype_var))

x_rep = tf.tile(x_tr, [7, 1])

mvn_tr = tfd.MultivariateNormalDiag(loc=15.01 * tf.ones(shape=(6,),dtype=dtype_var), scale_diag=0.01 * tf.ones(shape=(6,),dtype=dtype_var))

log_tr = mvn_tr.log_prob(x_rep)

x_back47 = tf.reshape(log_tr, shape=[4,7])

x_back74 = tf.reshape(log_tr, shape=[7,4])#this reshaping is correct
#pdb.set_trace()

# ############################################################################################
# #In the first steps of the first epoch, if I train with loss_q_star2, I get overflow in exp,
# #so, we need to use this loss instead -> see my notebook for the derivation
# F1 = (1.0 - q_min) * ratio_np1  # see my notebook for the notation of F1 and F2;
# # ratio_np1=means 1 sample from the batch; for each sample in the batch, we compute a q_opt and then average over the q
#
# F2 = 100 * logsumexp(F1)  # will have dimensions=1
# # we multiply here with 100, to increase the size of the operands, to make the optimization process easier, i.e. increase the accuracy of the solution
#
# LHS_term = 100 * np.log((1.0 - q_min) * eq_term + 1) + np.log(np.float64(N_z_IS))
#
# F1_tf = tf.multiply(1.0 - q1_star, ratio)
# F2_tf = tf.multiply(1.0,tf.reduce_logsumexp(F1,axis=0))
# LHS_term_tf =
#
# return (LHS_term - F2) ** 2
# ############################################################################################

############################################################################################

#ratio = log_pxz + log_pz - log_qzx
#log_px_IS = -tf.log(float(N_z_IS) + tf.reduce_logsumexp(ratio,axis=0))#tf.reduce_logsumexp(ratio,axis=0)) = computes the log(sum(exp(ratio))) across axis=0

loss = -mean_ELBO
#loss = -mean_CUBO
#loss = mean_CUBO - mean_ELBO # minimize  the gap between the ELBO and the CUBO

#loss_IS = -tf.reduce_mean(log_px_IS_logsumexp)# mean over the batch size

mean_IS = tf.reduce_mean(log_px_IS_logsumexp)

loss_IS = tf.abs(mean_CUBO - mean_IS) # minimize the absolute value of the gap between CUBO and IS_log_px


pxz = mvn_pxz.prob(x2)
pz = mvn_pz.prob(z_samples_IS)
qzx = mvn_qzx.prob(z_samples_IS)

#q1 = tf.reshape(q1,[-1,1])
q_node_IS = -log_pxz - log_pz + log_qzx
q_node_IS = tf.transpose(q_node_IS,perm=[1,0])# will have dimensions=[batch_size,N_z_IS]

op_node_IS = tf.einsum('i,j->ij',(1 - q1),q_node_IS[0])# outer product of two vectors, dimension=[Nq,N_z_IS]

ratio_q = tf.exp(op_node_IS)

S = tf.reduce_mean(ratio_q,axis=1)
Bound_q = tf.div((1 - S),tf.multiply((1 - q1),S))

# q_node2 = tf.div(qzx,tf.multiply(pxz,pz))
# q_node2 = tf.transpose(q_node2,perm=[1,0])# will have dimensions=[batch_size,N_z_IS]
#
# ratio_q2 = tf.pow(q_node2,1-q_loss)# outer product of two vectors
#
# S2 = tf.reduce_mean(ratio_q2,axis=0)
# Bound_q2 = tf.div((1 - S2),tf.multiply((1 - q_loss),S2))
q_node2 = q_node_IS
ratio_q2 = tf.reduce_logsumexp(op_node_IS,axis=1)

S2 = tf.exp(-tf.log(np.float64(N_z_IS))+ratio_q2)

Bound_q2 = tf.div((1 - S2),tf.multiply((1 - q1),S2))

#loss = -Bound_q2
##########################################################################
#Compute qELBO
q_node3 = log_pxz + log_pz - log_qzx
q_node3 = tf.transpose(q_node3,perm=[1,0])# will have dimensions=[batch_size,N_z_IS]

#op_node3 = tf.einsum('i,j->ij',(1 - q1),q_node3[0])# outer product of two vectors, dimension=[Nq,N_z_IS]

op_node3 = tf.einsum('i,j->ij',(1 - q1),q_node3[0])# outer product of two vectors, dimension=[Nq,N_z_IS]

ratio_q3 = tf.reduce_logsumexp(op_node3,axis=1)

S3 = tf.exp(-tf.log(np.float64(N_z_IS))+ratio_q3)

qELBO = tf.div(S3 - 1.0,1-q1)

##########################################################################
# Define the optimization step
opt = tf.train.AdamOptimizer( learning_rate=0.12,beta1=0.9,beta2=0.999)
train_step = opt.minimize(loss)
train_step_IS = opt.minimize(loss_IS)
train_step_q_star = opt.minimize(loss_q_star)#uses qCUBO_loss
train_step_q_star2 = opt.minimize(loss_q_star2)#uses qELBO_loss
##########################################################################

##########################################################################
# Initialize session
N_epochs = 1000
N_steps = 100
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

#q = q_loss * np.ones(shape=(Nq,))
    # print("\n log_px=", log_px_np)
q = np.linspace(1.0 + 1e-6, 1.0 + 1e-4, Nq)
#
# for N in range(1, N_epochs+1, 1):
#     x_batch = mvn_px_np.rvs(batch_size)
#
#     feed_dict = {x1: x_batch, q1: q, q1_star: 1.001}
#
#     mean_ELBO_step_np = []
#     _loss_step = []
#     steps = []
#     gap = []
#     for step in range(1, N_steps+1,1):
#
#         #Get input data: sample x_batch from p(x)
#         x_batch = mvn_px_np.rvs(batch_size)
#
#         feed_dict = {x1: x_batch, q1: q, q1_star: 1.001}
#
#         # mean_test_np, cov_test_np, \
#         # A1_np, A_np, AAT_np, \
#         # mean_z_test_np, std_z_test_np = sess.run([mean_test, cov_test,
#         #                                           A1, A, AAT,
#         #                                           mean_z_test, std_z_test], feed_dict=feed_dict)
#
#         # W_np, g_np, log_var_np, mu_qzx_np, std_qzx_np = sess.run([W,g,log_var,
#         #                                                         mu_qzx, std_qzx], feed_dict=feed_dict)
#
#         # zs_np, zs_t_np, A_np, b_np, mu_xz1_np, mu_xz_np, std_xz_np, mu_qzx_np = sess.run(
#         #     [z_samples, z_samples_transposed, A, b, mu_xz1, mu_xz, std_xz, mu_qzx], feed_dict=feed_dict)
#
#         # zs_np, mu_qzx_np, std_qzx_np, mu_xz_np, std_xz_np = sess.run(
#         #     [z_samples, mu_qzx, std_qzx, mu_xz, std_xz], feed_dict=feed_dict)
#         # zs_np, mu_qzx_np, lpr_np, mean_KL_np, loss_np = sess.run(
#         #     [z_samples, mu_qzx, log_probs_reconst, mean_KL, loss], feed_dict=feed_dict)
#
#         # Sig_np, A_np, ATA_np, tn2_np, mu_pzx_np, cov_pzx_np, cov_qzx_np, ELBO_np, std_qzx_np, zs_IS_np  = sess.run(
#         #     [Sig, A, ATA, tmp_node2, mu_pzx, cov_pzx, cov_qzx, ELBO, std_qzx, z_samples_IS], feed_dict=feed_dict)
#         #
#         # zs_IS_np, mu_xz1_np, mu_xz2_np, mu_xz_np, A1_np, A2_np,\
#         # lpxz_np, x1_np, x1t_np, x2_np, lpz_np, lqzx_np, r_np, lpx_np  = sess.run(
#         #     [z_samples_IS, mu_xz1, mu_xz2, mu_xz, A1, A2,
#         #      log_pxz, x1, x1_tile, x2, log_pz, log_qzx, ratio, log_px_IS], feed_dict=feed_dict)
#
#         #_, loss_np1 = sess.run([train_step,loss], feed_dict=feed_dict)
#
#         # [mBQl_np, lpx_np, ELBO_np, mean_ELBO_np, loss_np, BQ_np, lpxz_np, cov_pzx_np, cov_qzx_np] = sess.run(
#         #     [mean_Bound_q_loss, log_px_IS, ELBO, mean_ELBO, loss, Bound_q, log_pxz, cov_pzx, cov_qzx], feed_dict=feed_dict)
#         #
#         # [mBQl_np, mean_ELBO_np, loss_np, qn2_np, rq2, lpxz, lpz, lqzx, BQ1, BQ2, s, s2, qe] = sess.run(
#         #     [mean_Bound_q_loss, mean_ELBO, loss, q_node2, ratio_q2, log_pxz, log_pz, log_qzx, Bound_q, Bound_q2, S, S2, qELBO],
#         #     feed_dict=feed_dict)
#         #
#         #
#         # print("\n Before the update step:")
#         # [mE, mBQL, mBQL2,
#         #  on2, rq2, S2np, qn,
#         #  on2t, rq2t, S2t, qnt, mlBQ2t] = sess.run([mean_ELBO, mean_Bound_q_loss, mean_Bound_q_loss2,
#         #                                             op_node2, ratio_q_loss2, S_loss2, q_node,
#         #                                             op_node2_test, ratio_q_loss2_test, S_loss2_test, q_node_test,
#         #                                             mean_log_Bound_q_loss2_test], feed_dict=feed_dict)
#         # print("\n mean_log_Bound_q_loss2_test=",mlBQ2t)
#
#
#         # _, loss_np = sess.run([train_step, loss], feed_dict=feed_dict)
#         #
#
#         # L_np, loss_np,\
#         # sqc, msqc,\
#         # mcubo, melbo,\
#         #     mlcubo = sess.run([L, loss,
#         #                                                    S_qCUBO, mean_S_qCUBO,
#         #                                                    mean_CUBO, mean_ELBO,
#         #                                                    mean_L_CUBO], feed_dict=feed_dict)
#         # #pdb.set_trace()
#
#         # eq_term = mcubo - 0.5 * (mcubo - melbo)
#         # # eq_term = mcubo
#         # var0 = [1.0 + 1e-3]
#         # var_opt = minimize(root_qCUBO, var0, args=(L_np, eq_term))
#         # q_star = var_opt.x
#         #
#         # mqc_star = compute_qCUBO(q_star, L_np)
#         #
#         # feed_dict = {x1: x_batch, q1: q, q1_star: q_star}
#
#         # ##########################################################################################
#         # mcubo, mean_ELBO_np, ratio_np = sess.run([mean_CUBO, mean_ELBO, ratio],feed_dict=feed_dict)
#         #
#         # eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
#         # var0 = [1.001]
#         # var_opt2 = minimize(root_qELBO, var0, args=(ratio_np, eq_term))
#         # q_star2 = var_opt2.x
#         # mqe_star = compute_qELBO(q_star2, ratio_np)
#         #
#         # feed_dict = {x1: x_batch, q1: q, q1_star: q_star2}
#         #
#         # _, lqs, lf2, lf3,\
#         #     mqel = sess.run([train_step, loss_q_star, log_F2, log_F3,
#         #                              mean_qELBO_loss], feed_dict=feed_dict)
#         #
#         # print("\n mean_qELBO_loss=", mqel)
#         #
#         # pdb.set_trace()
#         # ##########################################################################################
#
#         if N <= start_epoch:
#             _, loss_np = sess.run([train_step,loss], feed_dict=feed_dict)
#             loss_plot = loss_np
#             #print("\n Trained with ELBO loss")
#         # else:
#         #     _, loss_np, lqs = sess.run([train_step_q_star, loss, loss_q_star], feed_dict=feed_dict)
#         #     loss_plot = lqs
#
#             # _, loss_np, loss_IS_np = sess.run([train_step, loss, loss_IS], feed_dict=feed_dict)
#             # loss_plot = loss_IS_np
#
#             #print("\n Trained with log_px_IS loss")
#
#             # x = mvn_px_np.rvs(batch_size)
#             # log_px = mvn_px_np.logpdf(x)
#             #
#             # feed_dict = {x1: x_batch, q1: q}
#             #
#             # [lpxis, lpxislse, r] = sess.run([log_px_IS, log_px_IS_logsumexp, ratio], feed_dict=feed_dict)
#             #
#             # pdb.set_trace()
#             # gap.append(np.mean(log_px-lpxis))
#             # steps.append(step)
#
#
#             # [loss_np, lpxis, r] = sess.run([ loss, log_px_IS, ratio], feed_dict=feed_dict)
#         #
#         #
#         # print("\n loss=",loss_np)
#         # print("\n log_px_IS=", lpxis)
#         # print("\n ratio=", r)
#         #
#         # pdb.set_trace()
#
#         # print("\n After the update step:")
#         # [mE, mBQL, mBQL2,
#         #  on2, rq2, S2np, qn,
#         #  on2t, rq2t, S2t, qnt, mlBQ2t] = sess.run([mean_ELBO, mean_Bound_q_loss, mean_Bound_q_loss2,
#         #                                             op_node2, ratio_q_loss2, S_loss2, q_node,
#         #                                             op_node2_test, ratio_q_loss2_test, S_loss2_test, q_node_test,
#         #                                             mean_log_Bound_q_loss2_test], feed_dict=feed_dict)
#         # print("\n mean_log_Bound_q_loss2_test=", mlBQ2t)
#         # pdb.set_trace()
#
#         #s should be == np.exp(-np.log(N_z_IS) + rq2), when rq2 = tf.reduce_logsumexp(op_node,axis=1)
#
#
#         #mean_ELBO_step_np.append(mE)
#         #steps.append(step)
#
#         #_loss_step.append(-loss_np)
#
#         #log_px_np = mvn_px_np.logpdf(x_batch[0])#qELBO is computed for the same element of the batch
#
#         #lpr_np, en_np, mrl_np, kl_np, mkl_np, loss_np = sess.run([log_probs_reconst, exp_node, mean_reconstruction_loss, KL, mean_KL, loss], feed_dict=feed_dict)
#
#         #pdb.set_trace()
#         # if step % 100 == 0:
#         #     print("\n step=", step)
#         #     print("\n ELBO=", mean_ELBO_np)
#         #     print("\n mean_Bound_q_loss=", mBQl_np)
#         #     print("\n log_px=", log_px_np)
#         #     print("\n qELBO=", qe)
#
#             #print("\n log_px_IS=", lpx_np)
#
#             #print("\n cov_pzx=", cov_pzx_np)
#             #print("\n var_qzx=", cov_qzx_np)
#
#             #pdb.set_trace()
#
#         # _, loss_np = sess.run(
#         #     [train_step, loss], feed_dict=feed_dict)
#     # print("\n ELBO=", ELBO_np)
#     # print("\n log_px_IS=", lpx_np)
#     #
#     # print("\n cov_pzx=", cov_pzx_np)
#     # print("\n var_qzx=", cov_qzx_np)
#
#     # y_log_px = np.mean(log_px_np) * np.ones(shape=(N_steps,))
#     # plt.figure(figsize=(20, 10))
#     #
#     # colours = "rgb"
#     # start = 0
#     # end = 200
#     # plt.plot(steps[start:end], mean_ELBO_step_np[start:end], color=colours[0], label='mean_ELBO - $\mathbf{(q=1.0)}$')
#     # plt.plot(steps[start:end], y_log_px[start:end], color=colours[1], label='mean_True_log_p(x) - $\mathbf{(q=1.0)}$')
#     # plt.plot(steps[start:end], _loss_step[start:end], color=colours[2], label='-mean_loss - $\mathbf{(q<1.0)}$')
#     # plt.grid()
#     # plt.legend(loc='best')
#     # plt.show()
#     # # plt.savefig('log_px_logq_qx.png')
#     #
#     # pdb.set_trace()
#
#     #print("\n Epoch=",N)
#
#     #print("\n Epoch=", N, " step=", step, " Loss=", loss_plot)
#     print("\n Epoch=", N, " step=", step)
#     if N <= start_epoch:
#         print("\n Trained with ELBO loss")
#         #print("\n mean_log_px-mean_log_px_IS=",np.mean(gap))
#     else:
#         #print("\n Trained with log_px_IS loss")
#         print("\n Trained with loss_q_star")
#         #print("\n mean_log_px-mean_log_px_IS=", np.mean(gap))
#
#     if N > start_epoch:
#         #print("\n Epoch=",N," step=",step," Loss=",loss_np)
#
#         # [mE, mBQL, mBQL2,
#         #  on2, rq2, S2np, qn,
#         #  on2t, rq2t, S2t, qnt, mlBQ2t] = sess.run([mean_ELBO, mean_Bound_q_loss, mean_Bound_q_loss2,
#         #                                            op_node2, ratio_q_loss2, S_loss2, q_node,
#         #                                            op_node2_test, ratio_q_loss2_test, S_loss2_test, q_node_test,
#         #                                            mean_log_Bound_q_loss2_test], feed_dict=feed_dict)
#         # print("\n mean_log_Bound_q_loss2_test=", mlBQ2t)
#         # pdb.set_trace()
#
#         #_, loss_np = sess.run([train_step, loss], feed_dict=feed_dict)
#
#         #pdb.set_trace()
#         # L_np, loss_np, sqc, msqc, mcubo, melbo = sess.run([L, loss, S_qCUBO, mean_S_qCUBO, mean_CUBO, mean_ELBO], feed_dict=feed_dict)
#         #
#         # eq_term = mcubo - 0.5 * (mcubo - melbo)
#         # # eq_term = mcubo
#         # var0 = [1.0 + 1e-3]
#         # var_opt = minimize(root_qCUBO, var0, args=(L_np, eq_term))
#         # q_star = var_opt.x
#         #
#         # mqc_star = compute_qCUBO(q_star, L_np)
#         # _, loss_np = sess.run([train_step, loss], feed_dict=feed_dict)
#         #
#         # pdb.set_trace()
#
#         # ##########################################################################################
#         # # qCUBO_loss
#         # L_np, loss_np, \
#         # sqc, msqc, \
#         # mcubo, melbo,\
#         #     mlcubo = sess.run([L, loss,
#         #                        S_qCUBO, mean_S_qCUBO,
#         #                        mean_CUBO, mean_ELBO,
#         #                        mean_L_CUBO], feed_dict=feed_dict)
#         #
#         # eq_term = mcubo - 0.25 * (mcubo - melbo)
#         # # eq_term = mcubo
#         # var0 = [1.0 + 1e-3]
#         # var_opt = minimize(root_qCUBO, var0, args=(L_np, eq_term))
#         # q_star = var_opt.x
#         #
#         # mqc_star = compute_qCUBO(q_star, L_np)
#         #
#         # feed_dict = {x1: x_batch, q1: q, q1_star: q_star}
#         # _, lqs, lf2, lf3 = sess.run([train_step, loss_q_star, log_F2, log_F3], feed_dict=feed_dict)
#         # ##########################################################################################
#
#         ##########################################################################################
#         # qELBO_loss
#         mcubo, mean_ELBO_np, ratio_np = sess.run([mean_CUBO, mean_ELBO, ratio],feed_dict=feed_dict)
#
#         eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
#         var0 = [1.001]
#         var_opt2 = minimize(root_qELBO, var0, args=(ratio_np, eq_term))
#         q_star2 = var_opt2.x
#         mqe_star = compute_qELBO(q_star2, ratio_np)
#
#         feed_dict = {x1: x_batch, q1: q, q1_star: q_star2}
#
#         _, lqs, lf2, lf3,\
#             mqel = sess.run([train_step, loss_q_star, log_F2, log_F3,
#                                      mean_qELBO_loss], feed_dict=feed_dict)
#
#         print("\n mean_qELBO_loss=", mqel)
#
#         pdb.set_trace()
#         ##########################################################################################
#
#         if N > 1:
#             print("\n q_star=",q_star)
#             print("\n log_F3=", lf3)
#             v = plot_funcs2(mvn_px_np, q, q_star)
#
#         #pdb.set_trace()
#         #
#         # feed_dict = {x1: x_batch, q1: q, q1_star: q_star}
#         # _, lqs, lf2, lf3 = sess.run([train_step, loss_q_star, log_F2, log_F3], feed_dict=feed_dict)
#         #
#         # print("\n log_F2=", lf2)
#         # print("\n log_F3=", lf3)
#
#         # pdb.set_trace()
#
#         #v = plot_funcs2(mvn_px_np, q, q_star)
#
#         #v = plot_funcs(mvn_px_np,q)
#         #
#         # print("\n step=", step)
#         # print("\n ELBO=", mean_ELBO_np)
#         # print("\n mean_Bound_q_loss=", mBQl_np)
#         # print("\n log_px=", log_px_np)
#         # print("\n qELBO=", qe)
#
#         #pdb.set_trace()
#
#
# ##########################################################################


for N in range(1, N_epochs+1, 1):
    #pdb.set_trace()
    x_batch = mvn_px_np.rvs(batch_size)

    feed_dict = {x1: x_batch, q1: q, q1_star: 1.001}

    print("\n Test tf.tile and tf.reshape")
    xb47, xb74 = sess.run([x_back47, x_back74],feed_dict=feed_dict)

    pdb.set_trace()
    mean_ELBO_step_np = []
    _loss_step = []
    steps = []
    gap = []

    #pdb.set_trace()
    for step in range(1, N_steps+1,1):

        #Get input data: sample x_batch from p(x)
        x_batch = mvn_px_np.rvs(batch_size)

        feed_dict = {x1: x_batch, q1: q, q1_star: 1.0-1e-4}

        # ##########################################################################################
        # mcubo, mean_ELBO_np, ratio_np = sess.run([mean_CUBO, mean_ELBO, ratio],feed_dict=feed_dict)
        #
        # eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
        # var0 = [1.001]
        # var_opt2 = minimize(root_qELBO, var0, args=(ratio_np, eq_term))
        # q_star2 = var_opt2.x
        # mqe_star = compute_qELBO(q_star2, ratio_np)
        #
        # feed_dict = {x1: x_batch, q1: q, q1_star: q_star2}
        #
        # _, lqs, lf2, lf3,\
        #     mqel = sess.run([train_step, loss_q_star, log_F2, log_F3,
        #                              mean_qELBO_loss], feed_dict=feed_dict)
        #
        # print("\n mean_qELBO_loss=", mqel)
        #
        # pdb.set_trace()
        # ##########################################################################################

        if N <= start_epoch:
            # _, loss_np = sess.run([train_step,loss], feed_dict=feed_dict)#train with ELBO
            # loss_plot = loss_np

            # find a value for q and train with the qELBO
            #mcubo, mean_ELBO_np, ratio_np, rqel = sess.run([mean_CUBO, mean_ELBO, ratio, ratio_qELBO_loss], feed_dict=feed_dict)
            #
            # pdb.set_trace()
            # #print("\n Epoch N=", N, " starting minimizing routine to find q for qELBO")
            #eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
            #
            #q0 = 1.0 + 1.0 / eq_term + 1e-10
            #var0 = [q0]
            #
            #var_opt3 = minimize(root_qELBO_no_exp, var0, args=(np.mean(ratio_np, axis=1), eq_term),
            #                    bounds=[(1.0 + 1.0 / eq_term, None)])
            #
            # #pdb.set_trace()
            #q_star3 = var_opt3.x
            # mqe_star3 = compute_qELBO(q_star3, ratio_np)
            #
            q_star3 = 1.0 - 1e-4
            feed_dict = {x1: x_batch, q1: q, q1_star: q_star3}
            feed_dict3 = {x1: x_batch, q1: q, q1_star: q_star3}

            _, lqelbo, onqel, rqel = sess.run([train_step_q_star2, loss_q_star2, op_node_qELBO_loss, ratio_qELBO_loss], feed_dict=feed_dict)  # train with ELBO

            #print("\n mqe_star3=", mqe_star3)
            print("\n loss_qELBO=", lqelbo)

            #pdb.set_trace()
            if step % 100 ==0:
                print("\n Epoch=", N, " step=", step)
                lqs2, lpxislse = sess.run([loss_q_star2, log_px_IS_logsumexp], feed_dict=feed_dict)  # train with ELBO
                log_px = mvn_px_np.logpdf(x_batch)

                print("\n Epoch=", N, " step=", step, " loss_q_star2=", lqs2, " log_px_IS_logsumexp=", np.mean(lpxislse))
                print("\n mean_err=", np.mean(abs(log_px - lpxislse)))

                pdb.set_trace()
        if N > start_epoch:
            # ##########################################################################################
            # # qCUBO_loss
            # L_np, loss_np, \
            # sqc, msqc, \
            # mcubo, melbo,\
            #     mlcubo = sess.run([L, loss,
            #                        S_qCUBO, mean_S_qCUBO,
            #                        mean_CUBO, mean_ELBO,
            #                        mean_L_CUBO], feed_dict=feed_dict)
            #
            # eq_term = mcubo - 0.25 * (mcubo - melbo)
            # # eq_term = mcubo
            # var0 = [1.0 + 1e-3]
            # var_opt = minimize(root_qCUBO, var0, args=(L_np, eq_term))
            # q_star = var_opt.x
            #
            # mqc_star = compute_qCUBO(q_star, L_np)
            #
            # feed_dict = {x1: x_batch, q1: q, q1_star: q_star}
            # _, lqs, lf2, lf3 = sess.run([train_step, loss_q_star, log_F2, log_F3], feed_dict=feed_dict)
            # ##########################################################################################

            # ##########################################################################################
            # # qELBO_loss
            # mcubo, mean_ELBO_np, ratio_np = sess.run([mean_CUBO, mean_ELBO, ratio],feed_dict=feed_dict)
            #
            # #print("\n Epoch N=",N," starting minimizing routine to find q for qELBO")
            # eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
            #
            # q0 = 1.0 + 1.0/eq_term + 1e-10
            # var0 = [q0]
            #
            # # check if eq_term<0; this minimization procedure only works for eq_term<0 -> see my notebook
            # q_star2_all=[]
            # for j in range(0,batch_size,1):
            #     ratio_np1 = ratio_np.T[j]
            #     rez = root_qELBO_no_exp(q0, ratio_np1, eq_term)
            #
            #     var_opt2 = minimize(root_qELBO_no_exp, var0, args=(ratio_np1, eq_term), bounds=[(1.0 + 1.0/eq_term, None)])
            #
            #     q_star2_all.append(var_opt2.x)

            # var_opt3 = minimize(root_qELBO, var0, args=(np.mean(ratio_np,axis=1), eq_term),
            #                     bounds=[(1.0 + 1.0 / eq_term, None)])
            #
            # var_opt4 = minimize(root_qELBO_no_exp, var0, args=(np.mean(ratio_np, axis=1), eq_term),
            #                     method='L-BFGS-B',
            #                     bounds=[(1.0 + 1.0 / eq_term, None)],
            #                     options={'ftol':1e-12,
            #                              'gtol':1e-10,
            #                              'eps':1e-10})
            #
            # q_star3 = var_opt3.x
            # mqe_star3 = compute_qELBO(q_star3, ratio_np)
            #
            # q_star4 = var_opt4.x
            # mqe_star4 = compute_qELBO(q_star4, ratio_np)
            #
            # print("\n min_root_qELBO=",var_opt3.x)
            # print("\n min_root_qELBO_no_exp=", var_opt4.x)
            # print("\n mqe_star3=", mqe_star3)
            # print("\n mqe_star4=", mqe_star4)
            #
            # pdb.set_trace()
            # q_star2 = var_opt3.x#np.mean(q_star2_all)
            # mqe_star = compute_qELBO(q_star2, ratio_np)
            # # np.mean(ratio_np,axis=1)
            #
            # #mqe_rez = root_qELBO_no_exp(q_star2, ratio_np, eq_term)#mean over the batch size
            # #mqe_rez = root_qELBO(q_star2, ratio_np, eq_term)  # mean over the batch size
            #
            # print("\n mean_ELBO=", mean_ELBO_np)
            # print("\n mean_CUBO=", mcubo)
            # print("\n mqe_star=", mqe_star)
            # #print("\n mqe_rez=", mqe_rez)
            # pdb.set_trace()

            # q_star2 = 1.0 + 1e-3
            # feed_dict = {x1: x_batch, q1: q, q1_star: q_star2}
            #
            # _, lqs, lf2, lf3,\
            #     mqel = sess.run([train_step, loss_q_star, log_F2, log_F3,
            #                              mean_qELBO_loss], feed_dict=feed_dict)
            #print("\n mean_qELBO_loss=", mqel)
            #print("\n Epoch=", N, " step=", step)

            # find a value for q and train with the qELBO
            mcubo, mean_ELBO_np, ratio_np, lpxislse = sess.run([mean_CUBO, mean_ELBO, ratio, log_px_IS_logsumexp], feed_dict=feed_dict)

            #print("\n Epoch N=", N, " starting minimizing routine to find q for qELBO")

            #eq_term = mean_ELBO_np + 0.5 * (np.mean(lpxislse) - mean_ELBO_np)
            eq_term = mean_ELBO_np + 0.75 * (mcubo - mean_ELBO_np)

            q0 = 1.0 + 1.0 / eq_term + 1e-10
            if q0 < q_star3:
                var0 = [q0]
            else:
                var0 = [q_star3]

            var_opt3 = minimize(root_qELBO_no_exp, var0, args=(np.mean(ratio_np, axis=1), eq_term),
                                bounds=[(1.0 + 1.0 / eq_term, None)])

            #pdb.set_trace()
            q_star3 = var_opt3.x
            #mqe_star3 = compute_qELBO(q_star3, ratio_np)

            feed_dict3 = {x1: x_batch, q1: q, q1_star: q_star3}
            _, lqelbo, onqel = sess.run([train_step_q_star2, loss_q_star2, op_node_qELBO_loss], feed_dict=feed_dict3)  # train with ELBO

            #print("\n mqe_star3=", mqe_star3)
            #print("\n loss_qELBO=", lqelbo)

            #pdb.set_trace()
            if step % 100 ==0:
                print("\n Epoch=", N, " step=", step)

                #v = plot_funcs2(mvn_px_np, q)

                pdb.set_trace()
            ##########################################################################################
    lqs2, lpxislse = sess.run([loss_q_star2, log_px_IS_logsumexp], feed_dict=feed_dict3)  # train with ELBO
    log_px = mvn_px_np.logpdf(x_batch)

    print("\n Epoch=", N, " step=", step, " loss_q_star2=", lqs2, " log_px_IS_logsumexp=", np.mean(lpxislse))
    print("\n mean_err=", np.mean(abs(log_px-lpxislse)))
    #pdb.set_trace()

    if N > 1:
        # # qELBO_loss
        # mcubo, mean_ELBO_np, ratio_np = sess.run([mean_CUBO, mean_ELBO, ratio], feed_dict=feed_dict)
        #
        # print("\n Epoch N=", N, " starting minimizing routine to find q for qELBO")
        # eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
        #
        # q0 = 1.0 + 1.0 / eq_term + 1e-10
        # var0 = [q0]
        #
        # var_opt3 = minimize(root_qELBO, var0, args=(np.mean(ratio_np, axis=1), eq_term),
        #                     bounds=[(1.0 + 1.0 / eq_term, None)])
        #
        # var_opt4 = minimize(root_qELBO_no_exp, var0, args=(np.mean(ratio_np, axis=1), eq_term),
        #                     method='L-BFGS-B',
        #                     bounds=[(1.0 + 1.0 / eq_term, None)],
        #                     options={'ftol': 1e-12,
        #                              'gtol': 1e-10,
        #                              'eps': 1e-10})
        #
        # q_star3 = var_opt3.x
        # mqe_star3 = compute_qELBO(q_star3, ratio_np)
        #
        # q_star4 = var_opt4.x
        # mqe_star4 = compute_qELBO(q_star4, ratio_np)
        #
        # print("\n min_root_qELBO=", var_opt3.x)
        # print("\n min_root_qELBO_no_exp=", var_opt4.x)
        # print("\n mqe_star3=", mqe_star3)
        # print("\n mqe_star4=", mqe_star4)
        #
        # pdb.set_trace()
        # q_star2 = var_opt3.x  # np.mean(q_star2_all)
        # mqe_star = compute_qELBO(q_star2, ratio_np)
        # # np.mean(ratio_np,axis=1)
        #
        # # mqe_rez = root_qELBO_no_exp(q_star2, ratio_np, eq_term)#mean over the batch size
        # # mqe_rez = root_qELBO(q_star2, ratio_np, eq_term)  # mean over the batch size
        #
        # print("\n mean_ELBO=", mean_ELBO_np)
        # print("\n mean_CUBO=", mcubo)
        # print("\n mqe_star=", mqe_star)
        # # print("\n mqe_rez=", mqe_rez)

        # # find a value for q and train with the qELBO
        # mcubo, mean_ELBO_np, ratio_np = sess.run([mean_CUBO, mean_ELBO, ratio], feed_dict=feed_dict)
        #
        # print("\n Epoch N=", N, " starting minimizing routine to find q for qELBO")
        # eq_term = mean_ELBO_np + 0.5 * (mcubo - mean_ELBO_np)
        #
        # q0 = 1.0 + 1.0 / eq_term + 1e-10
        # var0 = [q0]
        #
        # var_opt3 = minimize(root_qELBO, var0, args=(np.mean(ratio_np, axis=1), eq_term),
        #                     bounds=[(1.0 + 1.0 / eq_term, None)])
        #
        # # pdb.set_trace()
        # q_star3 = var_opt3.x
        # #mqe_star3 = compute_qELBO(q_star3, ratio_np)
        #
        # feed_dict = {x1: x_batch, q1: q, q1_star: var_opt3.x}
        # _, lqelbo, onqel = sess.run([train_step_q_star2, loss_q_star2, op_node_qELBO_loss],
        #                             feed_dict=feed_dict)  # train with ELBO
        #
        # pdb.set_trace()

        v = plot_funcs2(mvn_px_np, q, q_star3)
        #pdb.set_trace()
