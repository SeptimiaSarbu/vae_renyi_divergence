import numpy as np
import tensorflow as tf
import time
from scipy.misc import logsumexp
from network.network import construct_network
from .loss_functions import reconstruction_loss
from .loss_functions import log_prior
from .vae import variational_lowerbound

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST')

import pdb
from scipy.optimize import minimize


#q1_star = tf.placeholder(tf.float32, shape=[])
q1_star1 = tf.placeholder(tf.float32, shape=[])#we need to pass 1.0-q0_star, because of float32 and numerical issues
#for qIWAE q is very close to one and gets passed through the placeholder as 1.0, which gives nan in the q-loss

q0_star = 1.0-1e-6

def compute_qIWAE(q_min, logF_np_reshaped):
    #pdb.set_trace()
    t1 = (1.0 - q_min) * logsumexp(logF_np_reshaped, axis=0)
    t2 = np.exp(t1)
    qIWAE_loss_np = (t2 - 1.0)/(1.0 - q_min)
    mean_qIWAE_loss_np = np.mean(qIWAE_loss_np)  # take the mean over the batch size
    return mean_qIWAE_loss_np

def root_qIWAE(var, logF_np_reshaped, eq_term):
    #pdb.set_trace()
    q_min = var
    t1 = (1.0 - q_min) * logsumexp(logF_np_reshaped, axis=0)
    t2 = np.exp(t1)
    qIWAE_loss_np = (t2 - 1.0)/(1.0 - q_min)
    mean_qIWAE_loss_np = np.mean(qIWAE_loss_np)  # take the mean over the batch size
    
    #print("q_min=",q_min)
    #print("\n (mean_qIWAE_loss_np-eq_term)**2=", (mean_qIWAE_loss_np-eq_term)**2)
    #pdb.set_trace()
    return (mean_qIWAE_loss_np-eq_term)**2

def root_qIWAE_no_exp(var, ratio_np_reshaped, eq_term):
    #pdb.set_trace()
    q_min = var
    N_num_samples = ratio_np_reshaped.shape[0]#[ratio_np_reshaped]=[num_samples,batch_size]
    N_batch_size = ratio_np_reshaped.shape[1]

    F1 = ratio_np_reshaped # see my notebook for the notation of F1 and F2;
    # ratio_np1=means 1 sample from the batch; for each sample in the batch, we compute a q_opt and then average over the q

    F2 = logsumexp(F1,axis=0)  # will have dimensions=1
    # we multiply here with 100, to increase the size of the operands, to make the optimization process easier, i.e. increase the accuracy of the solution

    LHS_term = np.log((1.0 - q_min) * eq_term + 1)#this is a number
    LHS_term_tile = LHS_term * np.ones(N_batch_size)
    RHS_term = (1.0 - q_min) * F2

    obj_min = np.sum((LHS_term_tile - RHS_term)**2)

    return obj_min

# def iwae(x, encoder, decoder, num_samples, batch_size, alpha = 0.0):
#     """
#     Compute the loss function of VR lowerbound
#     """
#     #logpxz, logqzx, z_list = reconstruction_loss(x, encoder, decoder, num_samples)
#     logpxz = 0.0
#     logqzx = 0.0
#     L = len(encoder.S_layers)
#     x_rep = tf.tile(x, [num_samples, 1])
#     input = x_rep
#
#     # do encoding
#     samples = []
#     for l in xrange(L):
#         output, logq = encoder.S_layers[l].encode_and_log_prob(input)
#         logqzx = logqzx + logq
#         samples.append(output)
#         input = output
#
#     # do decoding
#     samples = list(reversed(samples))
#     samples.append(x_rep)
#     for l in xrange(L):
#         _, logp = decoder.S_layers[l].encode_and_log_prob(samples[l], eval_output = samples[l+1])
#         logpxz = logpxz + logp
#
#     logpz = log_prior(output, encoder.S_layers[l].get_prob_type())
#     logF = logpz + logpxz - logqzx
#
#     # first compute lowerbound
#     K = float(num_samples)
#     logF_matrix = tf.reshape(logF, [num_samples, batch_size]) * (1 - alpha)
#     logF_max = tf.reduce_max(logF_matrix, 0)
#     logF_matrix -= logF_max
#     logF_normalizer = tf.clip_by_value(tf.reduce_sum(tf.exp(logF_matrix), 0), 1e-9, np.inf)
#     logF_normalizer = tf.log(logF_normalizer)
#     # note here we need to substract log K as we use reduce_sum above
#     if np.abs(alpha - 1.0) > 10e-3:
#         lowerbound = tf.reduce_mean(logF_normalizer + logF_max - tf.log(K)) / (1 - alpha)
#     else:
#         lowerbound = tf.reduce_mean(logF)
#
#     # now compute the importance weighted version of gradients
#     log_ws = tf.reshape(logF_matrix - logF_normalizer, shape=[-1])
#     ws = tf.stop_gradient(tf.exp(log_ws), name = 'importance_weights_no_grad')
#     params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#     gradients = tf.gradients(-logF * ws, params)
#     grad = zip(gradients, params)
#
#     return lowerbound, grad
#
# def make_functions_vae(models, input_size, num_samples, batch_size, alpha = 0.0):
#     encoder, decoder = models
#
#     input = tf.placeholder(tf.float32, [batch_size, input_size])
#     lowerbound, grad = iwae(input, encoder, decoder, num_samples, batch_size, \
#                                         alpha)
#
#     learning_rate_ph = tf.placeholder(tf.float32, shape = [])
#     optimizer = \
#             tf.train.AdamOptimizer(learning_rate=learning_rate_ph, \
#                                    beta1=0.9, beta2=0.999, epsilon=10e-8 \
#                                    ).apply_gradients(grad)
#
#     def updateParams(sess, X, learning_rate = 0.0005):
#         opt, cost = sess.run((optimizer, lowerbound),
#                            feed_dict={input: X,
#                                       learning_rate_ph:learning_rate})
#         return cost
#
#     return updateParams, lowerbound


def iwae(x, encoder, decoder, num_samples, batch_size, alpha=0.0):
    """
    Compute the loss function of VR lowerbound
    """
    # logpxz, logqzx, z_list = reconstruction_loss(x, encoder, decoder, num_samples)
    logpxz = 0.0
    logqzx = 0.0
    L = len(encoder.S_layers)
    x_rep = tf.tile(x, [num_samples, 1])
    input = x_rep

    # do encoding
    samples = []
    for l in range(L):
        output, logq = encoder.S_layers[l].encode_and_log_prob(input)
        logqzx = logqzx + logq
        samples.append(output)
        input = output

    # do decoding
    samples = list(reversed(samples))
    samples.append(x_rep)
    for l in range(L):
        _, logp = decoder.S_layers[l].encode_and_log_prob(samples[l], eval_output=samples[l + 1])
        logpxz = logpxz + logp

    logpz = log_prior(output, encoder.S_layers[l].get_prob_type())
    logF = logpz + logpxz - logqzx

    # first compute lowerbound
    K = float(num_samples)
    logF_matrix = tf.reshape(logF, [num_samples, batch_size]) * (1 - alpha)
    logF_max = tf.reduce_max(logF_matrix, 0)
    logF_matrix -= logF_max
    logF_normalizer = tf.clip_by_value(tf.reduce_sum(tf.exp(logF_matrix), 0), 1e-9, np.inf)
    logF_normalizer = tf.log(logF_normalizer)
    # note here we need to substract log K as we use reduce_sum above
    if np.abs(alpha - 1.0) > 10e-3:
        tmp_l = logF_normalizer + logF_max - tf.log(K)
        lowerbound = tf.reduce_mean(logF_normalizer + logF_max - tf.log(K)) / (1 - alpha)
    else:
        lowerbound = tf.reduce_mean(logF)

    ########################################################################################################################
    logF_reshaped = tf.reshape(logF,shape=[num_samples,batch_size]) #will have dimensions [num_samples,batch_size]

    #T1 = (1.0 - q1_star) * tf.reduce_logsumexp(logF_reshaped,axis=0)
    T1 = (q1_star1) * tf.reduce_logsumexp(logF_reshaped, axis=0)

    T2 = tf.exp(T1)

    #qIWAE_loss = tf.div(T2 - 1.0,1.0 - q1_star)
    qIWAE_loss = tf.div(T2 - 1.0, q1_star1)
    mean_qIWAE_loss = tf.reduce_mean(qIWAE_loss)  # take the mean over the batch size

    CUBO = 0.5 * (-tf.log(float(num_samples)) + tf.reduce_logsumexp(2.0 * logF_reshaped, axis=0))  # standard CUBO, n=2
    #print("In iwae num_samples=",num_samples)
    #pdb.set_trace()
    #CUBO = tf.reduce_logsumexp(logF_reshaped, axis=0)
    mean_CUBO = tf.reduce_mean(CUBO)  # mean over the batch size

    ########################################################################################################################
    
    return lowerbound, logF, mean_qIWAE_loss, mean_CUBO, tmp_l
    
def qiwae(num_samples, batch_size, logF):
    #pdb.set_trace()
    logF_reshaped = tf.reshape(logF, shape=[num_samples, batch_size])  # will have dimensions [num_samples,batch_size]

    # T1 = (1.0 - q1_star) * tf.reduce_logsumexp(logF_reshaped,axis=0)
    T1 = (q1_star1) * tf.reduce_logsumexp(logF_reshaped, axis=0)

    T2 = tf.exp(T1)

    # qIWAE_loss = tf.div(T2 - 1.0,1.0 - q1_star)
    qIWAE_loss = tf.div(T2 - 1.0, q1_star1)
    mean_qIWAE_loss = tf.reduce_mean(qIWAE_loss)  # take the mean over the batch size

    qLowerbound = mean_qIWAE_loss
    return qLowerbound

def make_functions_vae(models, input_size, num_samples, batch_size, alpha=0.0):
    encoder, decoder = models

    input = tf.placeholder(tf.float32, [batch_size, input_size])
    lowerbound, logF, mean_qIWAE_loss, mean_CUBO, tmp_l = iwae(input, encoder, decoder, num_samples, batch_size, \
                            alpha)
    qLowerbound = qiwae(num_samples, batch_size, logF)

    learning_rate_ph = tf.placeholder(tf.float32, shape=[])
    optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate_ph, \
                                   beta1=0.9, beta2=0.999, epsilon=10e-8 \
                                   ).minimize(-qLowerbound)

    def updateParams(sess, X, learning_rate=0.0005):

        global q0_star

        cost, logF_np, mqiwael_np, mcubo, tl = sess.run((lowerbound, logF, mean_qIWAE_loss, mean_CUBO, tmp_l),
                             feed_dict={input: X,
                                        q1_star1: 1.0-q0_star,
                                        learning_rate_ph: learning_rate})
        #pdb.set_trace()
        Nq = logF_np.shape[0]
        
        eq_term = cost + 0.5 * (mcubo - cost)  # these elements do not depend on q1_star; cost here is the standard IWAE

        #eq_term = cost - 0.01 * cost
        q0 = 1.0 + 1.0 / eq_term + 1e-10

        var0 = [q0]
        logF_np_reshaped = np.reshape(logF_np, (num_samples, batch_size))
        var_opt3 = minimize(root_qIWAE, var0, args=(logF_np_reshaped, eq_term), method='L-BFGS-B',
                            bounds=[(1.0 + 1.0 / eq_term, 1.1)], options={'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-9})
        q0_star = var_opt3.x[0]
        mqiwael = compute_qIWAE(q0_star,logF_np_reshaped)

        mqiwael_after = sess.run(qLowerbound,
                                 feed_dict={input: X, q1_star1: 1.0 - q0_star, learning_rate_ph: learning_rate})
        #pdb.set_trace()

        opt = sess.run(optimizer, feed_dict={input: X, q1_star1: 1.0 - q0_star, learning_rate_ph: learning_rate})

        miwae, cost_q, logF_np, mcubo = sess.run((lowerbound, qLowerbound, logF, mean_CUBO),
                             feed_dict={input: X,
                                        q1_star1: 1.0-q0_star,
                                        learning_rate_ph: learning_rate})

        #pdb.set_trace()
        
        # logF_np_reshaped = np.reshape(logF_np, (num_samples, batch_size))  # will have dimensions [num_samples,batch_size]
        # rez = compute_qIWAE(q0_star, logF_np_reshaped)
        # pdb.set_trace()
        # cost, logF_np, mqiwael, mcubo = sess.run((lowerbound, logF, mean_qIWAE_loss, mean_CUBO),
        #                                          feed_dict={input: X,
        #                                                     q1_star1: 1.0-q0_star,
        #                                                     learning_rate_ph: learning_rate})
        # q_test = sess.run(q1_star1,feed_dict={input: X, q1_star1: 1.0-q0_star, learning_rate_ph: learning_rate})
        # pdb.set_trace()
        
        #opt = sess.run(optimizer, feed_dict={input: X, q1_star1: 1.0-q0_star, learning_rate_ph: learning_rate})

        #cost_q, mqiwael_q, mcubo_q, tmp_l_np = sess.run((lowerbound, mean_qIWAE_loss, mean_CUBO, tmp_l),
        #                                      feed_dict={input: X,
        #                                                 q1_star1: 1.0-q0_star,
        #                                                 learning_rate_ph: learning_rate})

        #print("\n mean_CUBO_q=", mcubo_q)
        #print("\n lowerbound_q=", cost_q)
        #print("\n mean_qIWAE_loss_q=", mqiwael_q)
        #print("\n 1.0-q0_star=", 1.0-q0_star)
        #pdb.set_trace()
        
        return cost_q, logF_np, miwae, mcubo

    return updateParams, qLowerbound


def init_optimizer(models, input_size, batch_size = 100, num_samples = 1, **kwargs):
    
    encoder = models[0]; decoder = models[1]
    # vae
    if 'alpha' not in kwargs:
        alpha = 0.0
    else:
        alpha = kwargs['alpha']
    updateParams, lowerbound = \
        make_functions_vae(models, input_size, \
                           num_samples, batch_size, \
                           alpha)

    def fit(sess, X, n_iter = 100, learning_rate = 0.0005, verbose = True):
        # first make batches of source data
        [N, dimX] = X.shape        
        N_batch = N / batch_size
        if np.mod(N, batch_size) != 0:
            N_batch += 1      
        print("training the model for %d iterations with lr=%f" % \
            (n_iter, learning_rate))

        begin = time.time()
        for iteration in range(1, n_iter + 1):
            iteration_qLowerbound = 0
            iteration_miwae = 0
            iteration_mcubo = 0
            #ind_s = np.random.permutation(range(N))

            for j in range(0, int(N_batch)):
                # indl = j * batch_size
                # indr = (j+1) * batch_size
                # ind = ind_s[indl:min(indr, N)]
                # if indr > N:
                #     ind = np.concatenate((ind, ind_s[:(indr-N)]))
                # batch = X[ind]

                batch_label = mnist.train.next_batch(batch_size)
                batch = batch_label[0]

                qLowerbound_np, logF_np, miwae, mcubo = updateParams(sess, batch, learning_rate)
                
                #print("\n Train set: mean_CUBO=", mcubo)
                #print("\n Train set: mean_qIWAE_loss=", lowerbound_q_np)
                #print("\n Train set: mean_IWAE=", miwae)
                
                #pdb.set_trace()
                
                iteration_qLowerbound += qLowerbound_np * batch_size
                iteration_miwae += miwae * batch_size
                iteration_mcubo += mcubo * batch_size

            if verbose:
                end = time.time()
                print("Iteration %d, mcubo = %.2f, time = %.2fs"
                      % (iteration, iteration_mcubo / N, end - begin))
                print("Iteration %d, qLowerbound = %.2f, time = %.2fs"
                      % (iteration, iteration_qLowerbound / N, end - begin)) 
                print("Iteration %d, miwae = %.2f, time = %.2fs"
                      % (iteration, iteration_miwae / N, end - begin))
                #print("Iteration %d, mcubo = %.2f, time = %.2fs"
                #      % (iteration, iteration_mcubo / N, end - begin))

                begin = end
                
        
    def eval_test_ll(sess, X, num_samples):
        #lowerbound = sess.run(variational_lowerbound(X, encoder, decoder, num_samples, X.shape[0], 0.0))
        global q0_star

        batch_size = X.shape[0]
        
        meanIWAE_np, logF_np, \
        mean_qIWAE_loss_np, mean_CUBO_np, tmp_l_np = sess.run(iwae(X, encoder, decoder, num_samples, batch_size, alpha),
                                                              feed_dict={q1_star1: 1.0-q0_star})

        Nq = logF_np.shape[0]

        eq_term = meanIWAE_np + 0.5 * (mean_CUBO_np - meanIWAE_np)  # these elements do not depend on q1_star
        #eq_term = lowerbound_np + 0.1 * lowerbound_np

        q0 = 1.0 + 1.0 / eq_term + 1e-10
        var0 = [q0]
        
        logF_np_reshaped = np.reshape(logF_np, (num_samples, batch_size))
        
        var_opt3 = minimize(root_qIWAE, var0, args=(logF_np_reshaped, eq_term), method='L-BFGS-B',
                            bounds=[(1.0 + 1.0 / eq_term,None )], options={'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-10})
        q0_star = var_opt3.x[0]
        
        #mqiwae_np = compute_qIWAE(q0_star,logF_np_reshaped)

        logF_placeholder = tf.placeholder(tf.float32, [num_samples * batch_size])
        mqiwae_after_np = sess.run(qiwae(num_samples, batch_size, logF_placeholder), feed_dict={logF_placeholder: logF_np, q1_star1: 1.0-q0_star})
        
        log_px_IS_np = sess.run(variational_lowerbound(X, encoder, decoder, num_samples, X.shape[0], 0.0))

        miwae = 0.0#meanIWAE_np
        mqiwae = mqiwae_after_np#mean_qIWAE_loss_q_np
        cost_q = log_px_IS_np
        mcubo = 0.0#mean_CUBO_np

        #pdb.set_trace()
        
        #after we have found the right q for this test bach, we report the log_px_IS (identical to the standard IWAE with M=1, K=num_samples)
        #print("In eval_test_ll: num_samples=",num_samples)
        #pdb.set_trace()
        #lowerbound_q_np, logF_q_np, mean_qIWAE_loss_q_np, mean_CUBO_q_np, tmp_l_np = sess.run(
        #    iwae(X, encoder, decoder, num_samples, batch_size, alpha), feed_dict={q1_star1: 1.0-q0_star})

        return cost_q, miwae, mcubo, mqiwae

    def score(sess, X, num_samples = 100):
        """
        Computer lower bound on data, following the IWAE paper.
        """
        
        begin = time.time()
        print('num. samples for eval:', num_samples)
        
        # compute log_q
        log_px_IS_total = 0
        miwae_total = 0
        mqiwae_total = 0
        mcubo_total = 0

        num_data_test = X.shape[0]
        if num_data_test % batch_size == 0:
            num_batch = num_data_test / batch_size
        else:
            num_batch = num_data_test / batch_size + 1
        
        for i in range(int(num_batch)):
            # indl = i*batch_size
            # indr = min((i+1)*batch_size, num_data_test)
            # minibatch = X[indl:indr]

            batch_label = mnist.test.next_batch(batch_size)
            minibatch = batch_label[0]

            log_px_IS_np, miwae_q, mcubo_q, mqiwae_q = eval_test_ll(sess, minibatch, num_samples)

            log_px_IS_total += log_px_IS_np * batch_size
            mqiwae_total += mqiwae_q * batch_size
            miwae_total += miwae_q * batch_size
            mcubo_total += mcubo_q * batch_size

            #print("\n Test set: mean_CUBO=", mcubo_q)
            #print("\n Test set: lowerbound=", lowerbound_q_np)
            #print("\n Test set: mean_qIWAE_loss=", mqiwael_q)

            #pdb.set_trace()

        end = time.time()
        time_test = end - begin
        log_px_IS_total = log_px_IS_total / float(num_data_test)
        miwae_total = miwae_total / float(num_data_test)
        mqiwae_total = mqiwae_total / float(num_data_test)
        mcubo_total = mcubo_total / float(num_data_test)
        
        print("\n Test set: mcubo_total=", mcubo_total)
        print("\n Test set: log_px_IS_total=", log_px_IS_total)
        print("\n Test set: miwae_total=", miwae_total)
        print("\n Test set: mqiwae_total=", mqiwae_total)

        return log_px_IS_total, time_test
     
    return fit, score                              
