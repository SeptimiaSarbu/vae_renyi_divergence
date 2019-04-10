import numpy as np
import tensorflow as tf
import time
from scipy.misc import logsumexp
from network.network import construct_network
from .loss_functions import reconstruction_loss
from .loss_functions import log_prior

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST')

import pdb
from scipy.optimize import minimize

#q1_star = tf.placeholder(tf.float32, shape=[])
q1_star1 = tf.placeholder(tf.float32, shape=[])#we need to pass 1.0-q0_star, because of float32 and numerical issues

q0_star = 1.0-1e-6

def root_qELBO(var, ratio_np, eq_term, num_samples):
    #pdb.set_trace()
    q_min = var
    op_node_qELBO_loss_np = (1.0 - q_min) * ratio_np

    ratio_qELBO_loss_np = logsumexp(op_node_qELBO_loss_np, axis=0)  # will have dimensions=[batch_size]

    S_qELBO_loss_np = np.exp(-np.log(float(num_samples)) + ratio_qELBO_loss_np)  # we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate

    qELBO_loss_np = (S_qELBO_loss_np - 1.0)/( 1.0 - q_min)
    mean_qELBO_loss_np = np.mean(qELBO_loss_np)  # take the mean over the batch size

    return (mean_qELBO_loss_np-eq_term)**2

def compute_qELBO(q_min, ratio_np, num_samples):
    #pdb.set_trace()
    op_node_qELBO_loss_np = (1.0 - q_min) * ratio_np

    ratio_qELBO_loss_np = logsumexp(op_node_qELBO_loss_np, axis=0)  # will have dimensions=[batch_size]

    S_qELBO_loss_np = np.exp(-np.log(float(num_samples)) + ratio_qELBO_loss_np)  # we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate

    qELBO_loss_np = (S_qELBO_loss_np - 1.0)/( 1.0 - q_min)
    mean_qELBO_loss_np = np.mean(qELBO_loss_np)  # take the mean over the batch size

    return mean_qELBO_loss_np

def root_qELBO_no_exp(var, ratio_np1, eq_term):
    #pdb.set_trace()
    q_min = var
    Nq = ratio_np1.shape[0]

    F1 = (1.0 - q_min) * ratio_np1 # see my notebook for the notation of F1 and F2;
    # ratio_np1=means 1 sample from the batch; for each sample in the batch, we compute a q_opt and then average over the q

    F2 = logsumexp(F1)  # will have dimensions=1
    # we multiply here with 100, to increase the size of the operands, to make the optimization process easier, i.e. increase the accuracy of the solution

    LHS_term = np.log((1.0 - q_min) * eq_term + 1) + np.log(float(Nq))

    return (LHS_term - F2)**2

def variational_lowerbound(x, encoder, decoder, num_samples, batch_size, \
        alpha = 1.0, backward_pass = 'full'):
    """
    Compute the loss function of VR lowerbound
    """
    #logpxz, logqzx, z_list = reconstruction_loss(x, encoder, decoder, num_samples)
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
        _, logp = decoder.S_layers[l].encode_and_log_prob(samples[l], eval_output = samples[l+1])
        logpxz = logpxz + logp

    logpz = log_prior(output, encoder.S_layers[l].get_prob_type())
    logF = logpz + logpxz - logqzx    

    ########################################################################################################################
    ############################################################################################
    # # Test the combination of tf.tile and tf.reshape
    # x_tr = tf.get_variable(name='x_tile_reshape',
    #                        initializer=tf.truncated_normal(shape=[4, 6], mean=15.01, stddev=0.01, dtype=dtype_var))
    # x_rep = tf.tile(x_tr, [7, 1])
    # mvn_tr = tfd.MultivariateNormalDiag(loc=15.01 * tf.ones(shape=(6,), dtype=dtype_var),
    #                                     scale_diag=0.01 * tf.ones(shape=(6,), dtype=dtype_var))
    # log_tr = mvn_tr.log_prob(x_rep)
    # x_back47 = tf.reshape(log_tr, shape=[4, 7])
    # x_back74 = tf.reshape(log_tr, shape=[7, 4])#this reshaping is correct
    # #pdb.set_trace()
    logF_reshaped = tf.reshape(logF,shape=[num_samples,batch_size]) #will have dimensions [num_samples,batch_size]

    #q1_star = 1.0 -1e-6

    #op_node_qELBO_loss = tf.multiply(1.0 - q1_star, logF_reshaped)
    op_node_qELBO_loss = tf.multiply(q1_star1, logF_reshaped)

    ratio_qELBO_loss = tf.reduce_logsumexp(op_node_qELBO_loss, axis=0)  # will have dimensions=[batch_size]

    S_qELBO_loss = tf.exp(-tf.log(float(num_samples)) + ratio_qELBO_loss)  # we need to account for N_z_IS, when approximating the expectation with a Monte Carlo estimate
    #qELBO_loss = tf.div(S_qELBO_loss - 1.0, 1.0 - q1_star)
    qELBO_loss = tf.div(S_qELBO_loss - 1.0, q1_star1)

    #qELBO_loss = tf.exp(-tf.log(float(num_samples)) + ratio_qELBO_loss - tf.log(1-q1_star))
    mean_qELBO_loss = tf.reduce_mean(qELBO_loss)  # take the mean over the batch size

    CUBO = 0.5 * (-tf.log(float(num_samples)) + tf.reduce_logsumexp(2.0 * logF_reshaped, axis=0))  # standard CUBO, n=2
    #CUBO = 0.5 * (-tf.log(float(1000)) + tf.reduce_logsumexp(2.0 * logF_reshaped, axis=0))  # standard CUBO, n=2

    mean_CUBO = tf.reduce_mean(CUBO)  # mean over the batch size
    ########################################################################################################################
    
    if backward_pass == 'max': 
        logF = tf.reshape(logF, [num_samples, batch_size])           
        logF = tf.reduce_max(logF, 0)
        lowerbound = tf.reduce_mean(logF)
    elif backward_pass == 'min':
        logF = tf.reshape(logF, [num_samples, batch_size])
        logF = tf.reduce_min(logF, 0)
        lowerbound = tf.reduce_mean(logF)
    elif np.abs(alpha - 1.0) < 10e-3:
        #lowerbound = tf.reduce_mean(logF)
        lowerbound = mean_qELBO_loss
    else:
        logF = tf.reshape(logF, [num_samples, batch_size])
        logF = logF * (1 - alpha)   
        logF_max = tf.reduce_max(logF, 0)           
        logF = tf.log(tf.clip_by_value(tf.reduce_mean(tf.exp(logF - logF_max), 0), 1e-9, np.inf))
        logF = (logF + logF_max) / (1 - alpha)
        lowerbound = tf.reduce_mean(logF)
    
    #lowerbound = mean_qELBO_loss
    return lowerbound, logF, mean_qELBO_loss, mean_CUBO#, logpz, logpxz, logqzx
    
def make_functions_vae(models, input_size, num_samples, batch_size, \
        alpha = 1.0, backward_pass = 'full'): 
    encoder, decoder = models  
 
    input = tf.placeholder(tf.float32, [batch_size, input_size])

    lowerbound, logF, mean_qELBO_loss, mean_CUBO = variational_lowerbound(input, encoder, decoder, num_samples, batch_size, \
                                        alpha, backward_pass)
                                        
    learning_rate_ph = tf.placeholder(tf.float32, shape = [])
    optimizer = \
            tf.train.AdamOptimizer(learning_rate=learning_rate_ph, \
                                   beta1=0.9, beta2=0.999, epsilon=10e-8 \
                                   ).minimize(-lowerbound)
    
    def updateParams(sess, X, learning_rate = 0.0005):

        #opt, cost0 = sess.run((optimizer, lowerbound), feed_dict={input: X,
        #                                                         q1_star: 1.0 - 1e-6,
        #                                                         learning_rate_ph: learning_rate})
        global q0_star
        #print("\n q0_star=",q0_star)
        cost, logF_np, mqelbol, mcubo = sess.run((lowerbound, logF, mean_qELBO_loss, mean_CUBO),
                                                 feed_dict={input: X,
                                                            q1_star1: 1.0-q0_star,
                                                            learning_rate_ph:learning_rate})

        Nq = logF_np.shape[0]

        eq_term = cost + 0.5 * (mcubo - cost)#these elements do not depend on q1_star

        q0 = 1.0 + 1.0 / eq_term + 1e-10
        var0 = [q0]
        #var_opt3 = minimize(root_qELBO, var0, args=(logF_np, eq_term, num_samples), bounds=[(1.0 + 1.0 / eq_term, None)])

        #var_opt3 = minimize(root_qELBO_no_exp, var0, args=(logF_np, eq_term, num_samples), method='L-BFGS-B',
        #                    bounds=[(1.0 + 1.0 / eq_term, None)], options={'ftol':1e-4,'gtol':1e-4,'eps':1e-6})

        # var_opt3 = minimize(root_qELBO_no_exp, var0, args=(logF_np, eq_term), method='L-BFGS-B',
        #                     bounds=[(1.0 + 1.0 / eq_term, None)], options={'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-10})
        # q0_star = var_opt3.x[0]
        logF_np_reshaped = np.reshape(logF_np, (num_samples, batch_size))

        var_opt3 = minimize(root_qELBO, var0, args=(logF_np_reshaped, eq_term, num_samples), method='L-BFGS-B',
                            bounds=[(1.0 + 1.0 / eq_term, 1.1)], options={'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-10})
        q0_star = var_opt3.x[0]
        #pdb.set_trace()

        #eq_term_comp = compute_qELBO(q0_star, logF_np_reshaped, num_samples)


        cost_q, mqelbol_q, mcubo_q = sess.run((lowerbound, mean_qELBO_loss, mean_CUBO),
                           feed_dict={input: X,
                                      q1_star1: 1.0-q0_star,
                                      learning_rate_ph:learning_rate})

        opt = sess.run((optimizer), feed_dict={input: X,
                                               q1_star1: 1.0 - q0_star,
                                               learning_rate_ph: learning_rate})

        # #var_opt3 = minimize(root_qELBO_no_exp, var0, args=(logF_np, eq_term, num_samples), method='L-BFGS-B',
        # #                    options={'ftol': 1e-4, 'gtol': 1e-4, 'eps': 1e-6})
        #
        # rez0 = root_qELBO_no_exp(q0, logF_np, eq_term)
        # rez_min = root_qELBO_no_exp(var_opt3.x, logF_np, eq_term)
        #
        # #pdb.set_trace()
        # q_min = q0
        # F10 = (1.0 - q_min) * logF_np  # see my notebook for the notation of F1 and F2;
        # # # ratio_np1=means 1 sample from the batch; for each sample in the batch, we compute a q_opt and then average over the q
        # #
        # F20 = logsumexp(F10)  # will have dimensions=1
        # logS0 = -np.log(float(Nq)) + F20
        # S0 = np.exp(logS0)
        #
        # LHS0 = np.log((1.0 - q_min) * eq_term + 1) + np.log(float(Nq))
        #
        # eq_term0 = (S0-1)/(1-q_min)
        #
        # q_min = var_opt3.x
        # F1 = (1.0 - q_min) * logF_np  # see my notebook for the notation of F1 and F2;
        # # # ratio_np1=means 1 sample from the batch; for each sample in the batch, we compute a q_opt and then average over the q
        # #
        # F2 = logsumexp(F1)  # will have dimensions=1
        # logS = -np.log(float(Nq)) + F2
        # S = np.exp(logS)
        #
        # LHS2 = np.log((1.0 - q_min) * eq_term + 1) + np.log(float(Nq))
        #
        # eq_term2 = (S - 1) / (1 - q_min)
        #
        # pdb.set_trace()
        #
        # # # we multiply here with 100, to increase the size of the operands, to make the optimization process easier, i.e. increase the accuracy of the solution
        # #
        # # LHS_term = np.log((1.0 - q_min) * eq_term + 1) + np.log(float(num_samples))
        # #
        # # return (LHS_term - F2) ** 2
        #
        # #print("\n bound on q=",1.0 + 1.0 / eq_term)
        # #print("\n q_opt=",var_opt3.x[0])
        #
        # #pdb.set_trace()


        # cost_q, mqelbol_q, mcubo_q = sess.run((lowerbound, mean_qELBO_loss, mean_CUBO),
        #                    feed_dict={input: X,
        #                               q1_star: var_opt3.x[0],
        #                               learning_rate_ph:learning_rate})

        '''
        print("\n Before q-optimization: mean_CUBO=", mcubo)
        print("\n Before q-optimization: lowerbound=", cost)
        print("\n Before q-optimization: mqelbol=", mqelbol)

        print("\n eq_term=", eq_term)
        print("\n bound on q with eq_term=", 1.0 + 1.0 / eq_term)
        print("\n q_opt=", var_opt3.x[0])

        print("\n After q-optimization: mean_CUBO=", mcubo_q)
        print("\n lowerbound_q=", cost_q)
        print("\n mqelbol_q=", mqelbol_q)

        #compute_mqelbo = compute_qELBO(var_opt3.x[0], logF_np, num_samples)
        #print("\n computed_mqelbol_q=", compute_mqelbo)

        #pdb.set_trace()
        '''
        return cost, logF_np, mqelbol, mcubo_q, cost_q, mqelbol_q

    return updateParams, lowerbound, logF, mean_qELBO_loss
                                        
def init_optimizer(models, input_size, batch_size = 100, num_samples = 1, **kwargs):
    
    encoder = models[0]; decoder = models[1]
    # vae
    if 'alpha' not in kwargs:
        alpha = 1.0
    else:
        alpha = kwargs['alpha']
    if 'backward_pass' not in kwargs:
        backward_pass = 'full'
    else:
        backward_pass = kwargs['backward_pass']
    updateParams, lowerbound, logF, mean_qELBO_loss = \
        make_functions_vae(models, input_size, \
                           num_samples, batch_size, \
                           alpha, backward_pass)

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
            iteration_lowerbound = 0
            iteration_mcubo_q = 0
            iteration_mqelbol_q = 0
            #ind_s = np.random.permutation(range(N))

            for j in range(0, int(N_batch)):
                # indl = int(j * batch_size)
                # indr = int((j+1) * batch_size)
                # ind = ind_s[indl:min(indr, N)]
                # if indr > N:
                #     ind = np.concatenate((ind, ind_s[:(indr-N)]))
                # batch = X[ind]
                batch_label = mnist.train.next_batch(batch_size)
                batch = batch_label[0]

                lowerbound_np, logF_np, mqelbol, mcubo_q, cost_q, mqelbol_q = updateParams(sess, batch, learning_rate)

                #lowerbound_np should be equal to lb_q
                #iteration_lowerbound += lowerbound_np * batch_size
                iteration_mcubo_q += mcubo_q * batch_size
                iteration_lowerbound += cost_q * batch_size
                iteration_mqelbol_q += mqelbol_q * batch_size
            #pdb.set_trace()
            if verbose:
                end = time.time()
                #print("mean_CUBO=",mcubo_q)
                print("Iteration %d, mcubo_q = %.2f, time = %.2fs"
                      % (iteration, iteration_mcubo_q / N, end - begin))
                print("Iteration %d, lowerbound_cost_q = %.2f, time = %.2fs"
                      % (iteration, iteration_lowerbound / N, end - begin))
                print("Iteration %d, lowerbound_mqelbo_q = %.2f, time = %.2fs"
                      % (iteration, iteration_mqelbol_q / N, end - begin))
                print("\n")

                #print("\n ratio=logF_np=",logF_np, "mean_qELBO_loss=",mqelbol)
                
                '''
                print("\n Before optimization of q: mean_CUBO=", mcubo)
                print("\n Before optimization of q: lowerbound=", lowerbound_np)
                print("\n Before optimization of q: mean_qELBO_loss=", mqelbol)
                print("\n After optimization of q: lowerbound_q=", lb_q)
                print("\n After optimization of q: mean_qELBO_loss_q=", mqelbol_q)
                #pdb.set_trace()
                '''                                
                begin = end
                
        
    def eval_test_ll(sess, X, num_samples):
        #lowerbound_np, logF_np, mean_qELBO_loss_np, mean_CUBO_np = sess.run(variational_lowerbound(X, encoder, decoder, num_samples, X.shape[0], 0.0))
        global q0_star
        #print("q0_star=",q0_star)
        #pdb.set_trace()
        
        cost, logF_np, mqelbol, mcubo = sess.run(variational_lowerbound(X, encoder, decoder, num_samples, X.shape[0], 1.0),
                                                 feed_dict={q1_star1: 1.0-q0_star})


        Nq = logF_np.shape[0]
        eq_term = cost + 0.5 * (mcubo - cost)  # these elements do not depend on q1_star
        q0 = 1.0 + 1.0 / eq_term + 1e-10
        var0 = [q0]
        #var_opt3 = minimize(root_qELBO_no_exp, var0, args=(logF_np, eq_term), method='L-BFGS-B',
        #                    bounds=[(1.0 + 1.0 / eq_term, None)], options={'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-10})
        #q0_star = var_opt3.x[0]
        logF_np_reshaped = np.reshape(logF_np, (num_samples, batch_size))

        var_opt3 = minimize(root_qELBO, var0, args=(logF_np_reshaped, eq_term, num_samples), method='L-BFGS-B',
                            bounds=[(1.0 + 1.0 / eq_term, 1.1)], options={'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-10})
        q0_star = var_opt3.x[0]
        
        lowerbound_np, logF_np, mean_qELBO_loss_np, mean_CUBO_np = sess.run(
            variational_lowerbound(X, encoder, decoder, num_samples, X.shape[0], 0.0),
            feed_dict={q1_star1: 1.0-q0_star})
        #lowerbound_np==log_px_IS, mean_qELBO_loss_np=mqelbo

        return lowerbound_np, logF_np, mean_qELBO_loss_np, mean_CUBO_np

    def score(sess, X, num_samples = 100):
        """
        Computer lower bound on data, following the IWAE paper.
        """
        
        begin = time.time()
        print('num. samples for eval:', num_samples)
        
        # compute log_q
        log_px_IS_total = 0
        mcubo_total = 0
        mqelbo_total = 0

        num_data_test = X.shape[0]
        if num_data_test % batch_size == 0:
            num_batch = num_data_test / batch_size
        else:
            num_batch = num_data_test / batch_size + 1
        
        for i in range(int(num_batch)):
            # indl = int(i*batch_size)
            # indr = int(min((i+1)*batch_size, num_data_test))
            # minibatch = X[indl:indr]
            batch_label = mnist.test.next_batch(batch_size)
            minibatch = batch_label[0]

            lowerbound, logF_np, mean_qELBO_loss_np, mean_CUBO_np = eval_test_ll(sess, minibatch, num_samples)
            #lowerbound_total += lowerbound * (indr - indl)
            log_px_IS_total += lowerbound * batch_size
            mcubo_total += mean_CUBO_np * batch_size
            mqelbo_total += mean_qELBO_loss_np * batch_size

            #print("\n Test set: mean_CUBO=", mean_CUBO_np)
            #print("\n Test set: lowerbound=", lowerbound)
            #print("\n Test set: mean_qELBO_loss=", mean_qELBO_loss_np)


        end = time.time()
        time_test = end - begin
        log_px_IS_total = log_px_IS_total / float(num_data_test)
        mcubo_total = mcubo_total / float(num_data_test)
        mqelbo_total = mqelbo_total / float(num_data_test)

        print("\n Test set: mean_CUBO=", mcubo_total)
        print("\n Test set: log_px_IS=", log_px_IS_total)
        print("\n Test set: mean_qELBO_loss=", mqelbo_total)

        return log_px_IS_total, time_test
     
    return fit, score                              
