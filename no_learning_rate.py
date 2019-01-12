import numpy as np
import tensorflow as tf

    self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # set initial feedforward and feedback weights
        p = self.config.state_size[0]
        #m = 512
        #j = 200
        m = 50
        j = 20
        n = 10
        var_xi = self.config.var_xi
        gamma = self.config.gamma

        #Scale weight initialization
        alpha0 = np.sqrt(2.0/p)
        alpha1 = np.sqrt(2.0/m)
        alpha2 = np.sqrt(2.0/j)
        alpha3 = 1

        #Plus one for bias terms
        A = tf.Variable(rng.randn(p+1,m)*alpha0, name="hidden_weights", dtype=tf.float32)
        W1 = tf.Variable(rng.randn(m+1,j)*alpha1, name="hidden_weights2", dtype=tf.float32)
        W2 = tf.Variable(rng.randn(j+1,n)*alpha2, name="output_weights", dtype=tf.float32)

        V1 = tf.Variable(gamma*np.eye(j), dtype=tf.float32)
        V2 = tf.Variable(gamma*np.eye(n), dtype=tf.float32)
        S1 = tf.Variable(np.zeros((m+1,j)), dtype=tf.float32)
        S2 = tf.Variable(np.zeros((j+1,n)), dtype=tf.float32)

        #The exact least squares solution for synth grad estimation
        B1 = tf.matmul(S1, tf.matrix_inverse(V1))
        B2 = tf.matmul(S2, tf.matrix_inverse(V2))

        # network architecture with ones added for bias terms
        e0 = tf.ones([self.config.batch_size, 1], tf.float32)
        e1 = tf.ones([self.config.batch_size, 1], tf.float32)
        x_aug = tf.concat([self.x, e0], 1)
        h1 = tf.sigmoid(tf.matmul(x_aug, A))
        h1_aug = tf.concat([h1, e1], 1)
        xi1 = tf.random_normal(shape=tf.shape(h1_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h1_tilde = h1_aug + xi1
        h2 = tf.sigmoid(tf.matmul(h1_tilde, W1))
        h2_aug = tf.concat([h2, e1], 1)
        xi2 = tf.random_normal(shape=tf.shape(h2_aug), mean=0.0, stddev=var_xi, dtype=tf.float32)
        h2_tilde = h2_aug + xi2
        y_p = tf.matmul(h2_tilde, W2)

        #Compute unperturbed output
        h2_0 = tf.sigmoid(tf.matmul(h1_aug, W1))
        h2_0_aug = tf.concat([h2_0, e1], 1)
        y_p_0 = tf.matmul(h2_0_aug, W2)

        self.trainable = [A, W1, W2, V1, V2, S1, S2]

        with tf.name_scope("loss"):
            #mean squared error
            self.loss_p = tf.reduce_sum(tf.pow(y_p-self.y, 2))/2
            self.loss = tf.reduce_sum(tf.pow(y_p_0-self.y, 2))/2
            e = (y_p_0 - self.y)
            h1_prime_0 = tf.multiply(h1_aug, 1-h1_aug)[:,0:m]
            h2_prime_0 = tf.multiply(h2_0_aug, 1-h2_0_aug)[:,0:j]

            #Compute updates for W and A (based on B)
            grad_W2 = tf.gradients(xs=W2, ys=self.loss)[0]
            lmda2 = tf.matmul(e, tf.transpose(B2[0:j,:]))
            d2 = np.multiply(h2_prime_0, lmda2)
            grad_W1 = tf.matmul(tf.transpose(h1_aug), d2)
            lmda1 = tf.matmul(d2, tf.transpose(B1[0:m,:]))
            d1 = np.multiply(h1_prime_0, lmda1)
            grad_A = tf.matmul(tf.transpose(x_aug), d1)

            np_est1 = tf.transpose(xi1)*(self.loss_p - self.loss)/var_xi/var_xi
            np_est2 = tf.transpose(xi2)*(self.loss_p - self.loss)/var_xi/var_xi
            #grad_B1 = tf.matmul(tf.matmul(B1, tf.transpose(d2)) - np_est1, d2)
            #grad_B2 = tf.matmul(tf.matmul(B2, tf.transpose(e)) - np_est2, e)

            new_W1 = W1.assign(W1 - self.config.learning_rate*grad_W1)
            new_W2 = W2.assign(W2 - self.config.learning_rate*grad_W2)
            new_A = A.assign(A - self.config.learning_rate*grad_A)

            #Train with exact least squares solution
            grad_V1 = tf.matmul(tf.transpose(d2), d2)
            grad_V2 = tf.matmul(tf.transpose(e), e)
            grad_S1 = tf.matmul(np_est1, d2)
            grad_S2 = tf.matmul(np_est2, e)

            #Update V and S
            new_V1 = V1.assign(V1 + grad_V1)
            new_V2 = V2.assign(V2 + grad_V2)            
            new_S1 = S1.assign(S1 + grad_S1)
            new_S2 = S2.assign(S2 + grad_S2)            
            self.train_step = [new_W1, new_A, new_V1, new_S1, new_W2, new_V2, new_S2]

            correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))