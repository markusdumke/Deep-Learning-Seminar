#' Implementation of RNN (one hidden layer)
#' character-level language model

#' split input into characters and give each character an unique integer id
#' @param x : character string, the input
#' @return output: a list with entries 
#' x_vec: an integer sequence, each number representing a unique sequence element
#' x_char: the input sequence splitted into characters
#' dict: a lookuptable specifying which integer corresponds to which symbol
make_dictionary <- function(x) {
  x_char <- strsplit(x, NULL)[[1]]
  characters <- unique(names(table(x_char)))
  dictionary <- data.frame(characters, seq(1, length(characters)))
  colnames(dictionary) <- c("characters", "integers")
  x_vec <- rep(NA, length(x_char))
  for(i in seq_along(x_vec)) {
    x_vec[i] <- which(dictionary$characters == x_char[i])
  }
  list(x_vec = x_vec, x_char = x_char, dict = dictionary)
}

#' represent characters as one-hot vector, one entry is 1, all other 0
#' @param x : integer vector, the sequence of symbols
#' @inheritParams train_rnn
#' @return a matrix, the input coded as one-hot vector
make_one_hot_coding <- function(x, n_vocab) {
  n_seq <- length(x)
  one_hot <- matrix(0, nrow = n_vocab, ncol = n_seq)
  one_hot[cbind(x, seq(1, n_seq))] <- 1
  one_hot
}

#' Function to return columnwise argmax
#' could be used to sample from RNN if x = o
#' @param x a matrix
#' @return integer vector specifying the indices of the maxima
argmax <- function(x) {
  apply(x, 2, which.max)
}

#' function returning training data (x, y)
#' output sequence is just input sequence shifted one in time
#' @param x: integer vector as specified by dict$x_vec
#' @inheritParams train_rnn
#' @param minibatch: #currently not implemented
#' @return a list with two entries: 
#' x and y, each either an integer vector or a matrix in one_hot_coding
make_train_data <- function(x, one_hot, n_vocab, minibatch) {

  x_train <- x[1:(length(x) - 1)]
  y <- x[2:length(x)] 
  
  if(one_hot == TRUE){
    x_train <- make_one_hot_coding(x_train, n_vocab)
    y <- make_one_hot_coding(y, n_vocab)
  }
  list(x = x_train, y = y)
}

#' Initialize weights to small random numbers
#' @param seed: set random seed
#' @inheritParams train_rnn
#' @return list with entries U, V, W, b, c
intialize_weights <- function(seed, n_hidden, n_vocab) {
  set.seed(seed)
  U <- matrix(runif(n_hidden * n_vocab, - 0.1, 0.1), ncol = n_vocab)
  V <- matrix(runif(n_hidden * n_vocab, - 0.1, 0.1), ncol = n_hidden)
  W <- matrix(runif(n_hidden * n_hidden, - 0.1, 0.1), ncol = n_hidden)
  b <- runif(n_hidden, - 0.1, 0.1)
  c <- runif(n_vocab, - 0.1, 0.1)
  list(U = U, V = V, W = W, b = b, c = c)
}

#' Compute softmax function
#' @param x: a numeric vector
softmax <- function(x) {
  exp(x) / sum(exp(x))
}

#' Forward Propagation, compute hidden state and output for each time step
#' multiplication with one-hot vector is equal to indexing with integer
#' @inheritParams train_rnn
#' @return list with hidden states h and output o
rnn_forward <- function(x, weights, n_hidden, n_vocab, one_hot) {
  U <- weights$U
  V <- weights$V
  W <- weights$W
  b <- weights$b
  c <- weights$c
  n_seq <- ifelse(is.matrix(x) == TRUE, ncol(x), length(x))
  h <- matrix(0, nrow = n_hidden, ncol = n_seq) 
  o <- matrix(0, nrow = n_vocab, ncol = n_seq)
  if(one_hot == TRUE){
    h[, 1] <- tanh(as.vector(U %*% x[, 1] + b)) # initialize h[, 0] = 0
    if(n_seq > 1){
      for(t in seq(2, n_seq)) {
        h[, t] <- tanh(W %*% h[, t - 1] + U %*% x[, t] + b)
        o[, t] <- softmax(V %*% h[, t] + c)
      }
    }
  } else{
    h[, 1] <- tanh(U[, x[1]] + b)
    if(n_seq > 1){
      for(t in seq(2, n_seq)) {
        h[, t] <- tanh(W %*% h[, t - 1] + U[, x[t]] + b)
        o[, t] <- softmax(V %*% h[, t] + c)
      }
    }
  }
  o[, 1] <- softmax(V %*% h[, 1] + c)

  list(h = h, o = o)
}

#' Computing gradients
#' @inheritParams train_rnn
#' @inheritParams rnn_backward
#' @return list with computed gradients for all weights
calculate_gradients <- function(o, h, x, y, weights, one_hot, n_vocab) {
  n_hidden <- nrow(h)
  n_seq <- ifelse(is.matrix(x) == TRUE, ncol(x), length(x))
  V <- weights$V
  W <- weights$W
  
  if(one_hot == TRUE){
    grad_o <- o - y
  } else {
    grad_o <- o
    ind <- matrix(c(y, seq_along(y)), ncol = 2)
    grad_o[ind] <- grad_o[ind] - 1
  }
  
  grad_c <- rep(0, n_vocab)
  grad_b <- rep(0, n_hidden)
  grad_W <- matrix(0, nrow = n_hidden, ncol = n_hidden)
  grad_V <- matrix(0, nrow = n_vocab, ncol = n_hidden)
  grad_U <- matrix(0, nrow = n_hidden, ncol = n_vocab)
  grad_h <- matrix(0, nrow = n_hidden, ncol = n_seq)
  grad_h[, n_seq] <- t(V) %*% grad_o[, n_seq]
    
  for(t in seq((n_seq - 1), 1)) {
    grad_h[, t] <- t(W) %*% grad_h[, t + 1] * (1 - h[, t + 1]^2) + t(V) %*% grad_o[, t]
  }
  
  if(n_seq > 1){
    for(t in seq(n_seq, 1)) {
      grad_U <- grad_U # + diag(1 - h[, t]^2) %*% grad_h[, t] %*% t(x[, t])
      grad_V <- grad_V + grad_o[, t] %*% t(h[, t])
      grad_b <- grad_b # + diag(1 - h[, t]^2) %*% grad_h[, t]
      grad_c <- grad_c + grad_o[, t]
    }  
    for(t in seq(n_seq, 2)) {
    grad_W <- grad_W + diag(1 - h[, t]^2) %*% grad_h[, t] %*% t(h[, t - 1]) # false?, loss not decreasing
    }
  }
  
  list(U = grad_U, V = grad_V, W = grad_W, b = grad_b, c = grad_c)
}

#' Cross entropy loss for multinoulli distribution
#' @inheritParams train_rnn
#' @inheritParams rnn_backward
#' @return scalar, the loss between o and y
loss <- function(o, y, one_hot, n_vocab) {
  if(one_hot == FALSE){
    y <- make_one_hot_coding(y, n_vocab)
  }
  - 1 / ncol(o) * sum(diag(t(y) %*% log(o)))
}

# Empirical check, if gradients are correct
# check_gradients <- function(x, y, weights, n_hidden, epsilon = 0.001) {
#   forward <- rnn_forward(x, weights, n_hidden)
#   o <- forward$o
#   h <- forward$h
#   backprop_gradient <- calculate_gradients(o, h, x, y, weights, one_hot, n_vocab)
#   weights$W[1] <- weights$W[1] + epsilon
#   o <- rnn_forward(x, weights, n_hidden)$o
#   loss_1 <- loss(o, y)
#   weights$W[1] <- weights$W[1] - 2 * epsilon
#   o <- rnn_forward(x, weights, n_hidden)$o
#   loss_2 <- loss(o, y)
#   est_gradient <- (loss_1 - loss_2) / (2 * epsilon)
# }

# Stochastic? Gradient Descent update
#' @inheritParams train_rnn
#' @return list with updated weights
sgd_update <- function(learning_rate, weights, gradients) {
  weights$U <- weights$U - learning_rate * gradients$U
  weights$V <- weights$V - learning_rate * gradients$V
  weights$W <- weights$W - learning_rate * gradients$W
  weights$b <- weights$b - learning_rate * gradients$b
  weights$c <- weights$c - learning_rate * gradients$c
  weights
}

#' Back propagation through time
#' @inheritParams train_rnn
#' @param o: matrix, outputs computed by rnn_forward
#' @param h: matrix, hidden states computed by rnn_forward
#' @return a list with two entries:
#' calculated loss and updated weights
rnn_backward <- function(learning_rate, o, h, x, y, weights, one_hot, n_vocab) {
  loss <- loss(o, y, one_hot, n_vocab)
  gradients <- calculate_gradients(o, h, x, y, weights, one_hot, n_vocab)
  
  weights <- sgd_update(learning_rate, weights, gradients)
  list(loss = loss, weights = weights)
}

#' Train full RNN model
#' @param steps: number of training iterations
#' @param learning_rate: scalar, the learning rate for the Gradient Descent update
#' @param x: either matrix in one-hot coding or vector of integers
#' @param y: either matrix in one_hot coding or integer vector, training targets
#' @param weights: list with entries U, V, W, b, c
#' @param n_hidden: integer, size of the hidden state
#' @param n_vocab: only needed if one_hot == FALSE
#' @param one_hot: logical, is y one-hot coded?
#' @return 
train_rnn <- function(steps, learning_rate, x, y, weights, n_hidden, 
                      n_vocab, one_hot = FALSE) {
  loss <- rep(0, steps)
  for(i in seq_len(steps)) {
    forward <- rnn_forward(x, weights, n_hidden, n_vocab, one_hot)
    h <- forward$h
    o <- forward$o
    bptt <- rnn_backward(learning_rate, o, h, x, y, weights, one_hot, n_vocab)
    weights <- bptt$weights
    loss[i] <- bptt$loss
  }
  list(loss = loss, weights = weights)
}

#---------------------------------------------------------
#' Sample from RNN
#' @inheritParams generate_text
#' @inheritParams rnn_backward
#' @return integer, the next predicted sequence symbol
predict_rnn <- function(o, sample) {
  if(sample == FALSE){
    int <- argmax(o)
  } else {
    int <- sample(size = 1, x = seq(nrow(o)), prob = o)
  }
  int
}

#' Generate new text by returning the next character with highest probability
#' @param len: length of output sequence
#' @inheritParams train_rnn
#' @param dict: dictionary
#' @param initial_character: char of length 1
#' @param sample: logical, sampling from output distribution?
#' If FALSE, always most likely character is returned
#' @return sequence of characters sampled from the RNN model
generate_text <- function(len, weights, dict, initial_character = FALSE, sample = TRUE) {
  if(initial_character == FALSE){
    initial_character <- sample(size = 1, levels(dict$dict$characters))
  }
  int <- rep(NA, len)
  int[1] <- which(initial_character == dict$dict$characters)
  n_vocab <- nrow(dict$dict)
  n_hidden <- length(weights$b)
  
  for(i in seq_len(len - 1)){
    forward <- rnn_forward(x = int[i], weights, n_hidden, n_vocab, one_hot = FALSE)
    int[i + 1] <- predict_rnn(forward$o, sample = sample)
  }
  dict$dict$characters[int]
}