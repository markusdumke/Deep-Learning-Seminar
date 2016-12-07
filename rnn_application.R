#---------------------------------------------------
# Application
source("rnn_functions.R")

input <- "The rules of the game have changed."

# read Obama speeches txt file
fi   <- file("data/obama.txt", "r")
obama <- paste(readLines(fi), collapse="\n")
close(fi)
obama <- gsub(pattern = "\n", replacement = "", x = obama)
input <- strsplit(obama, NULL)[[1]][1:10000]  # use only first 10000 characters
input <- paste0(input, collapse = "")

dict <- make_dictionary(input)
train <- make_train_data(x = dict$x_vec, one_hot = TRUE, n_vocab = nrow(dict$dict))
train <- make_train_data(x = dict$x_vec, one_hot = FALSE)
x <- train$x
y <- train$y

weights <- intialize_weights(seed = 281116, n_hidden = 10, n_vocab = nrow(dict$dict))
model <- train_rnn(2, 0.1, x, y, weights, n_hidden = 10, 
                   n_vocab = nrow(dict$dict), one_hot = FALSE)
weights <- model$weights
plot(model$loss, type = "l")
# Backpropagation formulas wrong? Loss not decreasing


new_text <- generate_text(100, weights, dict, initial_character = "T", sample = TRUE)