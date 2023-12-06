
# by gpt
# Create the array
array <- c(
  seq(from = 1, to = 49, by = 6),
  seq(from = 2, to = 50, by = 6),
  seq(from = 3, to = 51, by = 6),
  seq(from = 4, to = 52, by = 6),
  seq(from = 5, to = 53, by = 6),
  seq(from = 6, to = 54, by = 6)
)

# Print the array
print(array)

set.seed(20231215)


# Generate random order
random_order <- sample(1:36)

# Print the random order
print(random_order)

array[random_order]

cat(array[random_order], sep = '\n')





