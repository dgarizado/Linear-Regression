# Linear-Regression
42 project aiming to learn basics of Machine Learning. This project uses Gradient Descent Algorithm in order to aproximate a linear equation that describes the price of a car based on the mileage.


# Gradient Descent 

Having the hipothesis function defined as:
h(x) = t0 + t1 * x

The cost function is defined as:
J(t0, t1) = (1/2m) * sum((h(xi) - yi)^2)  <-- Mean Squared Error 

where m is the number of training examples, xi is the mileage of the i-th car and yi is the price of the i-th car.

The parameters t0 and t1 are updated iteratively using the following formulas:
t0 = t0 - alpha * (1/m) * sum(h(xi) - yi)
t1 = t1 - alpha * (1/m) * sum((h(xi) - yi) * xi)

why t0 -= alpha * (1/m) * sum(h(xi) - yi) and not t0 += alpha * (1/m) * sum(h(xi) - yi)?
The reason we use t0 -= alpha * (1/m) * sum(h(xi) - yi) instead of t0 += alpha * (1/m) * sum(h(xi) - yi) is because we want to minimize the cost function J(t0, t1).

We obtain the gradient of the cost function with respect to t0 and t1, which gives us the direction of steepest ascent. To minimize the cost function, we need to move in the opposite direction of the gradient, which is why we subtract the gradient from the current values of t0 and t1.

 - Obtaining the gradient of the cost function with respect to t0 and t1:
   - dJ/dt0 = (1/m) * sum(h(xi) - yi)
   - dJ/dt1 = (1/m) * sum((h(xi) - yi) * xi)