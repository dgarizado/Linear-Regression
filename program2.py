import csv
import math

def load_csv_data(file_path):
    km, price = [], []
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                km.append(float(row['km']))
                price.append(float(row['price']))
    except FileNotFoundError:
        pass
    return km, price

def train(km, price, learning_rate=0.01, epochs=1000):
    m = len(km)
    km_mean = sum(km) / m
    km_std_dev = math.sqrt(sum((x - km_mean) ** 2 for x in km) / m)
    km_normalized = [(x - km_mean) / km_std_dev for x in km]
    theta0 = 0.0
    theta1 = 0.0
    for _ in range(epochs):
        sum_error0 = 0.0
        sum_error1 = 0.0
        for i in range(m):
            prediction = theta0 + theta1 * km_normalized[i]
            error = prediction - price[i]
            sum_error0 += error
            sum_error1 += error * km_normalized[i]
        tmp_theta0 = learning_rate * (sum_error0 / m)
        tmp_theta1 = learning_rate * (sum_error1 / m)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    final_theta1 = theta1 / km_std_dev
    final_theta0 = theta0 - (final_theta1 * km_mean)
    return final_theta0, final_theta1

def save_parameters(file_path, theta0, theta1):
    with open(file_path, 'w') as file:
        file.write(f"theta0: {theta0}\ntheta1: {theta1}\n")

def main():
    km, price = load_csv_data('data.csv')
    if not km:
        return
    print("Training model...")
    theta0, theta1 = train(km, price)
    print("Model successfully trained.")
    print(f"Final parameters: theta0 = {theta0}, theta1 = {theta1}")
    save_parameters('model_parameters.txt', theta0, theta1)

if __name__ == "__main__":
    main()