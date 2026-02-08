import matplotlib.pyplot as plt
from train import load_csv_data

def load_model_parameters(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            t0 = float(lines[0].split(':')[1].strip())
            t1 = float(lines[1].split(':')[1].strip())
            return t0, t1
    except (FileNotFoundError, IndexError, ValueError):
        print(f"No model parameters found. Starting with default parameters.")
        return 0.0, 0.0

def predict_price(mileage, t0, t1):
    predicted_price = t0 + (t1 * mileage)
    if predicted_price < 0:
        predicted_price = 0.0
    return max(predicted_price, 0.0)

def plot_data(km, price, t0, t1, user_km=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(km, price, color='blue', label='Real Data')
    x_range = [min(km), max(km)]
    y_range = [predict_price(x, t0, t1) for x in x_range]
    equation_text = f"Price = {t0:.2f} {t1:.4f} * Mileage"
    plt.plot(x_range, y_range, color='red', linewidth=2, label=equation_text)
    start, end = 0, int(max(km)) + 20000
    if user_km is not None:
            user_price = predict_price(user_km, t0, t1)
            plt.scatter(user_km, user_price, color='gold', s=50, edgecolors='black', 
                        zorder=5, label=f'User Input: {user_km} km')      
    plt.xticks(range(start, end, 20000))
    plt.xlim(left=0)
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.title('Price vs Mileage Analysis')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def main():
    try:
        mileage = float(input("Enter the mileage of the car: "))
        if mileage < 0:
            print("Mileage cannot be negative.")
            return
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")
        return
    t0, t1 = load_model_parameters('model_parameters.txt')
    predicted_price = predict_price(mileage, t0, t1)
    print(f"The predicted price for {mileage} km is: ${predicted_price:.2f}")
    try:
        km, price = load_csv_data('data.csv')
        if km and price:
            plot_data(km, price, t0, t1, user_km=mileage)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading data: {exc} - The prediction will be shown without the training data plot.")

if __name__ == "__main__":
    main()