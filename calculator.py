# Module 2 - Python Mini Project: Simple Calculator

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return "Error: Cannot divide by zero"
    return a / b

def calculator():
    print("===== Simple Calculator =====")
    print("Operations: + , - , * , /")

    while True:
        try:
            num1 = float(input("\nEnter first number: "))
            operator = input("Enter operator (+, -, *, /): ")
            num2 = float(input("Enter second number: "))

            if operator == "+":
                result = add(num1, num2)
            elif operator == "-":
                result = subtract(num1, num2)
            elif operator == "*":
                result = multiply(num1, num2)
            elif operator == "/":
                result = divide(num1, num2)
            else:
                result = "Invalid operator"

            print("Result:", result)

        except ValueError:
            print("Invalid input. Please enter numeric values.")

        again = input("\nDo you want to calculate again? (yes/no): ")
        if again.lower() != "yes":
            print("Thank you for using the calculator!")
            break

# Run the calculator
calculator()