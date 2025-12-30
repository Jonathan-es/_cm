import matplotlib.pyplot as plt

def calculate_compound_interest():
    print("=== Compound Interest Calculator ===")
    
    # 1. Input variables
    try:
        # Principal (P): The money you start with
        P = float(input("Enter Principal (Starting Money): "))
        
        # Rate (r): The percentage the bank gives you (e.g., 5 for 5%)
        r = float(input("Enter Annual Interest Rate (%): ")) / 100
        
        # Time (t): How many years you wait
        t = int(input("Enter number of Years: "))
        
        # n: How many times per year interest is added (12 = monthly)
        n = int(input("Compounds per year (1=Yearly, 12=Monthly): "))
    except ValueError:
        print("Error: Please enter numbers only.")
        return

    # Lists to store data for the graph
    years = list(range(t + 1))
    simple_values = []
    compound_values = []

    print(f"\n{'Year':<5} | {'Simple Interest':<20} | {'Compound Interest':<20}")
    print("-" * 50)

    # 2. Calculation Loop
    for year in years:
        # Simple Interest: Grows in a straight line
        # Formula: A = P(1 + rt)
        amount_simple = P * (1 + r * year)
        
        # Compound Interest: Grows faster (Money makes more money)
        # Formula: A = P(1 + r/n)^(n*t)
        amount_compound = P * (1 + r / n) ** (n * year)
        
        simple_values.append(amount_simple)
        compound_values.append(amount_compound)
        
        print(f"{year:<5} | ${amount_simple:<19.2f} | ${amount_compound:<19.2f}")

    # 3. Final Summary
    final_amount = compound_values[-1]
    profit = final_amount - P
    
    print("-" * 50)
    print(f"Starting Principal: ${P:,.2f}")
    print(f"Final Amount:       ${final_amount:,.2f}")
    print(f"Total Profit:       ${profit:,.2f}")

    # 4. Draw the Graph
    plt.figure(figsize=(10, 6))
    plt.plot(years, simple_values, label='Simple Interest (Linear)', linestyle='--')
    plt.plot(years, compound_values, label='Compound Interest (Exponential)', linewidth=2)
    
    plt.title('Money Growth: Simple vs. Compound')
    plt.xlabel('Years')
    plt.ylabel('Money ($)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    calculate_compound_interest()