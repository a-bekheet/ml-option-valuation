def display_menu():
    """Display the main menu and get user choice."""
    print("\nOption Trading Model - Main Menu")
    print("-" * 50)
    print("1. Train new model")
    print("2. Run existing model")
    print("3. Analyze network architecture")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ")
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            print("Invalid choice. Please enter a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.") 