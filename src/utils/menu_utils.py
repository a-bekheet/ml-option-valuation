import logging
from pathlib import Path

def display_menu():
    """Display the main menu and get user choice."""
    print("\nOption Trading Model - Main Menu")
    print("-" * 50)
    print("1. Train new model")
    print("2. Run existing model")
    print("3. Analyze network architecture")
    print("4. Benchmark multiple architectures")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ")
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def run_application_loop(config, models, handlers, data_utils, visualization_utils, performance_utils):
    """
    Main application loop to handle menu choices.
    
    Args:
        config: Application configuration
        models: Dictionary containing model classes
        handlers: Dictionary containing handler functions
        data_utils: Dictionary containing data utility functions
        visualization_utils: Dictionary containing visualization utilities
        performance_utils: Dictionary containing performance utilities
    """
    try:
        # Validate paths
        try:
            data_path, models_dir = data_utils['validate_paths'](config)
            # Ensure performance logs directory exists
            Path(config['performance_logs_dir']).mkdir(parents=True, exist_ok=True)
        except FileNotFoundError as e:
            logging.error(str(e))
            print(f"\nError: {str(e)}")
            return
        
        while True:
            try:
                # Use extended menu with benchmarking option
                choice = display_menu()
                
                if choice == 1:
                    handlers['handle_train_model'](
                        config, 
                        models['HybridRNNModel'], 
                        models['GRUGRUModel'], 
                        models['LSTMLSTMModel'], 
                        visualization_utils['save_and_display_results'], 
                        performance_utils['extended_train_model_with_tracking'],
                        data_utils['get_available_tickers'],
                        data_utils['select_ticker'],
                        data_utils['StockOptionDataset']
                    )
                elif choice == 2:
                    handlers['handle_run_model'](
                        config, 
                        config['models_dir'], 
                        models['HybridRNNModel'], 
                        models['GRUGRUModel'], 
                        models['LSTMLSTMModel'],
                        data_utils['get_available_tickers'],
                        data_utils['select_ticker'],
                        data_utils['StockOptionDataset']
                    )
                elif choice == 3:
                    handlers['handle_analyze_architecture'](
                        config, 
                        models['HybridRNNModel'], 
                        models['GRUGRUModel'], 
                        models['LSTMLSTMModel'],
                        visualization_utils['display_model_analysis']
                    )
                elif choice == 4:
                    handlers['handle_benchmark_architectures'](
                        config, 
                        models['HybridRNNModel'], 
                        models['GRUGRUModel'], 
                        models['LSTMLSTMModel'],
                        performance_utils['benchmark_architectures'],
                        performance_utils['generate_architecture_comparison'],
                        performance_utils['visualize_architectures'],
                        data_utils['get_available_tickers'],
                        data_utils['select_ticker'],
                        data_utils['StockOptionDataset']
                    )
                elif choice == 5:
                    print("\nExiting program...")
                    logging.info("Application terminated by user")
                    break
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                continue
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                print(f"\nAn unexpected error occurred: {str(e)}")
                continue
    
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        logging.info("Application terminated by user")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        print(f"\nA critical error occurred: {str(e)}")
    finally:
        logging.info("Application shutdown")