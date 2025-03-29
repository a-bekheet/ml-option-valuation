# src/utils/menu_utils.py

import logging
from pathlib import Path # Use pathlib

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
            # NOTE: input() works in Colab notebooks but might not be suitable
            # for non-interactive scripts. Consider passing choices via arguments
            # or config for automated runs in Colab.
            choice = input("\nEnter your choice (1-5): ")
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError: # Handle case where input stream might be closed (e.g., running non-interactively)
             print("\nInput stream closed. Cannot get menu choice. Exiting.")
             return 5 # Default to exit choice

def run_application_loop(config, models, handlers, data_utils, visualization_utils, performance_utils):
    """
    Main application loop to handle menu choices.
    Uses paths defined in the config.
    """
    try:
        # Validate paths using paths from config (which should be environment-aware)
        # Assumes validate_paths is updated to handle potential Colab paths correctly
        # and creates directories if they don't exist.
        try:
            # Pass all relevant directory paths from config for validation/creation
            validate_paths(config) # Pass the whole config
            logging.info("Validated and ensured necessary directories exist.")
        except FileNotFoundError as e:
            logging.error(str(e))
            print(f"\nError: Required directory not found: {str(e)}")
            print("Please check the paths defined in your config or mount Google Drive if in Colab.")
            return # Exit if essential paths are missing

        while True:
            try:
                choice = display_menu()

                if choice == 1:
                    # Pass the whole config dict to handlers
                    handlers['handle_train_model'](
                        config,
                        models['HybridRNNModel'], models['GRUGRUModel'], models['LSTMLSTMModel'],
                        visualization_utils['save_and_display_results'], # Assumes this handles paths from config
                        performance_utils['extended_train_model_with_tracking'], # Assumes this handles paths from config
                        data_utils['get_available_tickers'], # Needs data_dir from config
                        data_utils['select_ticker'], # Interactive
                        data_utils['StockOptionDataset'] # Needs data_dir from config
                    )
                elif choice == 2:
                     handlers['handle_run_model'](
                         config, # Pass config instead of models_dir separately
                         models_dir=config['models_dir'], # Keep explicit models_dir for now if needed by signature
                         HybridRNNModel=models['HybridRNNModel'], GRUGRUModel=models['GRUGRUModel'], LSTMLSTMModel=models['LSTMLSTMModel'],
                         get_available_tickers=data_utils['get_available_tickers'],
                         select_ticker=data_utils['select_ticker'],
                         StockOptionDataset=data_utils['StockOptionDataset'],
                         # Ensure the enhanced runner is passed if available/intended
                         run_existing_model_with_visualization=handlers.get('run_existing_model_with_visualization') # Optional pass
                     )
                elif choice == 3:
                     handlers['handle_analyze_architecture'](
                         config,
                         models['HybridRNNModel'], models['GRUGRUModel'], models['LSTMLSTMModel'],
                         visualization_utils['display_model_analysis']
                     )
                elif choice == 4:
                     handlers['handle_benchmark_architectures'](
                         config,
                         models['HybridRNNModel'], models['GRUGRUModel'], models['LSTMLSTMModel'],
                         performance_utils['benchmark_architectures'],
                         performance_utils['generate_architecture_comparison'],
                         performance_utils['visualize_architectures'],
                         data_utils['get_available_tickers'],
                         data_utils['select_ticker'],
                         data_utils['StockOptionDataset']
                     )
                elif choice == 5:
                    print("\nExiting program...")
                    logging.info("Application terminated by user choice")
                    break # Exit the while loop

                # Consider removing this pause for non-interactive runs
                # input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\nOperation cancelled by user. Returning to menu.")
                logging.warning("Operation cancelled by KeyboardInterrupt.")
                continue # Go back to displaying the menu
            except Exception as e:
                logging.exception(f"Error during menu option execution: {str(e)}") # Log full traceback
                print(f"\nAn unexpected error occurred: {str(e)}")
                print("Please check the logs for more details.")
                # Depending on severity, you might want to break or continue
                # break # Example: Exit loop on any error
                continue # Example: Continue to menu after error


    except KeyboardInterrupt:
        print("\nApplication terminated by user (Ctrl+C)")
        logging.info("Application terminated by user (Ctrl+C)")
    except Exception as e:
        logging.exception(f"Critical error during application setup or loop: {str(e)}")
        print(f"\nA critical error occurred: {str(e)}")
    finally:
        logging.info("Application shutdown sequence finished.")


# --- MODIFIED validate_paths ---
def validate_paths(config: dict):
    """
    Validate and create necessary directories based on paths in the config.
    Uses pathlib for robust path handling.
    """
    required_dirs = ['data_dir', 'models_dir', 'performance_logs_dir', 'viz_dir']
    base_paths = {}

    for dir_key in required_dirs:
        dir_path_str = config.get(dir_key)
        if not dir_path_str:
            raise ValueError(f"Configuration missing required directory path: '{dir_key}'")

        dir_path = Path(dir_path_str)
        base_paths[dir_key] = dir_path

        # Special check for data_dir existence
        if dir_key == 'data_dir':
            logging.info(f"Checking data directory: {dir_path}")
            if not dir_path.exists():
                error_msg = (
                    f"Data directory not found: {dir_path}. "
                    f"Ensure the path is correct. If in Colab, check Drive mount and path.")
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)
            # Also check for the essential 'by_ticker' subdirectory structure needed later
            by_ticker_path = dir_path / 'by_ticker'
            if not by_ticker_path.exists():
                logging.warning(f"'by_ticker' subdirectory not found in {dir_path}. Normalization/loading might fail.")
            else:
                logging.info(f"Found 'by_ticker' subdirectory in {dir_path}")

        # Create other directories if they don't exist
        else:
            logging.info(f"Ensuring directory exists: {dir_path} (for {dir_key})")
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                 logging.error(f"Failed to create directory {dir_path}: {e}")
                 raise # Re-raise error if directory creation fails

    # Return the validated Path objects (optional, could modify config in-place)
    # return base_paths