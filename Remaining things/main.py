import sys
import os

# Add the current directory to path so we can import experiments
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments import pixel_relationships
from experiments import connected_components
from experiments import distance_measures
from experiments import sampling_quantization_extended
from experiments import image_statistics

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    while True:
        clear_screen()
        print("=========================================")
        print("   DIP Chapter 2 Experiments Workbench   ")
        print("=========================================")
        print("1. Pixel Relationships (Connectivity & Boundaries) [Topic A]")
        print("2. Connected Components (Labeling) [Topic A]")
        print("3. Distance Measures (Metrics & Transforms) [Topic B]")
        print("4. Sampling and Quantization (Resampling & Gray Levels) [Topic C]")
        print("5. Basic Image Statistics (Stats, Hist, Contrast) [Topic D]")
        print("0. Exit")
        print("=========================================")
        
        choice = input("Enter your choice (0-5): ")
        
        if choice == '1':
            pixel_relationships.run_experiment()
        elif choice == '2':
            connected_components.run_experiment()
        elif choice == '3':
            distance_measures.run_experiment()
        elif choice == '4':
            sampling_quantization_extended.run_experiment()
        elif choice == '5':
            image_statistics.run_experiment()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main_menu()
