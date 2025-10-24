#!/bin/bash
# MIMIC-IV-Ext Benchmark Launcher
# Quick launcher for common benchmark tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo ""
echo "======================================================================"
echo "  MIMIC-IV-Ext Benchmark Suite"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Function to show menu
show_menu() {
    echo -e "${BLUE}Select an option:${NC}"
    echo ""
    echo "  1) Quick Test (10 cases, ~2 minutes)"
    echo "  2) Run Full Benchmark (all cases, 8-10 hours)"
    echo "  3) Run Subset (specify number of cases)"
    echo "  4) Evaluate Latest Results"
    echo "  5) Evaluate Specific Run"
    echo "  6) Compare Multiple Runs"
    echo "  7) List All Runs"
    echo "  8) Exit"
    echo ""
}

# Function to run quick test
quick_test() {
    echo -e "${GREEN}Running quick test (10 cases)...${NC}"
    read -p "Model name (default: local-model): " model_name
    model_name=${model_name:-local-model}
    
    python3 run_benchmark.py --test --model-name "$model_name"
    
    if [ $? -eq 0 ]; then
        echo ""
        read -p "Evaluate results now? (y/n): " evaluate
        if [[ "$evaluate" == "y" || "$evaluate" == "Y" ]]; then
            python3 evaluate_results.py --latest
        fi
    fi
}

# Function to run full benchmark
full_benchmark() {
    echo -e "${YELLOW}WARNING: This will take 8-10 hours!${NC}"
    read -p "Are you sure? (y/n): " confirm
    
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Cancelled."
        return
    fi
    
    read -p "Model name (default: local-model): " model_name
    model_name=${model_name:-local-model}
    
    echo -e "${GREEN}Starting full benchmark...${NC}"
    python3 run_benchmark.py --model-name "$model_name"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Benchmark completed!${NC}"
        read -p "Evaluate results now? (y/n): " evaluate
        if [[ "$evaluate" == "y" || "$evaluate" == "Y" ]]; then
            python3 evaluate_results.py --latest
        fi
    fi
}

# Function to run subset
subset_benchmark() {
    read -p "Number of cases to run: " num_cases
    
    if ! [[ "$num_cases" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}Invalid number${NC}"
        return
    fi
    
    read -p "Model name (default: local-model): " model_name
    model_name=${model_name:-local-model}
    
    echo -e "${GREEN}Running benchmark on $num_cases cases...${NC}"
    python3 run_benchmark.py --num-cases "$num_cases" --model-name "$model_name"
    
    if [ $? -eq 0 ]; then
        echo ""
        read -p "Evaluate results now? (y/n): " evaluate
        if [[ "$evaluate" == "y" || "$evaluate" == "Y" ]]; then
            python3 evaluate_results.py --latest
        fi
    fi
}

# Function to evaluate latest
evaluate_latest() {
    echo -e "${GREEN}Evaluating latest results...${NC}"
    python3 evaluate_results.py --latest
}

# Function to evaluate specific run
evaluate_specific() {
    echo ""
    echo "Available runs:"
    python3 compare_runs.py --list
    echo ""
    read -p "Enter run ID: " run_id
    
    if [ -z "$run_id" ]; then
        echo -e "${RED}No run ID provided${NC}"
        return
    fi
    
    python3 evaluate_results.py "$run_id"
}

# Function to compare runs
compare_runs() {
    echo ""
    echo "Available runs:"
    python3 compare_runs.py --list
    echo ""
    read -p "Compare all runs? (y/n): " compare_all
    
    if [[ "$compare_all" == "y" || "$compare_all" == "Y" ]]; then
        python3 compare_runs.py --all
    else
        read -p "Enter run IDs (space-separated): " run_ids
        if [ -z "$run_ids" ]; then
            echo -e "${RED}No run IDs provided${NC}"
            return
        fi
        python3 compare_runs.py $run_ids
    fi
}

# Function to list runs
list_runs() {
    python3 compare_runs.py --list
}

# Main loop
while true; do
    show_menu
    read -p "Choice: " choice
    echo ""
    
    case $choice in
        1)
            quick_test
            ;;
        2)
            full_benchmark
            ;;
        3)
            subset_benchmark
            ;;
        4)
            evaluate_latest
            ;;
        5)
            evaluate_specific
            ;;
        6)
            compare_runs
            ;;
        7)
            list_runs
            ;;
        8)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    echo ""
done
