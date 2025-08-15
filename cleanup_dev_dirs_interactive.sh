#!/bin/bash

# Script to find and delete venv, .venv, and node_modules directories
# Interactive version - allows selecting which directories to delete

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Starting directory (current directory by default)
START_DIR="${1:-.}"

# Patterns to exclude (system package managers and tools)
EXCLUDE_PATTERNS=(
    "*/bun/install/*"
    "*/.bun/*"
    "*/.nvm/*"
    "*/.npm/*"
    "*/.pnpm/*"
    "*/.yarn/*"
    "*/.cargo/*"
    "*/.rustup/*"
    "*/.pyenv/*"
    "*/.rbenv/*"
    "*/.asdf/*"
    "*/Library/Caches/*"
    "*/AppData/*"
    "*/.cache/*"
    "*/homebrew/*"
    "*/.local/share/*"
)

echo -e "${YELLOW}Searching for venv, .venv, and node_modules directories in: $START_DIR${NC}"
echo -e "${CYAN}Excluding system package managers (bun, nvm, npm, etc.)${NC}"
echo ""

# Build find command with exclusions
FIND_CMD="find \"$START_DIR\""
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    FIND_CMD="$FIND_CMD -path \"$pattern\" -prune -o"
done

# Find all matching directories
echo -e "${GREEN}Finding directories...${NC}"
venv_dirs=$(eval "$FIND_CMD -type d -name 'venv' -print 2>/dev/null")
dot_venv_dirs=$(eval "$FIND_CMD -type d -name '.venv' -print 2>/dev/null")
node_dirs=$(eval "$FIND_CMD -type d -name 'node_modules' -print 2>/dev/null")

# Combine all directories
all_dirs=$(echo -e "$venv_dirs\n$dot_venv_dirs\n$node_dirs" | grep -v '^$' | sort -u)

if [ -z "$all_dirs" ]; then
    echo -e "${GREEN}No venv, .venv, or node_modules directories found.${NC}"
    exit 0
fi

# Store directories in array with numbers
readarray -t dir_array <<< "$all_dirs"
total_count=${#dir_array[@]}

echo -e "\n${YELLOW}Found $total_count directories:${NC}"
echo "========================================"

# Display directories with numbers and sizes
declare -a dir_sizes
for i in "${!dir_array[@]}"; do
    dir="${dir_array[$i]}"
    if [ -d "$dir" ]; then
        # Get size of directory
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        size_kb=$(du -sk "$dir" 2>/dev/null | cut -f1)
        dir_sizes[$i]=$size_kb
        
        # Show relative path if possible
        rel_path=$(realpath --relative-to="$START_DIR" "$dir" 2>/dev/null || echo "$dir")
        printf "  ${CYAN}[%2d]${NC} ${RED}%-60s${NC} ${GREEN}(%s)${NC}\n" $((i+1)) "$rel_path" "$size"
    fi
done

echo "========================================"

# Calculate total size
total_size=0
for size in "${dir_sizes[@]}"; do
    total_size=$((total_size + size))
done

# Convert total size to human readable
total_size_human=$(echo "$total_size" | awk '{
    if ($1 >= 1048576) printf "%.2f GB", $1/1048576
    else if ($1 >= 1024) printf "%.2f MB", $1/1024
    else printf "%d KB", $1
}')

echo -e "${YELLOW}Total space in all directories: $total_size_human${NC}\n"

# Interactive selection
echo -e "${CYAN}Select directories to delete:${NC}"
echo "  - Enter numbers separated by spaces (e.g., '1 3 5')"
echo "  - Enter ranges with dash (e.g., '1-5' or '3-7')"
echo "  - Enter 'all' to select everything"
echo "  - Enter 'none' or just press Enter to cancel"
echo ""
read -p "Your selection: " selection

# Process selection
if [[ -z "$selection" ]] || [[ "$selection" == "none" ]]; then
    echo -e "\n${YELLOW}Operation cancelled. No directories were deleted.${NC}"
    exit 0
fi

# Parse selection into array of indices
declare -a selected_indices

if [[ "$selection" == "all" ]]; then
    # Select all directories
    for i in "${!dir_array[@]}"; do
        selected_indices+=($i)
    done
else
    # Parse the selection
    for item in $selection; do
        if [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            # Range like "1-5"
            start=$((${BASH_REMATCH[1]} - 1))
            end=$((${BASH_REMATCH[2]} - 1))
            for ((j=start; j<=end && j<total_count; j++)); do
                selected_indices+=($j)
            done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            # Single number
            idx=$((item - 1))
            if [ $idx -ge 0 ] && [ $idx -lt $total_count ]; then
                selected_indices+=($idx)
            fi
        fi
    done
fi

# Remove duplicates and sort
selected_indices=($(printf "%s\n" "${selected_indices[@]}" | sort -nu))

if [ ${#selected_indices[@]} -eq 0 ]; then
    echo -e "\n${YELLOW}No valid directories selected. Operation cancelled.${NC}"
    exit 0
fi

# Show what will be deleted
echo -e "\n${YELLOW}Will delete ${#selected_indices[@]} directories:${NC}"
selected_size=0
for idx in "${selected_indices[@]}"; do
    dir="${dir_array[$idx]}"
    size_kb="${dir_sizes[$idx]}"
    selected_size=$((selected_size + size_kb))
    rel_path=$(realpath --relative-to="$START_DIR" "$dir" 2>/dev/null || echo "$dir")
    echo -e "  ${RED}$rel_path${NC}"
done

# Convert selected size to human readable
selected_size_human=$(echo "$selected_size" | awk '{
    if ($1 >= 1048576) printf "%.2f GB", $1/1048576
    else if ($1 >= 1024) printf "%.2f MB", $1/1024
    else printf "%d KB", $1
}')

echo -e "\n${YELLOW}Total space to be freed: $selected_size_human${NC}"

# Final confirmation
read -p "
Confirm deletion? (yes/no): " confirm

if [[ "$confirm" == "yes" ]] || [[ "$confirm" == "y" ]]; then
    echo -e "\n${RED}Deleting selected directories...${NC}"
    
    deleted_count=0
    failed_count=0
    
    for idx in "${selected_indices[@]}"; do
        dir="${dir_array[$idx]}"
        if [ -d "$dir" ]; then
            rel_path=$(realpath --relative-to="$START_DIR" "$dir" 2>/dev/null || echo "$dir")
            echo -n "Deleting: $rel_path ... "
            if rm -rf "$dir" 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
                ((deleted_count++))
            else
                echo -e "${RED}✗ (failed)${NC}"
                ((failed_count++))
            fi
        fi
    done
    
    echo ""
    echo -e "${GREEN}Successfully deleted $deleted_count directories${NC}"
    if [ $failed_count -gt 0 ]; then
        echo -e "${RED}Failed to delete $failed_count directories${NC}"
    fi
    echo -e "${GREEN}Freed approximately $selected_size_human of disk space${NC}"
else
    echo -e "\n${YELLOW}Operation cancelled. No directories were deleted.${NC}"
fi