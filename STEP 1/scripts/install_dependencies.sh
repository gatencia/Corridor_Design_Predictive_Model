# Shell script for additional setup
#!/bin/bash
# Dependency installation script for Elephant Corridor Analysis Project
# Handles system-level dependencies and environment setup

set -e  # Exit on any error

echo "🐘 Elephant Corridor Analysis - Dependency Installation"
echo "======================================================"

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Detect specific Linux distribution
        if [ -f /etc/debian_version ]; then
            echo "ubuntu"
        elif [ -f /etc/redhat-release ]; then
            echo "centos"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install system dependencies on Ubuntu/Debian
install_ubuntu_deps() {
    echo "📦 Installing system dependencies for Ubuntu/Debian..."
    
    # Update package list
    sudo apt update
    
    # Install GDAL, GEOS, PROJ
    sudo apt install -y \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        proj-bin \
        proj-data \
        libspatialindex-dev \
        build-essential \
        python3-dev
    
    # Install R
    if ! command_exists R; then
        echo "📈 Installing R..."
        sudo apt install -y r-base r-base-dev
    fi
    
    # Install additional tools
    sudo apt install -y \
        git \
        curl \
        wget \
        unzip
    
    echo "✅ Ubuntu/Debian system dependencies installed"
}

# Function to install system dependencies on CentOS/RHEL
install_centos_deps() {
    echo "📦 Installing system dependencies for CentOS/RHEL..."
    
    # Enable EPEL repository
    sudo yum install -y epel-release
    
    # Install GDAL, GEOS, PROJ
    sudo yum install -y \
        gdal \
        gdal-devel \
        geos \
        geos-devel \
        proj \
        proj-devel \
        spatialindex-devel \
        gcc \
        gcc-c++ \
        python3-devel
    
    # Install R
    if ! command_exists R; then
        echo "📈 Installing R..."
        sudo yum install -y R
    fi
    
    echo "✅ CentOS/RHEL system dependencies installed"
}

# Function to install system dependencies on macOS
install_macos_deps() {
    echo "📦 Installing system dependencies for macOS..."
    
    # Check if Homebrew is installed
    if ! command_exists brew; then
        echo "🍺 Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies via Homebrew
    brew install \
        gdal \
        geos \
        proj \
        spatialindex \
        r
    
    echo "✅ macOS system dependencies installed"
}

# Function to setup conda environment
setup_conda_env() {
    echo "🐍 Setting up Conda environment..."
    
    if ! command_exists conda; then
        echo "❌ Conda not found. Please install Miniconda or Anaconda first."
        echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
    
    # Create environment from yml file
    if [ -f "environment.yml" ]; then
        echo "📄 Creating environment from environment.yml..."
        conda env create -f environment.yml
        echo "✅ Conda environment 'elephant-corridors' created successfully"
        echo "   Activate with: conda activate elephant-corridors"
    else
        echo "❌ environment.yml not found in current directory"
        return 1
    fi
}

# Function to setup pip environment  
setup_pip_env() {
    echo "🐍 Setting up pip virtual environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        echo "📄 Installing packages from requirements.txt..."
        pip install -r requirements.txt
        echo "✅ Pip virtual environment created successfully"
        echo "   Activate with: source venv/bin/activate"
    else
        echo "❌ requirements.txt not found in current directory"
        return 1
    fi
}

# Function to install R packages
install_r_packages() {
    echo "📈 Installing R packages..."
    
    if ! command_exists R; then
        echo "❌ R not found. Please install R first."
        return 1
    fi
    
    # Install required R packages
    R --slave --no-restore --no-save << 'EOF'
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Install required packages
required_packages <- c("terra", "enerscape")

for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE)) {
        cat(paste("Installing", pkg, "...\n"))
        install.packages(pkg, dependencies = TRUE)
        if (!require(pkg, character.only = TRUE)) {
            stop(paste("Failed to install", pkg))
        }
    } else {
        cat(paste(pkg, "already installed\n"))
    }
}

cat("✅ All R packages installed successfully\n")
EOF
    
    echo "✅ R packages installation completed"
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    if [ -f "scripts/setup_environment.py" ]; then
        echo "🧪 Running environment verification..."
        
        # Try to activate conda environment and run verification
        if conda info --envs | grep -q "elephant-corridors"; then
            echo "   Using conda environment..."
            conda run -n elephant-corridors python scripts/setup_environment.py
        elif [ -f "venv/bin/activate" ]; then
            echo "   Using pip virtual environment..."
            source venv/bin/activate
            python scripts/setup_environment.py
            deactivate
        else
            echo "   Using system Python..."
            python3 scripts/setup_environment.py
        fi
    else
        echo "⚠️  Verification script not found. Manual verification recommended."
    fi
}

# Function to show next steps
show_next_steps() {
    echo ""
    echo "🎉 Installation completed!"
    echo "======================"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Activate your environment:"
    
    if conda info --envs | grep -q "elephant-corridors"; then
        echo "   conda activate elephant-corridors"
    elif [ -f "venv/bin/activate" ]; then
        echo "   source venv/bin/activate"
    fi
    
    echo ""
    echo "2. Verify installation:"
    echo "   python scripts/setup_environment.py"
    echo ""
    echo "3. Start analyzing elephant corridors:"
    echo "   jupyter lab notebooks/exploration.ipynb"
    echo ""
    echo "📖 Documentation: README.md"
    echo "🐛 Issues: https://github.com/yourusername/elephant-corridors/issues"
}

# Main installation function
main() {
    OS=$(detect_os)
    echo "🖥️  Detected OS: $OS"
    echo ""
    
    # Install system dependencies
    case $OS in
        "ubuntu")
            install_ubuntu_deps
            ;;
        "centos")
            install_centos_deps
            ;;
        "macos")
            install_macos_deps
            ;;
        "windows")
            echo "⚠️  Windows detected. Please use WSL or manual installation."
            echo "   See README.md for Windows installation instructions."
            exit 1
            ;;
        *)
            echo "⚠️  Unknown OS. Manual dependency installation required."
            echo "   Please install GDAL, GEOS, PROJ, and R manually."
            ;;
    esac
    
    echo ""
    
    # Setup Python environment
    echo "🐍 Setting up Python environment..."
    if command_exists conda; then
        echo "   Conda detected - using conda environment"
        setup_conda_env
    else
        echo "   Using pip virtual environment"
        setup_pip_env
    fi
    
    echo ""
    
    # Install R packages
    install_r_packages
    
    echo ""
    
    # Verify installation
    verify_installation
    
    echo ""
    
    # Show next steps
    show_next_steps
}

# Parse command line arguments
case "${1:-}" in
    "system")
        echo "Installing system dependencies only..."
        OS=$(detect_os)
        case $OS in
            "ubuntu") install_ubuntu_deps ;;
            "centos") install_centos_deps ;;
            "macos") install_macos_deps ;;
            *) echo "Unsupported OS for system installation" ;;
        esac
        ;;
    "conda")
        echo "Setting up conda environment only..."
        setup_conda_env
        ;;
    "pip")
        echo "Setting up pip environment only..."
        setup_pip_env
        ;;
    "r")
        echo "Installing R packages only..."
        install_r_packages
        ;;
    "verify")
        echo "Running verification only..."
        verify_installation
        ;;
    *)
        main
        ;;
esac