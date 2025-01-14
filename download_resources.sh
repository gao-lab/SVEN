#!/bin/sh

# init
mode=fast
while getopts "m:" opt; do
    case $opt in
        m)
        if [ "$OPTARG" = "fast" ] || [ "$OPTARG" = "full" ]; then
            mode=$OPTARG
        else
            echo "Invalid mode: $OPTARG" >&2
            exit 1
        fi
        ;;
        \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1
        ;;
    esac
    done

# download the resources
echo "Downloading resources..."
wget -c http://ftp.cbi.pku.edu.cn/pub/SVEN/resources.tar.gz
echo "Extracting resources..."
tar -xzf resources.tar.gz
echo "Finished extracting resources."

# download model parameters according to the mode
if [ "$mode" = "fast" ]; then
    echo "Downloading model parameters (fast mode)..."
    wget -c http://ftp.cbi.pku.edu.cn/pub/SVEN/model_params_fast.tar.gz
    echo "Extracting model parameters..."
    tar -xzf model_params_fast.tar.gz
    echo "Finished extracting model parameters."
elif [ "$mode" = "full" ]; then
    echo "Downloading model parameters (full mode)..."
    wget -c http://ftp.cbi.pku.edu.cn/pub/SVEN/model_params_full.tar.gz
    echo "Extracting model parameters..."
    tar -xzf model_params_full.tar.gz
    echo "Finished extracting model parameters."
fi
echo "All resources have been downloaded and extracted."
