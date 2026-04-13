#!/bin/bash

TIERS=("no-dp" "low-dp" "high-dp")
ROUNDS=(50)
EPOCHS=(1 5 20)

mkdir -p output/logs

for tier in "${TIERS[@]}"; do
    for rounds in "${ROUNDS[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
            
            LOG_FILE="output/logs/run_${tier}_R${rounds}_E${epochs}.log"
            
            echo "▶ Running: Tier=$tier | Rounds=$rounds | Epochs=$epochs"
            echo "  Saving logs to: $LOG_FILE"
            
            flwr run fl --stream \
                --run-config "privacy-tier='$tier' num-server-rounds=$rounds local-epochs=$epochs" \
                --federation-config "num-supernodes=4" > "$LOG_FILE" 2>&1
            
            echo "✔ Finished: Tier=$tier | Rounds=$rounds | Epochs=$epochs"
            echo "--------------------------------------------------"
            
        done
    done
done

