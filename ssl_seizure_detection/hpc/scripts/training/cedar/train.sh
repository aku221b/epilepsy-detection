#!/bin/bash

model="$1"
epochs="$2"
project_id="$3"
split="$4"
exp_id="$5"
data_size="$6"

# Define the list of patients (26 patients).
declare -a patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Get date time id
datetime_id=$(date "+%Y-%m-%d_%H.%M.%S")

# Loop over each patient and call the original script
for patient_id in "${patients[@]}"; do
    if [ "$model" = "VICRegT1" ]; then
        sigma="$7"
        tau="$8"
        sr="$9"
        bash vicregt1_train.sh \
        "${patient_id}" \
        "VICRegT1" \
        "${datetime_id}" \
        "${sigma}" \
        "${tau}" \
        "${sr}" \
        "${epochs}" \
        "${project_id}" \
        "${split}" \
        "${exp_id}" \
        "${data_size}"
    
    elif [ "$model" = "supervised" ]; then
        classify="$7"
        bash supervised_train.sh \
        "${patient_id}" \
        "supervised" \
        "${datetime_id}" \
        "${epochs}" \
        "${project_id}" \
        "${split}" \
        "${classify}" \
        "${exp_id}" \
        "${data_size}"
            
    elif [ "$model" = "downstream3" ]; then

        
        classify="$7"
        pretrained_datetime_id="$8"
        requires_grad="$9" # Frozen (0), unfrozen (1)
        transfer_id="${10}"

        bash downstream3_train.sh \
        "${patient_id}" \
        "downstream3" \
        "${datetime_id}" \
        "${epochs}" \
        "${project_id}" \
        "${split}" \
        "${classify}" \
        "${exp_id}" \
        "${pretrained_datetime_id}" \
        "${requires_grad}" \
        "${transfer_id}" \
        "${data_size}"

    fi
done