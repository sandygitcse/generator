#!/bin/bash

#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90 --output_dir Outputs_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90 --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --device cuda:2
#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90 --output_dir Outputs_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90 --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --device cuda:2
#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90_coeffsstl --output_dir Outputs_m4_normzscore_typeindustry_lr1e-3_numblocks8_period90_coeffsstl --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2
#python main.py m4daily --epochs 20 --N_input 140 --N_output 14 --saved_models_dir saved_models_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90_coeffsstl --output_dir Outputs_m4_normzscore_typeindustry_lr1e-4_numblocks8_period90_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2


#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore_coeffsstl --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3 --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3 --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3_coeffsstl --output_dir Outputs_e960_d96_vdim8_K4_metricnll_normzscore_lr1e-3_coeffsstl --normalize zscore_per_series --learning_rate 0.001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2


#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly --output_dir Outputs_etthourly --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --device cuda:2
#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly_coeffsstl --output_dir Outputs_etthourly_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2
#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly_coeffsstl_leak --output_dir Outputs_etthourly_coeffsstl_leak --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2
#python main.py etthourly --epochs 20 --N_input 2400 --N_output 24 --saved_models_dir saved_models_etthourly_coeffsstl_leak_unnorm --output_dir Outputs_etthourly_coeffsstl_leak_unnorm --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 4 --use_coeffs --device cuda:2


#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett --output_dir Outputs_ett --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett_coeffsstl --output_dir Outputs_ett_coeffsstl --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett_coeffsstl_leak --output_dir Outputs_ett_coeffsstl_leak --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2
#python main.py ett --epochs 20 --N_input 960 --N_output 96 --saved_models_dir saved_models_ett_coeffsstl_leak_unnorm --output_dir Outputs_ett_coeffsstl_leak_unnorm --normalize zscore_per_series --learning_rate 0.0001 --batch_size 64 --hidden_size 128 --num_grulstm_layers 1 --v_dim 8 --use_coeffs --device cuda:2

#--t2v_type None
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir scripteff_saved_models_taxi30min_e336_d168_t2vnone \
#	--output_dir scripteff_Outputs_taxi30min_e336_d168_t2vnone \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir scripteff_saved_models_taxi30min_e336_d168_t2vlocal \
#	--output_dir scripteff_Outputs_taxi30min_e336_d168_t2vlocal \
#	--t2v_type local \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir scripteff_saved_models_taxi30min_e336_d168_t2vglobalidx \
#	--output_dir scripteff_Outputs_taxi30min_e336_d168_t2vglobalidx \
#	--t2v_type idx \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir scripteff_saved_models_taxi30min_e336_d168_t2vglobalmdhlincomb \
#	--output_dir scripteff_Outputs_taxi30min_e336_d168_t2vglobalmdhlincomb \
#	--t2v_type mdh_lincomb \
#	--device cuda:2
#
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir scripteff_saved_models_taxi30min_e336_d168_t2vglobalmdhparti \
#	--output_dir scripteff_Outputs_taxi30min_e336_d168_t2vglobalmdhparti \
#	--t2v_type 'mdh_parti' \
#	--device cuda:2

#--t2v_type None
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir scripteff_saved_models_ett_e384_d192_t2vnone_featsnorm \
#	--output_dir scripteff_Outputs_ett_e384_d192_t2vnone_featsnorm \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir scripteff_saved_models_ett_e384_d192_t2vlocal_featsnorm \
#	--output_dir scripteff_Outputs_ett_e384_d192_t2vlocal_featsnorm \
#	--t2v_type local \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir scripteff_saved_models_ett_e384_d192_t2vglobalidx_dropoutt2v_featsnorm \
#	--output_dir scripteff_Outputs_ett_e384_d192_t2vglobalidx_dropoutt2v_featsnorm \
#	--t2v_type idx \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir scripteff_saved_models_ett_e384_d192_t2vglobalmdhlincomb \
#	--output_dir scripteff_Outputs_ett_e384_d192_t2vglobalmdhlincomb \
#	--t2v_type mdh_lincomb \
#	--device cuda:2
#
#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir scripteff_saved_models_ett_e384_d192_t2vglobalmdhparti \
#	--output_dir scripteff_Outputs_ett_e384_d192_t2vglobalmdhparti \
#	--t2v_type mdh_parti \
#	--device cuda:2

#--t2v_type None
#python main.py azure \
#	--N_input 720 --N_output 360 \
#	--saved_models_dir script_saved_models_azure_e720_d360_t2vnone \
#	--output_dir script_Outputs_azure_e720_d360_t2vnone \
#	--device cuda:0

#python main.py azure \
#	--N_input 720 --N_output 360 \
#	--saved_models_dir scripteff_saved_models_azure_e720_d360_t2vlocal \
#	--output_dir scripteff_Outputs_azure_e720_d360_t2vlocal \
#	--t2v_type local \
#	--device cuda:0

# Commands for AAAI2022
#python main.py taxi30min \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir aaai_saved_models_taxi30min_e336_d168 \
#	--output_dir aaai_Outputs_taxi30min_e336_d168 \
#	--device cuda:2

#python main.py ett \
#	--N_input 384 --N_output 192 \
#	--saved_models_dir aaai_saved_models_ett_e384_d192_chunkfix_3_lr1e5_prunelastday_rp \
#	--output_dir aaai_Outputs_ett_e384_d192_chunkfix_3_lr1e5_prunelastday_rp \
#	--K_list 1 4 6 12 \
#	--device cuda:2

#python main.py Solar \
#	--N_input 336 --N_output 168 \
#	--saved_models_dir aaai_saved_models_Solar_e336_d168_2_rp \
#	--output_dir aaai_Outputs_Solar_e336_d168_2_rp \
#	--K_list 1 2 6 12 \
#	--device cuda:2

#python main.py etthourly \
#	--N_input 168 --N_output 168 \
#	--saved_models_dir aaai_saved_models_etthourly_e168_d168_9_lr1e5_prunelastday_rp \
#	--output_dir aaai_Outputs_etthourly_e168_d168_9_lr1e5_prunelastday_rp \
#	--K_list 1 2 6 \
#	--device cuda:1

# python main.py electricity \
# 	--N_input 336 --N_output 336 \
# 	--saved_models_dir saved_models/check_336/\
# 	--output_dir Results/check_336\
# 	--nhead 4\
# 	--mask 1\
# 	--device cuda:0\
# 	--message incasting\
# 	--options train dev test



# python main.py gecco \
# 	--N_input 180 --N_output 180 \
# 	--saved_models_dir saved_models/gecco_min_gen/mask_testing_5 \
# 	--output_dir Results/gecco_min_gen/mask_testing_5 \
# 	--device cuda:0\
# 	--epochs 1\
# 	--mask 1 \
# 	--message change\
# 	--options train test dev



python main.py energy \
	--N_input 168 --N_output 168 \
	--saved_models_dir saved_models/energy_inj/testing_feat \
	--output_dir Results/energy_inj/testing_feats \
	--device cuda:1\
	--epochs 50\
	--mask 1\
	--message 50_epoch \
	--options train test dev



# python main.py smd \
# 	--N_input 50 --N_output 50 \
# 	--saved_models_dir saved_models/check__smd/\
# 	--output_dir Results/check_smd\
# 	--nhead 4\
# 	--mask 0\
# 	--device cuda:1\
# 	--message 50_50\
# 	--options train dev test

#python main.py foodinflation \
#	--N_input 90 --N_output 30 \
#	--saved_models_dir saved_models_foodinflation \
#	--output_dir Outputs_foodinflation \
#	--device cuda:1

# python3 main.py telemetry \
# 	--N_input 365 --N_output 100 \
# 	--saved_models_dir saved_models_telemetry_tyt \
# 	--output_dir Outputs_telemetry_t \
# 	--initialization 0\
# 	--device cuda:1


# python3 main.py outlier \
# 	--N_input 100 --N_output 50 \
# 	--saved_models_dir saved_models/saved_models_synthetic_out_huber \
# 	--output_dir Outputs/Outputs_synthetic_out_huber \
# 	--initialization 0\
# 	--device cuda:0

# Commands for Oracle and SimRetrieval
#python main.py ett \
#	--N_input 3840 --N_output 192 \
#	--saved_models_dir aaai_saved_models_ett_oracle \
#	--output_dir aaai_Outputs_ett_oracle \
#	--normalize same \
#	--device cuda:2

#python main.py Solar \
#	--N_input 1680 --N_output 168 \
#	--saved_models_dir aaai_saved_models_Solar_oracle \
#	--output_dir aaai_Outputs_Solar_oracle \
#	--normalize same \
#	--device cuda:2

#python main.py etthourly \
#	--N_input 840 --N_output 168 \
#	--saved_models_dir aaai_saved_models_etthourly_oracle \
#	--output_dir aaai_Outputs_etthourly_oracle \
#	--normalize same \
#	--device cuda:2
#
# python main.py electricity \
# 	--N_input 1680 --N_output 168 \
# 	--saved_models_dir saved_models/saved_models_electricity_oracle \
# 	--output_dir Outputs/Outputs_electricity_oracle \
# 	--normalize same \
# 	--device cuda:0
