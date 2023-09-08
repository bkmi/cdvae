
CKPT_ROOT="/checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/"

function submit {
    echo $*
    mkdir -p exp/slurm/`date -I`/
    sbatch --job-name=$1 \
        --output=exp/slurm/`date -I`/$1-%j.out --error=exp/slurm/`date -I`/$1-%j.out \
        --nodes=1 --ntasks-per-node=1 --cpus-per-task=5 \
        --gres=gpu:1 --signal=USR1@600 --open-mode=append \
        --time=72:00:00 --partition=ocp \
        --wrap="srun $2" --constraint="volta32gb"
}

function submit8 {
    echo $*
    mkdir -p exp/slurm
    sbatch --job-name=$1 \
        --output=exp/slurm/$1-%j.out --error=exp/slurm/$1-%j.out \
        --nodes=1 --ntasks-per-node=1 --cpus-per-task=5 \
        --gres=gpu:8 --signal=USR1@600 --open-mode=append \
        --time=72:00:00 --partition=ocp \
        --wrap="srun $2"
}

# name="mp20_scn"
# for lr in 0.0006 0.0008 0.002; do
# # for lr in 0.0001 0.0003 0.001 0.003; do
#     name="mp20_scn_4x256_lr${lr}"
#     submit "$name" "python cdvae/run.py data=mp_20 expname=${name} optim.optimizer.lr=${lr}"
# done

# name="mp20_escn"
# for lr in 0.0006 0.0008 0.002 0.0001 0.0003 0.001 0.003; do
#     name="mp20_escn_4x256_lr${lr}"
#     submit "$name" "python cdvae/run.py data=mp_20 expname=${name} optim.optimizer.lr=${lr}"
# done

#python cdvae/run.py data=mp_20 expname=mp20_base_tmp3 train.pl_trainer.gpus=8








# python run.py data=mp_40 expname=mp40_tmp_rad10 model.condition_on_prop=True model.prop_dropout=0.4 model.max_neighbors=20 model.radius=10

# name="mp20_base"
# submit "$name" "python run.py data=mp_20 expname=$name && \
#                 python ../scripts/evaluate.py --model_path $CKPT_ROOT/`date -I`/$name --tasks recon gen model.cond_value=0 --cond_scale=1 && \
#                 python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/`date -I`/$name --tasks recon gen --max-ehull 0.01"
# for lr in 0.0006 0.0008 0.001; do
# for drop in 0 0.1 0.2; do
#     name="mp20_ehull${drop}_lr${lr}"
#     args="model.condition_on_prop=True model.prop_dropout=${drop} optim.optimizer.lr=${lr}"
#     submit "$name" "python run.py data=mp_20 expname=$name $args && \
#                     python ../scripts/evaluate.py --model_path $CKPT_ROOT/`date -I`/$name --tasks recon gen model.cond_value=0 --cond_scale=1 --label cond1 && \
#                     python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/`date -I`/$name --tasks recon gen --max-ehull 0.01 --label cond1 && \
#                     python ../scripts/evaluate.py --model_path $CKPT_ROOT/`date -I`/$name --tasks recon gen model.cond_value=0 --cond_scale=6 --label cond6 && \
#                     python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/`date -I`/$name --tasks recon gen --max-ehull 0.01 --label cond6"
# done
# done

# for min_ehull in 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07; do
#     max_ehull=`echo "$min_ehull + 0.01" | bc`
#     echo "Run: mp20_ehull0.1_lr0.001 cond1 $min_ehull $max_ehull"
#     python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/2023-02-27/mp20_ehull0.1_lr0.001 --tasks recon --min-ehull $min_ehull --max-ehull $max_ehull --label cond1
#     echo "Run: mp20_ehull0.1_lr0.001 cond6 $min_ehull $max_ehull"
#     python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/2023-02-27/mp20_ehull0.1_lr0.001 --tasks recon --min-ehull $min_ehull --max-ehull $max_ehull --label cond6
#     echo "Run: mp20_base $min_ehull $max_ehull"
#     python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/2023-02-27/mp20_base --tasks recon --min-ehull $min_ehull --max-ehull $max_ehull
# done


######################################################################################

# HYDRA_FULL_ERROR=1 python run.py data=mp_full expname=mpfull_tmp model.condition_on_prop=True model.prop_dropout=0.4 train.pl_trainer.max_epochs=1
# HYDRA_FULL_ERROR=1 python run.py data=mp_40 expname=mpfull_tmp model.condition_on_prop=True model.prop_dropout=0.4 train.pl_trainer.max_epochs=4 train.pl_trainer.gpus=2

# name="mp40_base_lr0.0003_rad12_nbr30"
# submit "$name" "python run.py data=mp_40 expname=$name optim.optimizer.lr=0.0003 model.radius=12 model.max_neighbors=30  && \
#                 python ../scripts/evaluate.py --model_path $CKPT_ROOT/`date -I`/$name --tasks recon gen model.cond_value=0 --cond_scale=1 && \
#                 python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/`date -I`/$name --tasks recon gen --max-ehull 0.01"
# for lr in 0.0001 0.0003 0.0006; do
# for drop in 0.1 0.2 0.3; do
#     name="mp40_ehull${drop}_lr${lr}_rad12_nbr30"
#     args="model.condition_on_prop=True model.prop_dropout=${drop} optim.optimizer.lr=${lr} model.radius=12 model.max_neighbors=30"
#     submit "$name" "python run.py data=mp_40 expname=$name $args && \
#                     python ../scripts/evaluate.py --model_path $CKPT_ROOT/`date -I`/$name --tasks recon gen model.cond_value=0 --cond_scale=1 --label cond1 && \
#                     python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/`date -I`/$name --tasks recon gen --max-ehull 0.01 --label cond1 && \
#                     python ../scripts/evaluate.py --model_path $CKPT_ROOT/`date -I`/$name --tasks recon gen model.cond_value=0 --cond_scale=3 --label cond3 && \
#                     python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/`date -I`/$name --tasks recon gen --max-ehull 0.01 --label cond3"
# done
# done

name="mp40_base_lr0.0003_rad12_nbr30_propehull"
# submit "$name" "python run.py data=mp_40 expname=$name optim.optimizer.lr=0.0003 model.radius=12 model.max_neighbors=30 model.predict_property=True && \
#                 python ../scripts/evaluate.py --model_path $CKPT_ROOT/`date -I`/$name --tasks recon gen model.cond_value=0 --cond_scale=1 && \
#                 python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/`date -I`/$name --tasks recon gen --max-ehull 0.01"




# [
#     'builder_meta', 'nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty', 'formula_anonymous', 
#     'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'property_name', 'material_id', 'deprecated', 'deprecation_reasons', 
#     'last_updated', 'origins', 'warnings', 'structure', 'task_ids', 'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom', 
#     'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to', 'xas', 'grain_boundaries', 'band_gap', 
#     'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal', 'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 
#     'is_magnetic', 'ordering', 'total_magnetization', 'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units', 
#     'num_magnetic_sites', 'num_unique_magnetic_sites', 'types_of_magnetic_species', 'k_voigt', 'k_reuss', 'k_vrh', 'g_voigt', 'g_reuss', 
#     'g_vrh', 'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n', 'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 
#     'weighted_surface_energy', 'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed', 'possible_species', 
#     'has_props', 'theoretical', 'database_IDs'
# ]




######################################################################################



# function train {
#     echo $*
#     name=$1
#     args=$2
#     path=$CKPT_ROOT/`date -I`/$name

#     train_cmd="python run.py expname=$name $args"

#     eval_cmd1="python ../scripts/evaluate.py --model_path $path --tasks recon model.cond_value=0. --cond_scale=1. --label cond1"
#     eval_cmd2="python ../scripts/evaluate.py --model_path $path --tasks recon model.cond_value=0. --cond_scale=6. --label cond6"
#     eval_cmd3="python ../scripts/evaluate.py --model_path $path --tasks recon model.cond_value=0. --cond_scale=10. --label cond10"
#     metrics_cmd1="python ../scripts/compute_metrics.py --root_path $path --tasks recon --max-ehull 0.01 --label cond1"
#     metrics_cmd2="python ../scripts/compute_metrics.py --root_path $path --tasks recon --max-ehull 0.01 --label cond6"
#     metrics_cmd3="python ../scripts/compute_metrics.py --root_path $path --tasks recon --max-ehull 0.01 --label cond10"

#     # eval_cmd1="python ../scripts/evaluate.py --model_path $path --tasks recon gen model.cond_value=0. --cond_scale=1. --label cond1"
#     # eval_cmd2="python ../scripts/evaluate.py --model_path $path --tasks recon gen model.cond_value=0. --cond_scale=6. --label cond6"
#     # eval_cmd3="python ../scripts/evaluate.py --model_path $path --tasks recon gen model.cond_value=0. --cond_scale=10. --label cond10"
#     # metrics_cmd1="python ../scripts/compute_metrics.py --root_path $path --tasks recon gen --max-ehull 0.01 --label cond1"
#     # metrics_cmd2="python ../scripts/compute_metrics.py --root_path $path --tasks recon gen --max-ehull 0.01 --label cond6"
#     # metrics_cmd3="python ../scripts/compute_metrics.py --root_path $path --tasks recon gen --max-ehull 0.01 --label cond10"
#     # cmd="${train_cmd} && ${eval_cmd1} && ${eval_cmd2} && ${eval_cmd3} && ${metrics_cmd1} && ${metrics_cmd2} && ${metrics_cmd3}"
#     cmd="`${train_cmd}` && `${eval_cmd1}` && `${eval_cmd2}` && `${eval_cmd3}` && `${metrics_cmd1}` && `${metrics_cmd2}` && `${metrics_cmd3}`"
#     echo $cmd

#     mkdir -p exp/slurm
#     $cmd
#     # sbatch --job-name=$name --output=exp/slurm/$name-%j.out --error=exp/slurm/$name-%j.out \
#     #     --nodes=1 --ntasks-per-node=1 --cpus-per-task=5 --gres=gpu:1 --signal=USR1@600 \
#     #     --time=72:00:00 --partition=ocp --open-mode=append --wrap="srun ${cmd}"
# }

# train "mptiny_base" "data=mp_tiny model.condition_on_prop=True model.prop_dropout=0.4 train.pl_trainer.max_epochs=2"
# python ../scripts/evaluate.py --model_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/2023-02-27/mptiny_base --tasks recon model.cond_value=0. --cond_scale=6. --label cond6


# python run.py data=mp_20 expname=mptiny_base model.condition_on_prop=True model.prop_dropout=0.4 train.pl_trainer.max_epochs=2 && 
# python ../scripts/evaluate.py --model_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun//2023-02-27/mptiny_base --tasks recon model.cond_value=0. --cond_scale=1. --label cond1 && 
# python ../scripts/evaluate.py --model_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun//2023-02-27/mptiny_base --tasks recon model.cond_value=0. --cond_scale=6. --label cond6 && 
# python ../scripts/evaluate.py --model_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun//2023-02-27/mptiny_base --tasks recon model.cond_value=0. --cond_scale=10. --label cond10 && 
# python ../scripts/compute_metrics.py --root_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun//2023-02-27/mptiny_base --tasks recon --max-ehull 0.01 --label cond1 && 
# python ../scripts/compute_metrics.py --root_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun//2023-02-27/mptiny_base --tasks recon --max-ehull 0.01 --label cond6 && 
# python ../scripts/compute_metrics.py --root_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun//2023-02-27/mptiny_base --tasks recon --max-ehull 0.01 --label cond10






# python run.py data=mp_tiny expname=mptiny_tmp_ehull model.condition_on_prop=True model.prop_dropout=0.4 train.pl_trainer.max_epochs=2 
# python  ../scripts/evaluate.py --model_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/2023-02-27/mptiny_tmp_ehull/ --tasks recon +model.cond_value=0.
# python ../scripts/compute_metrics.py --root_path /checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/2023-02-27/mptiny_tmp_ehull/ --tasks recon +model.cond_value=0.
# python run.py data=mp_20 expname=mptiny_tmp_ehull model.condition_on_prop=True optim.optimizer.lr=0.0003 model.prop_dropout=0.3
# python run.py data=mp_tiny expname=mptiny_tmp model.condition_on_time=True model.condition_on_prop=True optim.optimizer.lr=0.0003 model.time_dropout=0.5 model.prop_dropout=0.3
# python run.py data=mp_20 expname=mp20_tmp
# python run.py data=mp_20 expname=mp20_tmp model.condition_on_time=True model.condition_on_prop=True optim.optimizer.lr=0.0003 model.time_dropout=0.1 model.prop_dropout=0.1 train.pl_trainer.max_epochs=2

# CKPT_ROOT="/checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/2023-02-26/"
# name="mp20_base"
# submit "$name" "python run.py data=mp_20 expname=$name && python ../scripts/evaluate.py --model_path $CKPT_ROOT/$name --tasks recon gen model.cond_value=0."
# for lr in 0.0003 0.0006 0.001; do
#     for drop in 0 0.1 0.2; do
#         name="mp20_noise${drop}_lr${lr}"
#         args="model.condition_on_time=True model.time_dropout=${drop} optim.optimizer.lr=${lr}"
#         submit "$name" "python run.py data=mp_20 expname=$name $args && python ../scripts/evaluate.py --model_path $CKPT_ROOT/$name --tasks recon gen model.cond_value=0."

#         name="mp20_noise${drop}_ehull${drop}_lr${lr}"
#         args="model.condition_on_prop=True model.prop_dropout=${drop} optim.optimizer.lr=${lr}"
#         submit "$name" "python run.py data=mp_20 expname=$name $args && python ../scripts/evaluate.py --model_path $CKPT_ROOT/$name --tasks recon gen model.cond_value=0."

#         name="mp20_ehull${drop}_lr${lr}"
#         args="model.condition_on_time=True model.condition_on_prop=True model.time_dropout=${drop} model.prop_dropout=${drop} optim.optimizer.lr=${lr}"
#         submit "$name" "python run.py data=mp_20 expname=$name $args && python ../scripts/evaluate.py --model_path $CKPT_ROOT/$name --tasks recon gen model.cond_value=0."
#     done
# done

# CKPT_ROOT="/checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/2023-02-26/"
# name="mp20_base"
# submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen > $CKPT_ROOT/$name/metrics_full.txt"
# submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0.01 > $CKPT_ROOT/$name/metrics_0_0.01.txt"
# submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0.001 > $CKPT_ROOT/$name/metrics_0_0.001.txt"
# submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0 > $CKPT_ROOT/$name/metrics_0_0.txt"
# for lr in 0.0003 0.0006 0.001; do
#     for drop in 0 0.1 0.2; do
#         name="mp20_noise${drop}_lr${lr}"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen > $CKPT_ROOT/$name/metrics_full.txt"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0.01 > $CKPT_ROOT/$name/metrics_0_0.01.txt"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0 > $CKPT_ROOT/$name/metrics_0_0.txt"

#         name="mp20_noise${drop}_ehull${drop}_lr${lr}"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen > $CKPT_ROOT/$name/metrics_full.txt"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0.01 > $CKPT_ROOT/$name/metrics_0_0.01.txt"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0 > $CKPT_ROOT/$name/metrics_0_0.txt"

#         name="mp20_ehull${drop}_lr${lr}"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen > $CKPT_ROOT/$name/metrics_full.txt"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0.01 > $CKPT_ROOT/$name/metrics_0_0.01.txt"
#         submit "$name" "python ../scripts/compute_metrics.py --root_path $CKPT_ROOT/$name --tasks recon gen --max-ehull 0 > $CKPT_ROOT/$name/metrics_0_0.txt"
#     done
# done













# submit "mp20_base" "python run.py data=mp_20 expname=mp20_base"
# for lr in 0.0003 0.0006 0.001; do
#     for drop in 0 0.1 0.2; do
#         submit "mp20_noise${drop}_lr${lr}"              "python run.py data=mp_20 expname=mp20_noise${drop}_lr${lr}              model.condition_on_time=True model.time_dropout=${drop} optim.optimizer.lr=${lr} "
#         submit "mp20_ehull${drop}_lr${lr}"              "python run.py data=mp_20 expname=mp20_ehull${drop}_lr${lr}              model.condition_on_prop=True model.prop_dropout=${drop} optim.optimizer.lr=${lr} "
#         submit "mp20_noise${drop}_ehull${drop}_lr${lr}" "python run.py data=mp_20 expname=mp20_noise${drop}_ehull${drop}_lr${lr} model.condition_on_time=True model.condition_on_prop=True model.time_dropout=${drop} model.prop_dropout=${drop} optim.optimizer.lr=${lr}"
#     done
# done

# CKPT_ROOT="/checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/2023-02-26/"
# submit "eval_base" "python ../scripts/evaluate.py --model_path $CKPT_ROOT/mp20_base --tasks recon gen +model.cond_value=0."
# for lr in 0.0003 0.0006 0.001; do
#     for drop in 0 0.1 0.2; do
#         submit "eval_noise${drop}_lr${lr}"              "python ../scripts/evaluate.py --model_path $CKPT_ROOT/mp20_noise${drop}_lr${lr}               --tasks recon gen +model.cond_value=0."
#         submit "eval_ehull${drop}_lr${lr}"              "python ../scripts/evaluate.py --model_path $CKPT_ROOT/mp20_ehull${drop}_lr${lr}               --tasks recon gen +model.cond_value=0."
#         submit "eval_noise${drop}_ehull${drop}_lr${lr}" "python ../scripts/evaluate.py --model_path $CKPT_ROOT/mp20_noise${drop}_ehull${drop}_lr${lr}  --tasks recon gen +model.cond_value=0."
# done
# done

# CKPT_BASE="/checkpoint/anuroops/chemistry/cdvae/cdvae/exp/hydra/singlerun/2023-02-26/mp20_base/"
# python ../scripts/evaluate.py --model_path $CKPT_BASE --tasks recon data=mp_tiny +model.cond_value=0.
# python ../scripts/compute_metrics.py --root_path $CKPT_BASE --tasks recon data=mp_tiny
# python ../scripts/evaluate.py --model_path $CKPT_ROOT/mp20_noise0_ehull0_lr0.0003 --tasks recon gen +model.cond_value=0.
