#!/bin/bash
# Program: ParMaCh - 1D Model of Solidification of Magma Chambers
# Script: Auxiliary script for the systematic search of the parametric space
# Run in background: nohup ./paper_landscape.sh --{flags} > /dev/null 2>&1 &

: '
    # Model B runs:
    > ./paper_landscape.sh --model=B --dir=B40const --hflux=40 --hfluxSI --ratio=10000 --latex --nohup
    > ./paper_landscape.sh --model=B --dir=B40vary  --hflux=40 --hfluxSI --HE_CSTM --ratio=10000 --latex --nohup
    > ./paper_landscape.sh --model=B --dir=B200const --hflux=200 --hfluxSI --ratio=10000 --latex --Ti97=0.81 --nohup
    > ./paper_landscape.sh --model=B --dir=B200vary --hflux=200 --hfluxSI --ratio=10000 --latex --Ti97=0.81 --nohup

    # Model A runs:
    > ./paper_landscape.sh --model=A --dir=A200const --hflux=200 --hfluxSI --ratio=50000 --latex --nohup
    > ./paper_landscape.sh --model=A --dir=A200vary --hflux=200 --hfluxSI --HE_CSTM --ratio=50000 --latex --nohup
    > ./paper_landscape.sh --model=A --dir=A800const --hflux=800 --hfluxSI --ratio=50000 --latex --Ti97=0.81 --nohup
    > ./paper_landscape.sh --model=A --dir=A800vary --hflux=800 --hfluxSI --HE_CSTM --ratio=50000 --latex --Ti97=0.81 --nohup
'

echo "------------------------------------------"
pwd=$(pwd)
echo "You are currently in $pwd."

make_newdir() {
    newdir="$1"
    if [ -e "$newdir" ]; then
        newdir="${newdir}copy"
    fi
    mkdir "$newdir"
    cp *.py "$newdir"/
    cd "$newdir" || return
}

# Global constants:
xdim=3; ydim=3; 

# Default values:
Ti97=0.92; hflux=40.0; N0HG97=1000.0; ratio=10000; dir="foo"; hfluxSI=0; model="(UNSPECIFIED)"; latex_flag=0; nohup_flag=0

# Process parsed arguments:
for arg in "$@"; do
    case $arg in 
        --Ti97=*)
            Ti97="${arg#*=}"    
            shift   
            ;;
        --hflux=*)
            hflux="${arg#*=}"
            shift
            ;;
        --N0HG97=*)
            N0HG97="${arg#*=}"
            shift
            ;;
        --dir=*)
            dir="${arg#*=}"
            shift
            ;;
        --hfluxSI)
            hfluxSI=1
            shift
            ;;        
        --HE_CSTM)
            HE_CSTM=1
            shift
            ;;
        --DEBUG100)
            DEBUG100=1
            shift
            ;;
        --DEBUG200)
            DEBUG200=1
            shift
            ;;
        --latex)
            latex_flag=1
            shift
            ;;
        --nohup)
            nohup_flag=1
            shift
            ;;
        --model=*)
            model="${arg#*=}"
            shift
            ;;
        --ratio=*)
            ratio="${arg#*=}"
            shift
            ;;
        *)
        echo "Unknown option: $arg!"
    esac
done

if [ -f main.py ]; then
    echo "Module main.py found."
else
    echo "Module main.py not found!."
    exit 1
fi

# Create a separate file and conduct the simulations there:
if [ -e "${dir}" ]; then    
    dir="${dir}_$(date +%Y-%m-%d_%H-%M-%S)"
fi
newdir="${dir}"
mkdir "$newdir"
cp *.py "$newdir"/
cd "$newdir"
path="$pwd/$newdir"

# Grid dimensions:
H0=(1000 100 10)                      # initial height of the chamber
G0=(1e-8 1e-7 1e-6)                   # growth rate amplitude
X0=(0.95 0.85 0.75 0.65)              # initial composition of the melt
nu=(0 1)                              # viscosity (off/on)
LOG_hflux=(--input_hflux)        
LOG_hfluxSI=(--hfluxSI)
LOG_SOLVER=(--SOLVER)
LOG_nu=(--nu)
if (( HE_CSTM == 1 )); then
    LOG_hvary=(--HE_CSTM)
else    
    LOG_hvary=()
fi
if (( DEBUG100 == 1 )); then
    LOG_DEBUG100=(--DEBUG100)
else
    LOG_DEBUG100=()
fi
if (( DEBUG200 == 1 )); then
    LOG_DEBUG200=(--DEBUG200)
else
    LOG_DEBUG200=()
fi

if [ ${#H0[@]} -gt $xdim ]; then
    echo "Grid dimension exceeded!"
    exit 1
fi
if [ ${#G0[@]} -gt $ydim ]; then
    echo "Grid dimension exceeded!"
    exit 1
fi

# Run ParMaCh simulations:
NPROC=$(nproc --all)
echo "$NPROC cores available."
running=0;
for i in "${!H0[@]}"; do
    H=${H0[$i]}
    if (( i == 0 )); then
        idxH=0
    elif (( i == 1 )); then
        idxH=3
    elif (( i == 2 )); then
        idxH=6
    fi
    for j in "${!G0[@]}"; do
        G=${G0[$j]}
        for k in "${!nu[@]}"; do
            nutmp=${nu[$k]}
            nodeidx=$(( idxH + (j + 1) ))
            if [ $nutmp -eq 0 ]; then
                tmpnode="node${nodeidx}_1" 
            else
                tmpnode="node${nodeidx}_2"
            fi 
            make_newdir "$tmpnode"
            pwd
            # TODO: decrease $ratio automatically for nu=1 simulations!
            for X0tmp in "${X0[@]}"; do 
                if [ $running -eq $NPROC ]; then
                    echo "Currently running $running processes, hold on."
                    wait; running=0
                fi 
                if [ $nutmp -eq 0 ]; then
                    command="$LOG_hflux $LOG_hfluxSI $LOG_hvary $LOG_DEBUG100 $LOG_DEBUG200 $LOG_SOLVER=0 --XL0="$X0tmp" --H0="$H" --V0HG97="$G" \
                    --hflux="$hflux" --Ti97="$Ti97" --ratio="$ratio" --SED_METHOD=4 --SCORR" 
                    if [ $nohup_flag -eq 0 ]; then
                        python3 main.py $command 
                        echo "Running: python3 main.py $command."
                    else
                        nohup python3 main.py $command > /dev/null 2>&1 & 
                        echo "Running: nohup python3 main.py $command."
                    fi
                else
                    command="$LOG_hflux $LOG_hfluxSI $LOG_hvary $LOG_DEBUG100 $LOG_DEBUG200 $LOG_SOLVER=0 --XL0="$X0tmp" --H0="$H" --V0HG97="$G" \
                    --hflux="$hflux" --Ti97="$Ti97" --ratio="$ratio" --SED_METHOD=4 --SCORR --nu"
                    if [ $nohup_flag -eq 0 ]; then
                        python3 main.py $command
                        echo "Running: python3 main.py $command."
                    else
                        nohup python3 main.py $command > /dev/null 2>&1 &
                        echo "Running: nohup python3 main.py $command."
                    fi
                fi
                ((running++))
                sleep 1.0
            done        
        cd "$path"
        done
    done
done    
wait

# Set up title for the landscape figure:
if (( HE_CSTM == 1 )); then
    subtitle="Model $model - Varying heat flux."
else
    subtitle="Model $model - Constant heat flux."
fi

# Run the mGrid.py and construct the landscape figure: 
echo "Running mGrid.py:"
if (( $latex_flag == 0 )); then
    nohup python3 mGrid.py --path="$path" --G0 "${G0[@]}" --H0 "${H0[@]}" --subtitle="$subtitle" > /dev/null 2>&1 & 
else
    nohup python3 mGrid.py --path="$path" --G0 "${G0[@]}" --H0 "${H0[@]}" "--latex" --subtitle="$subtitle" > /dev/null 2>&1 &
fi
wait
echo "Figure saved."

########################################################################
#% end of the script!