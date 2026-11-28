#!/bin/bash
# Experiments of the 2nd assignment (geometric verification with verify).
# Every run keeps its output file (-o), its full log and one line in
# results/<set>/verify_<experiment>/results.csv
#
# Usage: bash experiments_verify.sh C|rho|tau|T|D|best|final
#   C      all methods, C = 20, 50, 100 (also compares the three rankings)
#   rho    Lowe ratio            (exact search)
#   tau    inlier threshold      (exact search)
#   T      RANSAC trials, adaptive limit on/off (exact search)
#   D      all methods, D = 1000, 5000, 10000, 25000
#   best   all methods with the chosen values together (validation)
#   final  all methods and all D, and C = 20, 50 (SET=test)
# D and SET can be changed from the command line, e.g.
#   D=1000 bash experiments_verify.sh rho
#   SET=test bash experiments_verify.sh final

HP=../datasets/hpatches-sequences-release
MIR=../datasets/mirflickr
VOCAB=256
D=${D:-25000}
SET=${SET:-validation}

# parameters of the methods: the best values of the 1st assignment
EXACT_P=""
LSH_P="-k 4 -L 20 -w 0.1"
HC_P="-kproj 8 -w 0.05 -M 100000 -probes 100"
IVFFLAT_P="-nprobe 10"
IVFPQ_P="-nprobe 10 -M 32 -nbits 8"

# parameters of the verification: the best values found on validation
# (the defaults of the assignment are C=50, rho=0.8, tau=3, T=2000, adaptive=1)
#   C=100      +2.7 to +3.5 points of mAP@10 against C=50, 1.6 times the time
#   rho=0.7    same mAP@10 as 0.8, more accurate homographies, less time
#   tau=5      same mAP@10 as 2-8, the most accurate homographies
#   adaptive=0 all 2000 trials: 71% accurate homographies against 64%,
#              for 4% more time
# The C, rho, tau and T experiments were run with the defaults of the
# assignment and change one parameter at a time.
C=${C:-100}
RHO=${RHO:-0.7}
TAU=${TAU:-5}
TMAX=${TMAX:-2000}
P=0.99
ADAPTIVE=${ADAPTIVE:-0}

# -kclusters has no default in verify: the same number of clusters as the
# 1st assignment, ceil(sqrt(base size)), with base size = 115 + D
kclusters() {
    awk -v n=$((115 + $1)) 'BEGIN { k = int(sqrt(n)); if (k * k < n) k++; print k }'
}

# start an experiment: folder and a new csv
start() {
    exp=verify_$1
    dir=results/$SET/$exp
    mkdir -p "$dir"
    csv=$dir/results.csv
    echo "name,set,D,method,params,C,rho,tau,T,p,adaptive,before_recall5,before_recall10,before_map10,matches_recall5,matches_recall10,matches_map10,after_recall5,after_recall10,after_map10,candidate_ms,verification_ms,valid_rate,accurate_rate,corner_error_px,before_map10_illumination,after_map10_illumination,valid_illumination,accurate_illumination,before_map10_viewpoint,after_map10_viewpoint,valid_viewpoint,accurate_viewpoint,fewer_than_C" > "$csv"
}

# value that follows a word in a line, e.g. value "$line" mAP@10
value() {
    echo "$1" | sed -n "s/.*$2 \([0-9.e+-]*\).*/\1/p"
}

# line <log> <group> <what>: one line of the summary of a group
# (group All, Illumination or Viewpoint; what original, matches, inliers,
# time or valid)
line() {
    grep -A6 "^$2 (" "$1" | grep "^  $3"
}

# metrics of one ranking: recall5,recall10,map10
ranking() {
    l=$(line "$1" All "$2")
    echo "$(value "$l" Recall@5),$(value "$l" Recall@10),$(value "$l" mAP@10)"
}

# rates of valid and accurate homographies of a group: valid,accurate
rates() {
    l=$(line "$1" "$2" valid)
    echo "$l" | sed -n 's/.*valid homographies [0-9]* (\([0-9.e+-]*\)), accurate [0-9]* (\([0-9.e+-]*\)).*/\1,\2/p'
}

# run <name> <D> <method> <C> <rho> <tau> <T> <adaptive> [method parameters]
run() {
    name=$1 d=$2 method=$3 c=$4 rho=$5 tau=$6 tmax=$7 adaptive=$8
    shift 8
    extra=""
    if [ "$method" = ivfflat ] || [ "$method" = ivfpq ]; then
        extra="-kclusters $(kclusters $d)"
    fi
    echo "[$exp] $name"
    log=$dir/$name.log
    if ! ./verify -hp $HP -mir $MIR -split split.txt -cache cache -set $SET \
            -vocab $VOCAB -D $d -o "$dir/$name.txt" -$method "$@" $extra \
            -C $c -rho $rho -tau $tau -T $tmax -p $P -adaptive $adaptive > "$log" 2>&1; then
        echo "  failed, see $log"
        return
    fi
    time=$(line "$log" All time)
    corner=$(line "$log" All valid | sed -n 's/.*mean corner error \([0-9.e+-]*\|NA\).*/\1/p')
    illum=$(line "$log" Illumination original)
    illumAfter=$(line "$log" Illumination inliers)
    view=$(line "$log" Viewpoint original)
    viewAfter=$(line "$log" Viewpoint inliers)
    fewer=$(grep 'candidates:' "$log" | awk '{print $NF}')
    echo "$name,$SET,$d,$method,$* $extra,$c,$rho,$tau,$tmax,$P,$adaptive,$(ranking "$log" original),$(ranking "$log" matches),$(ranking "$log" inliers),$(value "$time" candidates),$(value "$time" verification),$(rates "$log" All),$corner,$(value "$illum" mAP@10),$(value "$illumAfter" mAP@10),$(rates "$log" Illumination),$(value "$view" mAP@10),$(value "$viewAfter" mAP@10),$(rates "$log" Viewpoint),$fewer" >> "$csv"
    echo "  mAP@10 $(value "$(line "$log" All original)" mAP@10) -> $(value "$(line "$log" All inliers)" mAP@10), verification $(value "$time" verification) ms/query"
}

# run_all <name suffix> <D> <C>: the five methods with the same D and C
run_all() {
    run exact_$1 $2 exact $3 $RHO $TAU $TMAX $ADAPTIVE $EXACT_P
    run lsh_$1 $2 lsh $3 $RHO $TAU $TMAX $ADAPTIVE $LSH_P
    run hypercube_$1 $2 hypercube $3 $RHO $TAU $TMAX $ADAPTIVE $HC_P
    run ivfflat_$1 $2 ivfflat $3 $RHO $TAU $TMAX $ADAPTIVE $IVFFLAT_P
    run ivfpq_$1 $2 ivfpq $3 $RHO $TAU $TMAX $ADAPTIVE $IVFPQ_P
}

# the defaults of the assignment, for the experiments that change one
# parameter at a time
assignment_defaults() {
    C=50 RHO=0.8 TAU=3 TMAX=2000 ADAPTIVE=1
}

case $1 in
C)
    assignment_defaults
    start C
    for c in 20 50 100; do
        run_all C$c $D $c
    done
    ;;
rho)
    assignment_defaults
    start rho
    for rho in 0.6 0.7 0.8 0.9; do
        run rho$rho $D exact $C $rho $TAU $TMAX $ADAPTIVE $EXACT_P
    done
    ;;
tau)
    assignment_defaults
    start tau
    for tau in 1 2 3 5 8; do
        run tau$tau $D exact $C $RHO $tau $TMAX $ADAPTIVE $EXACT_P
    done
    ;;
T)
    assignment_defaults
    start T
    for tmax in 50 200 500 2000 5000; do
        run T${tmax}_adaptive $D exact $C $RHO $TAU $tmax 1 $EXACT_P
    done
    for tmax in 500 2000; do
        run T${tmax}_fixed $D exact $C $RHO $TAU $tmax 0 $EXACT_P
    done
    ;;
D)
    assignment_defaults
    start D
    for d in 1000 5000 10000 25000; do
        run_all D$d $d $C
    done
    ;;
best)
    # the chosen values together, on validation
    start best
    run_all best $D $C
    ;;
final)
    # chosen values on the test set: all D with the chosen C, and the other
    # values of C with D = 25000
    start final
    for d in 1000 5000 10000 25000; do
        run_all D$d $d $C
    done
    for c in 20 50; do
        run_all C${c}_D25000 25000 $c
    done
    ;;
*)
    echo "Usage: bash $0 C|rho|tau|T|D|best|final"
    exit 1
    ;;
esac
