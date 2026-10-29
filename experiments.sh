#!/bin/bash
# Runs the experiments of the report. Every run keeps its output file (-o),
# its full log and one line in results/<set>/<experiment>/results.csv
#
# Usage: bash experiments.sh vocab|lsh|hypercube|ivfflat|ivfpq|D|final
# VOCAB, D and SET can be changed from the command line, e.g.
#   VOCAB=512 bash experiments.sh lsh
#   SET=test bash experiments.sh final

HP=../datasets/hpatches-sequences-release
MIR=../datasets/mirflickr
VOCAB=${VOCAB:-256}
D=${D:-25000}
SET=${SET:-validation}

# parameters of every method for the "D" and "final" experiments
# (set to the best values found on validation)
EXACT_P=""
LSH_P="-k 4 -L 20 -w 0.1"
HC_P="-kproj 8 -w 0.05 -M 100000 -probes 100"
IVFFLAT_P="-nprobe 10"
IVFPQ_P="-nprobe 10 -M 32 -nbits 8"

# start an experiment: empty folder with a new csv
start() {
    exp=$1
    dir=results/$SET/$exp
    rm -rf "$dir"
    mkdir -p "$dir"
    csv=$dir/results.csv
    echo "name,vocab,D,method,params,recall5,recall10,map10,time_ms,map_illumination,map_viewpoint,short_results,total_s" > "$csv"
}

# value that follows a word in a summary line, e.g. value "$line" mAP@10
value() {
    echo "$1" | sed -n "s/.*$2 \([0-9.e+-]*\).*/\1/p"
}

# run <name> <vocab> <D> <method> [method parameters]
run() {
    name=$1 vocab=$2 d=$3 method=$4
    shift 4
    echo "[$exp] $name"
    log=$dir/$name.log
    if ! ./search -hp $HP -mir $MIR -split split.txt -cache cache -set $SET \
            -vocab $vocab -D $d -o "$dir/$name.txt" -$method "$@" > "$log" 2>&1; then
        echo "  failed, see $log"
        return
    fi
    all=$(grep '^All' "$log")
    illum=$(grep '^Illumination' "$log")
    view=$(grep '^Viewpoint' "$log")
    short=$(grep 'fewer than' "$log" | awk '{print $NF}')
    total=$(grep '^Total time' "$log" | awk '{print $3}')
    echo "$name,$vocab,$d,$method,$*,$(value "$all" Recall@5),$(value "$all" Recall@10),$(value "$all" mAP@10),$(value "$all" time),$(value "$illum" mAP@10),$(value "$view" mAP@10),$short,$total" >> "$csv"
    echo "  mAP@10 $(value "$all" mAP@10), $(value "$all" time) ms"
}

case $1 in
vocab)
    start vocab
    for K in 128 256 512 1024; do
        run K$K $K $D exact
    done
    ;;
lsh)
    start lsh
    for L in 5 10 20 40; do
        for w in 0.05 0.1 0.2 0.4; do
            run L${L}_w$w $VOCAB $D lsh -k 4 -L $L -w $w
        done
    done
    ;;
hypercube)
    # M is large so that only probes limits the search
    start hypercube
    for kproj in 8 10 12 14; do
        for probes in 10 50 100 200; do
            run kproj${kproj}_probes$probes $VOCAB $D hypercube -kproj $kproj -w 0.05 -M 100000 -probes $probes
        done
    done
    ;;
ivfflat)
    start ivfflat
    for nprobe in 1 2 5 10 20 40; do
        run nprobe$nprobe $VOCAB $D ivfflat -nprobe $nprobe
    done
    ;;
ivfpq)
    start ivfpq
    for M in 8 16 32; do
        for nprobe in 1 5 10 20; do
            run M${M}_nprobe$nprobe $VOCAB $D ivfpq -nprobe $nprobe -M $M -nbits 8
        done
    done
    ;;
D)
    start D
    for d in 1000 5000 10000 25000; do
        run exact_D$d $VOCAB $d exact $EXACT_P
        run lsh_D$d $VOCAB $d lsh $LSH_P
        run hypercube_D$d $VOCAB $d hypercube $HC_P
        run ivfflat_D$d $VOCAB $d ivfflat $IVFFLAT_P
        run ivfpq_D$d $VOCAB $d ivfpq $IVFPQ_P
    done
    ;;
final)
    start final
    run exact $VOCAB $D exact $EXACT_P
    run lsh $VOCAB $D lsh $LSH_P
    run hypercube $VOCAB $D hypercube $HC_P
    run ivfflat $VOCAB $D ivfflat $IVFFLAT_P
    run ivfpq $VOCAB $D ivfpq $IVFPQ_P
    ;;
*)
    echo "Usage: bash $0 vocab|lsh|hypercube|ivfflat|ivfpq|D|final"
    exit 1
    ;;
esac
