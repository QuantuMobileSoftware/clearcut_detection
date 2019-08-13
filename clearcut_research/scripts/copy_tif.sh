mkdir $2/$3

for ch in rgb b4 b8
do
    ssh_tif=$1/$3_$ch.tif
    local_tif=$2/$3/$3_$ch.tif
    scp -P 9999 $ssh_tif $local_tif
done
