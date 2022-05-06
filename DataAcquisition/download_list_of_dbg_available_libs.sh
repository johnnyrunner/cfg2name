echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse
deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse
deb http://ddebs.ubuntu.com $(lsb_release -cs)-proposed main restricted universe multiverse" | \
sudo tee -a /etc/apt/sources.list.d/ddebs.list

sudo apt install ubuntu-dbgsym-keyring
sudo apt-get update

apt-cache dumpavail | grep "Package:" | grep "Package:" | awk '{print $2}' | sort -u | grep 'dbg' > all_dbg_libs.txt
wc -l all_dbg_libs.txt
