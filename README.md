How to look at all libs and available dbg symbols
ls /var/lib/apt/lists/*_Packages
grep ^Package /var/lib/apt/lists/archive.ubuntu.com_ubuntu_dists_focal*_Packages | awk '{print $2}' | sort -u 
grep ^Package /var/lib/apt/lists/ddebs.ubuntu.com_dists_foca*_Packages | awk '{print $2}' | sort -u | grep 'dbgsym' |wc -l

sudo apt-get -d install zzuf-dbgsym
dpkg-deb -xv zzuf-dbgsym_0.15-1_amd64.ddeb zuffy