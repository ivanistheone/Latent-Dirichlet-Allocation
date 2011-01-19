#!/usr/bin/env python2.5
import subprocess


SERVERS = ["cube", "hak", ]

CLIENTS = ["even","patch","nexus",
           "flicker","crp","sprint","hyper",
           "ruch", #"dragoon",
           "cnd181", #"logan",
           "samba","giant","harvey"]


def main():

    print "Staring to ping..."
    
    for host in CLIENTS:
        p = subprocess.Popen("ssh %s.cnd.mcgill.ca ls /usr/local/"% host, shell=True, stdout=subprocess.PIPE)
        p.wait()
        out = p.stdout.readlines()
        
        if p.returncode == 0:
            print host + ":"
            for file in out:
                if file[0:3]=="mat":
                    print file[:-1]
        else:
            print host + " is down"


        p = subprocess.Popen("ssh %s.cnd.mcgill.ca ps au | grep MATLAB | grep -v grep"% host, shell=True, stdout=subprocess.PIPE)
        p.wait()
        out = p.stdout.readlines()

        if p.returncode ==0:
            print "processes:"
            print out
            

        print ""

        #print out        
        #print c.stderr
        #c = subprocess.Popen("ssh even.cnd.mcgill.ca sudo cat /etc/cups/cupsd.conf | grep Listen", shell=True)


if __name__ == "__main__":
    main()

