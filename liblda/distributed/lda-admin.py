#!/usr/bin/env python2.5
import subprocess


SERVERS = ["qubit-3", "axon"]

CLIENTS = [ "open-3", "open-5", "open-8", "open-13",
			"socs-21", "socs-12",
			"lab1-1",
			"lab2-2",
			"lab3-5",
			"lab4-8",
		   ]


def main():

    print "Staring to ping..."
    
    for host in CLIENTS:
        p = subprocess.Popen("ping -c 1 %s.cs.mcgill.ca "% host, shell=True, stdout=subprocess.PIPE)
        p.wait()
        out = p.stdout.readlines()
        
        if p.returncode == 0:
            print host + " is up and pingable"
        else:
            print host + " is down"
        #print out        
        #print c.stderr
        #c = subprocess.Popen("ssh even.cnd.mcgill.ca sudo cat /etc/cups/cupsd.conf | grep Listen", shell=True)


if __name__ == "__main__":
    main()

