# -*- coding: utf-8 -*-
"""
fabfile for Latent-Dirichlet-Allocation
---------------------------------------
Use this script to check:
 - which hosts are up and running
 - have no users logged in
 - other stuff...
"""

from __future__ import with_statement # needed for python 2.5
from fabric.api import *
import os,sys
import time

# globals
env.project_name = 'Latent-Dirichlet-Allocation'
env.warn_only = True
env.LDAdir_SOCS = '/home/2002/isavov/LDA'

# environments
def localhost():
    "Use the local virtual server"
    env.hosts = ['127.0.0.1']
    env.user = 'ivan'
    env.path = '/Users/%(user)s/Documents/Projects/(project_name)s' % env
    env.virtualhost_path = env.path

def lab7():
    "The linux machines in lab7"
    id_list = range(1,29)
    all_hosts = [ "lab7-"+str(h)+".cs.mcgill.ca" for h in id_list ]
    cache_filename = "lab7-reachable.txt"
    # check if cached?
    recent = False
    if os.path.exists(cache_filename):
        recent = time.time() - os.path.getmtime(cache_filename) < 60*60
        # cache expires every hour
    ##### NOT CACHED ####################################################
    if not recent:
        env.hosts = []
        for host in all_hosts:
            env.host_string = host
            env.user = 'isavov'
            env.path = '/home/2002/isavov/LDA' # /var/www/%(project_name)s' % env
            verdict = check_if_up(host)
            if verdict:
                env.hosts.append( host )
        # populate cache
        fc = open(cache_filename, 'w')
        for h in env.hosts:
            fc.write(h+"\n")
    ##### CACHED #########################################################
    else: # read from reachable-hosts cache file
        env.hosts = []
        env.user = 'isavov'
        env.path = '/home/2002/isavov/LDA' # /var/www/%(project_name)s' % env
        for line in open(cache_filename, 'r').readlines():
            env.hosts.append( line.strip() )
    ##### END ############################################################


def check_if_up(host):
    env.host_string = host
    try:
        out=run("uptime")
        if "load average" in out:
            print "Host ", env.host_string, " is reachable, and ssh loginable."
            return True
        else:
            print "Host ", env.host_string, " is reachable, but no ssh access."
            return True
    except:
        print "host %s is unreachable" % env.host_string
        return False


def lda_run(command_line="--help"):
    """ Run a topic model on the cluster (in a screen session) """
    with cd(env.LDAdir_SOCS):
        print "Starting run.py with commands args: %s" % command_line
        run("./run.py %s" % command_line)
        print "done runinng..................."


def screen_run(command_line="--help"):
    """ same as above but runs the command using screen"""
    with cd(env.LDAdir_SOCS):
        print "Starting run.py with commands args: %s" % command_line
        run('/usr/bin/screen -S ldarun -d -m bash -c "ls; ./run.py %s; sleep 10;"' % command_line)
        #run('/usr/bin/screen -S ldarun -d -m bash -i -c "ls; ./run.py %s; sleep 10;"' % command_line)
        #                                            \____-i makes bash read .bash_profile etc...
        # at first I wanted screen to keep running after the program terminates, but
        # on second though I am happy with the current setup, where screen dies after tasks have finished
        # running
        print "done runinng..................."


def ls_screen():
    """ Lists remote screen sessions (one called ldarun was started by screen_run) """
    with hide('warnings'):
        out = run('/usr/bin/screen -ls' )
        if "No Sockets found" in out:
            pass
        else:
            print out


def kill_screen(command_line="--help"):
    """ Kills the remote screen session called ldarun """
    with cd(env.LDAdir_SOCS):
        print "Starting run.py with commands args: %s" % command_line
        run('/usr/bin/screen -S ldarun -p 0 -X quit' )
        # at first I wanted screen to keep running after the program terminates, but
        # on second though I am happy with the current setup, where screen dies after tasks have finished
        # running
        print "done runinng..................."



# tasks

def uptime():
    """ run finger on each host """
    try:
        out=run("uptime")
        print env.host
        print env.host_string
        print out
    except:
        print "host %(host)s is unreachable" % env


def finger():
    """ run finger on each host """
    try:
        out=run("finger")
        print env.host
        print out
    except:
        print "host %(host)s is unreachable" % env


def bare_finger():
    """ run finger on each host """
    out=run("finger")
    print env.host
    print out


def nobody():
    """ run finger on each host """
    try:
        out=run("finger")
        if "No one logged on" in out:
            print "nobody on %(host)s" % env
    except:
        print "host %(host)s is unreachable" % env


