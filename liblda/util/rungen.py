#!/usr/bin/env python

from liblda.extlibs import argparse
import sys,os
import random
import re
import tarfile


PROJECT_PATH=os.path.realpath(os.path.join(os.path.dirname(__file__),"../.."))
PACK_DIR=os.path.realpath(os.path.join(os.path.dirname(__file__),"packs/"))
RUNDIRS_ROOT = os.path.join(PROJECT_PATH, "data/runs/")


def mk_next_rundir(rundir_root=None):
    """
    Make a sequentially numbered rundir.
    If no rundir_root is specified the global RUNDIRS_ROOT is used.
    Directories are created sequentially in the format 'run%04d'
    """
    if not rundir_root:
        rundir_root=RUNDIRS_ROOT

    contents = os.listdir(rundir_root)
    contents = sorted( contents )


    # get max directory number
    max = 0         # biggest rundir number
    count =0
    for filename in contents:
        if filename[0:3] != "run":
            continue
        count_str = filename[3:7]
        try:
            count = int(count_str)
            if count > max:
                max=count
        except ValueError:
            continue

    new = count+1 # (if empty starts at run0001)
    newname = "run%04d" % new
    rundir = os.path.join(rundir_root,newname)

    # mkdir it !
    try:
        os.mkdir(rundir)
    except OSError:
        print("ERROR: dir already exists...")

    # return full path
    return rundir






def symlink_datafile(datafile=None):
    """
    Look in pervious rundirs and symlink to the original
    docword.txt to save space.
    """
    pass

def rungen(name,run_path=RUNDIRS_ROOT, T=10, NITER=100, SEED=123):
    """ Creates all the necessary setup for 1 experimental run
        returns a dict of values
        { T
          NITER
          SEED
          runcommand
          wpfile
        }
        another"""


    dirname = "run"+str(name)+"_T"+str(T)+"_NITER"+str(NITER)+"_seed"+str(SEED)
    rundir = os.path.join(run_path,dirname)

    if os.path.exists(rundir):
        print "Error -- rundir already exists... exiting"
        return None

    # create dir
    os.mkdir(rundir)
    os.chdir(rundir)

    # prepare files
    packfile = os.path.join(PACK_DIR, sys.platform+"-run-pack.tar")
    tar = tarfile.open(packfile)
    tar.extractall()
    tar.close()


    runcommand = rundir+"/topicmodel "+str(T)+" "+str(NITER)+" "+str(SEED)
    wpfile = rundir+"/wp.txt"
    dpfile = rundir+"/dp.txt"

    # the dict we will be returning
    rundict ={}
    rundict["name"]=name
    rundict["T"]=T
    rundict["NITER"]=NITER
    rundict["SEED"]=SEED
    rundict["rundir"]=rundir
    rundict["runcommand"]=runcommand
    rundict["wpfile"]=wpfile
    rundict["dpfile"]=dpfile


    return rundict




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepares a run" )
    parser.add_argument('--T')
    parser.add_argument('--RUNDIRS_ROOT')
    parser.add_argument('--NITER')
    parser.add_argument('--SEED')
    parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args()

    kwargs ={}
    if args.RUNDIRS_ROOT:
        kwargs["run_root"]=args.RUNDIRS_ROOT
    if args.T:
        kwargs["T"]=args.T
    if args.NITER:
        kwargs["NITER"]=args.NITER
    if args.SEED:
        kwargs["SEED"]=args.SEED


    print rungen("tmp",**kwargs)
    #run_path=args.RUN_ROOT, T=args.T, NITER=args.NITER, SEED=args.SEED)




