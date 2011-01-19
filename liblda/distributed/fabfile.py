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

# globals
env.project_name = 'Latent-Dirichlet-Allocation'
env.use_photologue = False # django-photologue gallery module



# environments

def localhost():
    "Use the local virtual server"
    env.hosts = ['127.0.0.1']
    env.user = 'ivan'
    env.path = '/Users/%(user)s/Documents/Projects/(project_name)s' % env
    env.virtualhost_path = env.path

def lab7():
    ""
    id_list = range(1,29)
    env.hosts = [ "lab7-"+str(h)+".cs.mcgill.ca" for h in id_list ]
    env.user = 'isavov'
    env.path = '/home/2002/isavov/LDA' # /var/www/%(project_name)s' % env
    env.virtualhost_path = env.path

# Fatal error: Timed out trying to connect to lab7-11.cs.mcgill.ca
# how do you make this robust even if some hosts are unreachable?

# tasks


def finger():
    """ run finger on each host """
    out=run("finger")
    if "No one logged on" in out:
        print "nobody on %(host)s" % env




def test():
    "Run the test suite and bail out if it fails"
    result = local("cd %(path)s; python manage.py test" % env) #, fail="abort")


# def setup():
#     """
#     Setup a fresh virtualenv as well as a few useful directories, then run
#     a full deployment
#     """
#     require('hosts', provided_by=[localhost,webserver])
#     require('path')
#     sudo('aptitude install -y python-setuptools')
#     sudo('easy_install pip')
#     sudo('pip install virtualenv')
#     sudo('aptitude install -y apache2-threaded')
#     sudo('aptitude install -y libapache2-mod-wsgi') # beware, outdated on hardy!
#     # we want to get rid of the default apache config
#     sudo('cd /etc/apache2/sites-available/; a2dissite default;', pty=True)
#     sudo('mkdir -p %(path)s; chown %(user)s:%(user)s %(path)s;' % env, pty=True)
#     run('ln -s %(path)s www;' % env, pty=True) # symlink web dir in home
#     with cd(env.path):
#         run('virtualenv .;' % env, pty=True)
#         run('mkdir logs; chmod a+w logs; mkdir releases; mkdir shared; mkdir packages;' % env, pty=True)
#         if env.use_photologue: run('mkdir photologue');
#         run('cd releases; ln -s . current; ln -s . previous;', pty=True)
#     deploy()
#
# def deploy():
#     """
#     Deploy the latest version of the site to the servers,
#     install any required third party modules,
#     install the virtual host and then restart the webserver
#     """
#     require('hosts', provided_by=[localhost,webserver])
#     require('path')
#     import time
#     env.release = time.strftime('%Y%m%d%H%M%S')
#     upload_tar_from_git()
#     install_requirements()
#     install_site()
#     symlink_current_release()
#     migrate()
#     restart_webserver()
#
# def deploy_version(version):
#     "Specify a specific version to be made live"
#     require('hosts', provided_by=[localhost,webserver])
#     require('path')
#     env.version = version
#     with cd(env.path):
#         run('rm releases/previous; mv releases/current releases/previous;', pty=True)
#         run('ln -s %(version)s releases/current' % env, pty=True)
#     restart_webserver()
#
# def rollback():
#     """
#     Limited rollback capability. Simple loads the previously current
#     version of the code. Rolling back again will swap between the two.
#     """
#     require('hosts', provided_by=[localhost,webserver])
#     require('path')
#     with cd(env.path):
#         run('mv releases/current releases/_previous;', pty=True)
#         run('mv releases/previous releases/current;', pty=True)
#         run('mv releases/_previous releases/previous;', pty=True)
#     restart_webserver()
#
# # Helpers. These are called by other functions rather than directly

# def upload_tar_from_git():
#     "Create an archive from the current Git master branch and upload it"
#     require('release', provided_by=[deploy, setup])
#     local('git archive --format=tar master | gzip > %(release)s.tar.gz' % env)
#     run('mkdir -p %(path)s/releases/%(release)s' % env, pty=True)
#     put('%(release)s.tar.gz' % env, '%(path)s/packages/' % env)
#     run('cd %(path)s/releases/%(release)s && tar zxf ../../packages/%(release)s.tar.gz' % env, pty=True)
#     local('rm %(release)s.tar.gz' % env)
#
# def install_site():
#     "Add the virtualhost file to apache"
#     require('release', provided_by=[deploy, setup])
#     #sudo('cd %(path)s/releases/%(release)s; cp %(project_name)s%(virtualhost_path)s%(project_name)s /etc/apache2/sites-available/' % env)
#     sudo('cd %(path)s/releases/%(release)s; cp vhost.conf /etc/apache2/sites-available/%(project_name)s' % env)
#     sudo('cd /etc/apache2/sites-available/; a2ensite %(project_name)s' % env, pty=True)
#
# def install_requirements():
#     "Install the required packages from the requirements file using pip"
#     require('release', provided_by=[deploy, setup])
#     run('cd %(path)s; pip install -E . -r ./releases/%(release)s/requirements.txt' % env, pty=True)
#
# def symlink_current_release():
#     "Symlink our current release"
#     require('release', provided_by=[deploy, setup])
#     with cd(env.path):
#         run('rm releases/previous; mv releases/current releases/previous;')
#         run('ln -s %(release)s releases/current' % env)
#         if env.use_photologue:
#             run('cd releases/current/%(project_name)s/static; rm -rf photologue; ln -s %(path)s/photologue;' % env, pty=True)
#
# def migrate():
#     "Update the database"
#     require('project_name')
#     run('cd %(path)s/releases/current/%(project_name)s;  ../../../bin/python manage.py syncdb --noinput' % env, pty=True)
#
# def restart_webserver():
#     "Restart the web server"
#     sudo('/etc/init.d/apache2 reload', pty=True)
