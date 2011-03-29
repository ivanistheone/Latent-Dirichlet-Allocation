from django.conf.urls.defaults import *

import os

from django.contrib.staticfiles.urls import staticfiles_urlpatterns



# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    (r'^browse/doc-graph.html', 'django_tmve.tmve_app.views.doc_graph'),
    # Example:
    # (r'^quant_ph/', include('quant_ph.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    (r'^admin/', include(admin.site.urls)),
)



# static/
urlpatterns += staticfiles_urlpatterns()



# static serve for media (user uploads)

from django.conf import settings

if settings.DEBUG:
    urlpatterns += patterns('',
        (r'^media/(.*)$', 'django.views.static.serve', {'document_root': os.path.join(settings.PROJECT_ROOT, '..', 'media')}),
    )



