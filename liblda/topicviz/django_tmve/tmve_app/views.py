# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.template import RequestContext, Context, loader, Template



def doc_graph(request):
    """ shows a listing of documents (left pane) and their
        topic contents (viz via td's with width prop to topic content)
    """

    docs = [
             {"meta":{"title":"Long title... "},  "get_safe_title":"Safe_title", "title":"normal_title" },
             {"meta":{"title":"Long title2... "},  "get_safe_title":"S2afe_title", "title":"2normal_title" },
             {"meta":{"title":"Long title3... "},  "get_safe_title":"S3afe_title", "title":"3normal_title" }
             ]

    rtopics = [
                [ {"percent":40, "display_title":"dlkjdasldlkj display title", "link":"thelink", "id":3 },
                  {"percent":20, "display_title":"display title", "link":"thelink", "id":67 },
                  {"percent":40, "display_title":"dlkjdasldlkjtitle", "link":"thelink", "id":4 } ],
                [ {"percent":40, "display_title":"dlkjdasldlkj display title", "link":"thelink", "id":3 },
                  {"percent":20, "display_title":"display title", "link":"thelink", "id":67 },
                  {"percent":40, "display_title":"dlkjdasldlkjtitle", "link":"thelink", "id":4 } ],
                [ {"percent":40, "display_title":"dlkjdasldlkj display title", "link":"thelink", "id":3 },
                  {"percent":20, "display_title":"display title", "link":"thelink", "id":67 },
                  {"percent":40, "display_title":"dlkjdasldlkjtitle", "link":"thelink", "id":4 } ]
               ]

    t = loader.get_template('browse/doc-graph.html')
    c = RequestContext(request, {
           'document_list':docs,
           'doc_rtopic_list': rtopics,
                                        })
    return HttpResponse(t.render(c))


