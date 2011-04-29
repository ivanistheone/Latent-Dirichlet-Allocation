

# learned from
# http://www.b-list.org/weblog/2006/jun/14/django-tips-template-context-processors/


def rel_to_root(request):
    """ This method injects the variable REL_TO_ROOT in the RequestContext
        being used to render a template.
        for a URL like /dir/subdir/file.html
        REL_TO_ROOT will be set to ../../
        assumptions:    absolute paths without server is used
                        site root is /   (i.e. website is not in a subdir, or if it is
                                          the the webserver rewrites the request before
                                          passing it to django )
    """
    req_path = request.path                 # ex: /dir/subdir/file.html
    fragments = req_path.split('/')         #  ['', 'dir', 'subdir', 'file.html']
    depth = len(fragments) -2               # 2
    if depth < 0:
        depth=0
    return {"REL_TO_ROOT":"../"*depth}



def tmve_constants(request):
    """ Inserts into context the following TMVE global variables:
            tmv_title
    """
    from tmve_app import tmve_settings

    pass
