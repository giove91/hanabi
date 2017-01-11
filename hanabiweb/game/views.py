from django.shortcuts import render
from django.http import HttpResponse

from django.views import View



class IndexView(View):
    template_name = 'index.html'
    
    def get(self, request, *args, **kwargs):
        context = {}
        return render(request, self.template_name, context)


