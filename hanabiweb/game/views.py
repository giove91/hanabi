from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponse
from django.contrib.auth import logout
from django.views import View
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required, user_passes_test

from models import *



def logout_view(request):
    logout(request)
    return redirect('index')


class IndexView(View):
    template_name = 'index.html'
    
    def get(self, request, *args, **kwargs):
        context = {}
        return render(request, self.template_name, context)


