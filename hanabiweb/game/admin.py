from django.contrib import admin

from .models import *

admin.site.register(Entity)
admin.site.register(Team)
admin.site.register(AI)
admin.site.register(Player)
admin.site.register(GameSetup)
admin.site.register(Result)

